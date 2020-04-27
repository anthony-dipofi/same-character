import os

import numpy as np
import cv2
import torch
from PIL import Image
import torchvision
#from torchvision.transforms import ToTensor, ToPILImage
import random
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import torch.optim as optim
import math
from functools import reduce

def coordConv(sample):
    shape = sample.shape
        
    grid = np.mgrid[0:0.1*shape[-2]:0.1,0:0.1*shape[-2]:0.1]

    tgrid = torch.tensor(grid).float()

    s = torch.stack([tgrid]*shape[0],dim=0)

    return s

class Conv2dResBlock(nn.Module):
    def __init__(self, res_depth, bottleneck_depth, stride = (1,1)):
        super(Conv2dResBlock,self).__init__()
        self.stride = stride
        
        self.res_depth = res_depth

        self.conv1 = nn.Conv2d( res_depth,  res_depth, kernel_size=(1, 1), stride = (1,1))
        self.conv2 = nn.Conv2d( res_depth,  bottleneck_depth, (3, 3), stride = stride, padding = (1,1))
        self.conv3 = nn.Conv2d( bottleneck_depth,  res_depth, kernel_size=(1, 1), stride = (1,1))

        if(self.stride != (1,1)):
            self.conv_res = nn.Conv2d(res_depth, res_depth, kernel_size = (3,3), stride = stride, padding = (1,1))
        
        for i in range(1, len(list(self.modules()))):
            list(self.modules())[i].weight.data.normal_(0.0, 0.3)

    def forward(self, ins):
        v = ins
        
        v = v.permute([0,2,3,1])
        v = F.layer_norm(v, [self.res_depth])
        v = v.permute([0,3,1,2])

        v = F.leaky_relu(self.conv1(v))
        v = F.leaky_relu(self.conv2(v))
        v = self.conv3(v)

        if self.stride == (1,1):
            res = ins
        else:
            res = self.conv_res(ins)

        v = v + res

        return v

class AttentionBlock(nn.Module):
    def __init__(self, model_features, num_heads, dropout_prob = 0.5):
        super(AttentionBlock,self).__init__()
        self.num_heads = num_heads
        self.model_features = model_features
        
        self.feature_dropout = nn.Dropout(p=dropout_prob, inplace=False)
        
        self.attention = nn.MultiheadAttention(embed_dim = model_features, num_heads = self.num_heads)

        self.pos_mlp1 = nn.Conv1d(model_features, model_features, kernel_size=1,stride=1)
        self.pos_mlp2 = nn.Conv1d(model_features, model_features, kernel_size=1,stride=1)
        
    def forward(self, query, key, value):
        res_v = query
        
        #print("q",query.shape)
        #print("k",key.shape)
        #print("v",value.shape)

        query_v = F.layer_norm(query,[self.model_features])        
        key_v = F.layer_norm(key,[self.model_features])
        value_v = F.layer_norm(value,[self.model_features])
        
        att_v = self.attention(query_v, key_v, value_v)

        att_v = self.feature_dropout(att_v[0])
        
        v = res_v + att_v

        #Position-Wise MLP
        res_v = v

        v = F.layer_norm(v,[self.model_features]).permute(1,2,0).contiguous()
        
        mlp_v = F.relu(self.pos_mlp1(v))
        mlp_v = self.pos_mlp2(mlp_v).permute(2,0,1).contiguous()

        mlp_v = self.feature_dropout(mlp_v)
        
        v = res_v + mlp_v

        return v
        
        
class clfr2_wAux(nn.Module):
    def __init__(self,  
                 res_strides=(2,2,2,2,2),
                 res_depth = 64,
                 classes = 1000,
                 model_features=128, 
                 num_attnblocks=6,
                 num_heads = 4,
                 in_features = 3,
                 out_classes = 2,
                 input_size = 256,
                 dropout_prob = 0.5):
        
        super(clfr2_wAux, self).__init__()
        
        self.res_depth = res_depth
        self.model_features = model_features
        self.model_channels = 1
        self.features = model_features
        self.out_classes = out_classes
        self.num_heads = num_heads
        self.classes = classes
        
        
        self.num_attnblocks = num_attnblocks
        
        self.coord_init = nn.Conv2d( 2,  model_features, kernel_size = (1, 1), stride=(1, 1))
        
        conv_out_features = int((input_size/reduce(lambda x,y:x*y, res_strides))**2)
        self.conv_init = nn.Conv2d( in_features, res_depth,kernel_size = (1,1), stride = (1,1))
        
        self.resblocks = nn.ModuleList()
        for i in range(0, len(res_strides)):
            self.resblocks.append(Conv2dResBlock(res_depth, res_depth*2,  stride = res_strides[i]))
                        
        self.res_out = nn.Conv2d(res_depth, model_features, kernel_size = (1,1), stride=(1,1))
        self.res_out_clasf = nn.Conv2d(res_depth, classes, kernel_size = (1,1), stride=(1,1))
                
        self.initial_embed = nn.Parameter(
            torch.tensor(
                np.random.normal(
                    loc = 0, 
                    scale = 0.01, 
                    size = (1, self.model_channels, model_features)
                )
            ).float()
        )
        
        self.channel_dropout = nn.Dropout2d(p=dropout_prob, inplace = False)
        self.feature_dropout = nn.Dropout(p=dropout_prob, inplace = False)
        
        
        self.attn_blocks = nn.ModuleList()
        for i in range(num_attnblocks):
            self.attn_blocks.append(AttentionBlock(model_features = model_features, num_heads = self.num_heads, dropout_prob=dropout_prob))
        

        self.out_conv = nn.Conv1d(model_features, out_classes, kernel_size=1,stride=1)


    def forward(self, ins):

        #load images from batch
        im1_v = ins['img_1']
        im2_v = ins['img_2']

        #initialize cuda for images
        im1_v = im1_v.cuda()
        im2_v = im2_v.cuda()
        
        #store batch size
        batch_size = len(im1_v)
        
        #concatenate images along batch dimension to pass both images through the resnet in parrallel
        im_v = torch.cat([im1_v, im2_v], dim = 0)
        
        #pass images through resnet
        v = F.leaky_relu(self.conv_init(im_v))        
        for resblock in self.resblocks:
            v = resblock(v)

        classes_v = self.res_out_clasf(v)
        im1_classes_v = classes_v[:batch_size]
        im1_classes_v = im1_classes_v.view([batch_size, self.classes, -1]).permute((2,0,1)).contiguous()
        im1_classes_v = F.layer_norm(im1_classes_v, [self.classes]).mean(dim=0)

        im1_classes_v = torch.clamp(im1_classes_v, -10, 10)
        im1_classes_v = F.log_softmax(im1_classes_v, dim=1)

        im2_classes_v = classes_v[batch_size:]
        im2_classes_v = im2_classes_v.view([batch_size, self.classes, -1]).permute((2,0,1)).contiguous()
        im2_classes_v = F.layer_norm(im2_classes_v, [self.classes]).mean(dim=0)
        
        im2_classes_v = torch.clamp(im2_classes_v, -10, 10)
        im2_classes_v = F.log_softmax(im2_classes_v, dim=1)

        
        #project resnet outputs to model dimension
        input_v = self.res_out(v)
        
        #initialize coords and project to model dimensions
        coord = self.coord_init(coordConv(input_v).cuda())

        #add coordinate embedings to resnet output
        input_v = coord + input_v
        
        #seperate images
        im1_v = v[:batch_size]
        im2_v = v[batch_size:]

        #reshape, normalize and dropout the encodings for each image    
        im1_v = im1_v.view([batch_size, self.model_features, -1]).permute((2,0,1)).contiguous()
        im1_v = F.layer_norm(im1_v, [self.model_features])
        im1_v = self.channel_dropout(im1_v)
        im1_v = self.feature_dropout(im1_v)

        im2_v = im2_v.view([batch_size, self.model_features, -1]).permute((2,0,1)).contiguous()
        im2_v = F.layer_norm(im2_v, [self.model_features])
        im2_v = self.channel_dropout(im2_v)
        im2_v = self.feature_dropout(im2_v)

        im_v = torch.cat([im1_v, im2_v], dim = 0)

        #tile learned initial embedding along batch dimension
        v = self.initial_embed.repeat([1, batch_size, 1])
        v = self.feature_dropout(v)
        v = F.layer_norm(v, [self.model_features])

        #loop through attention blocks
        for i in range(self.num_attnblocks):            
            query = v
            key = im_v
            value = im_v
            v = self.attn_blocks[i](query = query, key = key, value = value)

           
        #normalize output of attenetion blocks    
        v = F.layer_norm(v,[self.features]).permute([1,2,0])
        

        out_v = self.out_conv(v)
        v = torch.clamp(v, -10, 10)

        out_v = F.log_softmax(out_v, dim=1)

        return {"out": out_v, "im1_class": im1_classes_v, "im2_class": im2_classes_v}
        