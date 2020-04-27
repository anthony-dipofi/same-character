import os

import torch
from PIL import Image
import torchvision
import random
import multiprocessing

#folder = "d1k/d1k/"

def preprocess_image(img):
    #img = img.resize(size = (128,128))
    
    transforms = torchvision.transforms.Compose(
        [
            #torchvision.transforms.RandomAffine(15),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            #torchvision.transforms.ColorJitter(brightness=0.1, 
            #                                   contrast=0.1, 
            #                                   saturation=0.1, 
            #                                   hue = 0.1),
            #torchvision.transforms.RandomGrayscale(p=0.1),
            #torchvision.transforms.RandomResizedCrop(512, 
            #                                         scale=(0.6, 0.9), 
            #                                         ratio=(0.75, 1.3333), 
            #                                         interpolation=2),
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor()

        ]
    )
    img = transforms(img)
    return img
    

def load_image(filename):
    img = Image.open(filename)
    img = preprocess_image(img)
    
    return img

def sample_image(folder):    
    files = os.listdir(folder)
    img_index = random.randint(0,len(files)-1)
    img_filename = folder +"/" + files[img_index]
    img = load_image(img_filename)
    
    return img
    
def sample_example(folder):
    dirs = os.listdir(folder)
    
    y = random.randint(0,1)
    
    num_classes = len(dirs)
    
    if y == 1:
        img_1_class_idx = random.randint(0, num_classes-1)
        img_2_class_idx = img_1_class_idx
        fldr = folder + dirs[img_1_class_idx]
        img_1 = sample_image(fldr)
        img_2 = sample_image(fldr)
    
       
    if y == 0:
        img_1_class_idx = random.randint(0, num_classes-1)
        img_2_class_idx = random.randint(0, num_classes-1)
        while img_1_class_idx == img_2_class_idx:
            img_2_class_idx = random.randint(0, num_classes-1)
            
        fldr_1 = folder + dirs[img_1_class_idx]
        fldr_2 = folder + dirs[img_2_class_idx]
        img_1 = sample_image(fldr_1)
        img_2 = sample_image(fldr_2)
          
    return {'img_1': img_1, 'img_2':img_2, 'y': y, 'img_1_class': img_1_class_idx, 'img_2_class': img_2_class_idx}


def sample_batch(batch_size, folder):
    exs = []
    for i in range(batch_size):
        ex = sample_example(folder)
        #ex = sample_example_class()
        exs.append(ex)
        
    img_1 = torch.stack([x['img_1'] for x in exs], dim = 0)
    img_2 = torch.stack([x['img_2'] for x in exs], dim = 0)
    y = torch.tensor([x['y'] for x in exs])
    img_1_class = torch.tensor([x['img_1_class'] for x in exs])
    img_2_class = torch.tensor([x['img_2_class'] for x in exs])
    
    return {'img_1': img_1, 'img_2':img_2, 'y': y, 'img_1_class': img_1_class, 'img_2_class': img_2_class}

def sampler_proc(queue, batch_size, folder):
    while True:
        try:
            batch = sample_batch(batch_size, folder)
        except Exception:
            continue
        queue.put(batch)

class batch_sampler():
    def __init__(self, folder, batch_size, num_procs=1, max_queue_size = 10):
        self.folder = folder
        self.num_procs = num_procs
        self.batch_size = batch_size
        
        self.batch_queue = multiprocessing.Queue(max_queue_size)
                
        self.procs = []
        for i in range(num_procs):
            new_p = multiprocessing.Process(target=sampler_proc, args = (self.batch_queue, batch_size, folder))
            new_p.start()
            self.procs.append(new_p)
            
    def next(self):
        return self.batch_queue.get()     
    
    def close(self):
        for p in self.procs:
            p.terminate()
            