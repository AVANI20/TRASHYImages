import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, models, transforms
from torchvision.transforms import transforms
import os
import pandas as pd
import shutil
import sys
from picamera2 import Picamera2, Preview
import time
from time import sleep
from datetime import datetime
import RPi.GPIO as GPIO
from gpiozero import Button, LED, PWMLED
from pathlib import Path
#userPath = os.path.expanduser("~")
dest = "/home/trashypi/Trashy"
source = "/tmp/trashy/"

#Assign GPIO pins
button = Button(2)

yellow_led = LED(17) #yellowbin
blue_led = LED(27) #bluebin
green_led = LED(22) #blackbin
red_led = PWMLED(23) #glassbin
white_led = PWMLED(24) #trash

# Start the camera preview
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)

def save():
    dest = "/home/trashypi/Trashy/trashyDataset/pred"
    source = "/tmp/trashy/"
    for file_name in os.listdir(source):
        source = source + file_name
        for root, subfolders, filenames in os.walk(dest):
            for i in subfolders:
                filepath = root + "/" + i
                shutil.copy(source,filepath)
        print("copied")

def take_photo():
    print("Pressed")
    white_led.on()
    picam2.start_preview(Preview.QTGL)
    #timestamp = datetime.now().isoformat()
    picam2.start()
    time.sleep(5)
    picam2.capture_file("/tmp/trashy/img.jpg")
    picam2.stop_preview()
    picam2.stop()
    save()
    white_led.off()
    time.sleep(1)
    
def augment():
    ''' Function that sets up data augmentation transforms.
    After loading the data into memory, can call this function to get the transforms and apply
    them to the data.
    '''
    # Data augmentation and normalization for training
    # Just normalization for validation and test sets
    data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5417, 0.5311, 0.5700], [0.7856, 0.7939, 0.8158])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        ]),
        'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5417, 0.5311, 0.5700], [0.7856, 0.7939, 0.8158])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5417, 0.5311, 0.5700], [0.7856, 0.7939, 0.8158])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        'pred': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5417, 0.5311, 0.5700], [0.7856, 0.7939, 0.8158])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def load_data(outDataPath):
    ''' Function to load the  data from the given path
    Aplies the datatransforms given via augment() and creates and returns
    dataloader objects for the train and val datasets, the sizes of the 
    datasets, and  the list of classnames'''
    
    # Get data transforms
    data_transforms = augment()
    
    # Create an ImageFolder dataloader for the input data
    # See https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(outDataPath, x),
                                              data_transforms[x]) for x in ['train', 'val', 'test', 'pred']}

        
    # Create DataLoader objects for each of the image datasets returned by ImageFolder
    # See https://pytorch.org/docs/stable/data.html
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=4) for x in ['train', 'val', 'test', 'pred']}
    
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test', 'pred']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, datasets_sizes, class_names

def create_grid_for_mb(i, inputs, num_images, class_names, preds, labels):
    ''' Creates a grid showing predicted and ground truth labels for subset of images of a minibatch.
        Params:
             -  i:               the  minibatch number 
             -  inputs:          images
             -  num_images:      number of images to plot in the grid; height and width of grid are np.sqrt(num_images)
             -  class_names:     class labels
             -  preds:           model predictions 
             -  labels:          ground truth labels
    '''
    images_so_far = 0
    
    for j in range(inputs.size()[0]):
        images_so_far += 1
        
        if images_so_far >= num_images:
            break  
        
    return class_names[preds[j]], class_names[labels[j]]    


def led_select(label):
    print("Throw trash in this Waste bin:", label)
    if label == "yellow bin":
        yellow_led.on()
        time.sleep(5)
    if label == "blue bin":
        blue_led.on()
        time.sleep(5)
    if label == "black bin":
        green_led.on()
        time.sleep(5)
    if label == "glass bin":
        red_led.on()
        time.sleep(5)
    if label == "electronics":
        white_led.on()
        time.sleep(5)
    else:
        yellow_led.off()
        blue_led.off()
        green_led.off()
        red_led.off()
        white_led.off()
        
def remove():
    dest = "/home/trashypi/Trashy/trashyDataset/pred"
    for root, subfolders, filenames in os.walk(dest):
        for filename in filenames:
            filepath = root + "/" + filename
            os.remove(filepath)
    print("deleted")


class VGG(object):

    def __init__(self, pretrained_model, device, num_classes=25, lr=0.0001, reg=0.0, dtype=np.float32, mode="ft_extract"):
        self.params = {}
        self.reg = reg
        self.dtype = dtype 
        self.model = pretrained_model
        self.num_classes = num_classes
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

        self.set_parameter_requires_grad(mode)
        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1]                  
        features.extend([nn.Linear(num_features, num_classes).to(self.device)]) 
        self.model.classifier = nn.Sequential(*features)            
                            
    def set_parameter_requires_grad(self, mode):
        if mode == "ft_extract":
            for param in self.model.features.parameters():
                param.requires_grad = False
        elif mode == "finetune_last":
            for param in self.model.features[:19].parameters():
                param.requires_grad = False
        
                
    def gather_optimizable_params(self):
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_optimize.append(param)

        return params_to_optimize

                
                
    def load_model(self, path, train_mode = False):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

        if train_mode == False:
            self.model.eval()

        return self.model


    def visualize_model(self, dataloaders, num_images=25):
        self.model.train(False)
        self.model.eval()
        
        images_so_far = 0
                                                   
        with torch.no_grad():
            for i, data in enumerate(dataloaders['pred']):
                inputs, labels = data
                size = inputs.size()[0]
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = self.model(inputs)                
                _, preds = torch.max(outputs, 1)
                    
                predict, actual = create_grid_for_mb(i, inputs, num_images, class_names, preds, labels)
                return predict



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT').to(device)
    vgg_model = VGG(vgg19, device, num_classes=25)
    print("initial")
    vgg_model.load_model('/home/trashypi/Trashy/trashyModel_VGG7.pt', train_mode = False)
    print("model")
    buttonState = GPIO.input(2)
    while buttonState == True:
        print("Press the button to capture image...")
        # Quickly blink status light
        white_led.blink(0.1,0.1)
        time.sleep(2)
        button.wait_for_press()
        take_photo()
        # Run photo through VGG19 model
        pathname = '/home/trashypi/Trashy/trashyDataset'
        dataloaders, dataset_sizes, class_names = load_data(pathname)
        print("loaded data")
        predictedClass = vgg_model.visualize_model(dataloaders, num_images=25)
        print("Predicted class of trash: ", predictedClass)
        if predictedClass == "beverage cans" or predictedClass == "metal containers" or predictedClass == "other metal objects" or predictedClass == "spray cans" or predictedClass == "tetra pak" or predictedClass == "paper cups" or predictedClass == "plastic bags" or predictedClass == "plastic bottles" or predictedClass == "plastic containers" or predictedClass == "plastic cups":
            label = "yellow bin"
        elif predictedClass == "cardboard" or predictedClass == "news paper" or predictedClass == "paper":
            label = "blue bin"
        elif predictedClass == "cigarette butt" or predictedClass == "gloves" or predictedClass == "masks" or predictedClass == "medicines" or predictedClass == "syringe":
            label = "black bin"
        elif predictedClass == "glass":
            label = "glass bin"
        elif predictedClass == "construction scrap" or predictedClass == "electrical cables" or predictedClass == "electronic chips" or predictedClass == "smartphones" or predictedClass == "laptops" or predictedClass == "small appliances":
            label = "electronics"
        else:
            print("Wrong prediction")
        led_select(label)
        print("Would you like to capture another image?\nType input: (yes/no)")
        userInput = str(input())
        if userInput == "yes":
            time.sleep(1)
            continue
        elif userInput == "no":
            break
        else:
            # Pulse status light
            print("Wrong input..Cannot capture image..")
            white_led.pulse(2,1)
            break
        time.sleep(1)
    remove()
