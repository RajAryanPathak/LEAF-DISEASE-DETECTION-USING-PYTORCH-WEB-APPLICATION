from django.shortcuts import render, redirect, HttpResponse
from django.core.files.storage import FileSystemStorage
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import plotly
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models as m
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import json
import os
from os.path import exists
from django.contrib.auth import login as l, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

from .forms import *
from .models import *
import wikipedia
with open('./models/categories.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
def load_checkpoint(filepath):
    #checkpoint = torch.load()
    checkpoint = torch.load(filepath, map_location=map_location)

    model = m.resnet152()
    
    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 39
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']


path = F"./models/plants9615_checkpoint.pth" 
# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint(path)
idx_to_class = { v : k for k,v in class_to_idx.items()}


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    return npImage



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class


# Display an image along with the top 5 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    

    fig, ax2 = plt.subplots()
    
    

    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes],rotation = 45)
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()

    ax2.invert_yaxis()
    plt.savefig('media/books_read.png')


def seg(fna):
    
    import numpy as np 
    import cv2
    # load image
    image = cv2.imread(fna)
    # create hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # set lower and upper color limits
    low_val = (0,60,0)
    high_val = (179,255,255)
    # Threshold the HSV image 
    mask = cv2.inRange(hsv, low_val,high_val)
    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
    # apply mask to original image
    result = cv2.bitwise_and(image, image,mask=mask)


    zz = copy.copy(result)
    ## convert to hsv
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask>0
    img = np.zeros_like(result, np.uint8)

    img[imask] = result[imask]



    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    value = 90

    vValue = hsvImg[...,2]
    hsvImg[...,2]=np.where((255-vValue)<value,255,vValue+value)

    g1=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

    g1=cv2.cvtColor(g1,cv2.COLOR_RGB2BGR)

    hsv1 = cv2.cvtColor(g1, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask1 = cv2.inRange(hsv1, (36, 25, 25), (70, 255,255))



    imask1 = mask1>0
    green = result
    green[imask1] = g1[imask1]




    b,g,r=cv2.split(green)
    gtbgr =cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


    row, column = g.shape
    # Create an zeros array to store the sliced image
    g12 = np.zeros((row,column),dtype = 'uint8')
    
    # Specify the min and max range
    min_range = 0
    max_range = 120
    
    # Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
    for i in range(row):
        for j in range(column):
            if g[i,j]>min_range and g[i,j]<max_range:
                g12[i,j] = 0
            elif(g[i,j]>120 and g[i,j]<250):
                g12[i,j] = 200
                
            else:
                g12[i,j] = 255
    # Display the image
    cv2.imwrite(r'C:\Users\Admin\Desktop\leafcopy\media\a.jpg',g12)
    return  './media/a.jpg'


def login(request):
        if request.method == 'POST':
            print('post')
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                l(request, user)
                # Redirect to a success page.
                print(type(user))
                return redirect('index')
            else:
                return render(request, 'login.html')

        else:
            return render(request, 'login.html')
    
def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            l(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def index(request):
    return render(request, 'index.html')



def predictimage(request):
    
    fileObj = request.FILES['filepath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    
    fna = '.'+filePathName

    p, c = predict(fna, loaded_model)
    print(p,c)
    fig  = view_classify(fna, p, c, cat_to_name)
    print(fna)
    fig1  = seg(fna)
    print(fig1)
    context = {'filePathName':filePathName,'fig':'./media/books_read.png','fig1':fig1, 'c':c[0]}
    print(fig)
    return render(request,'show.html',context) 


def forum(request):
    if request.method == 'POST':
        temp = Forum(user=request.user)
        temp.save()
        # print(temp.id,Blog.objects.get(id=temp.id))

        form4 = WriteBlog(request.POST,instance=Forum.objects.get(id=temp.id))
        print(form4.errors)
        if form4.is_valid():
            print('valid')

            print('save')
            form4.save()
            return redirect('index')
    else:
        form4 = WriteBlog()
        a={'form4':form4}
        return render(request, 'forum.html',a)


def wikip(request):
    if request.method == 'POST':
            print('post')
            username = request.POST.get('username')
            try:
                result = wikipedia.summary(username, sentences = 2)  
            except:
                result=wikipedia.search(username, results = 5) 
            return render(request, 'wikip.html',{'result':result})
    else:
        return render(request, 'wikip.html')
    
def logout_view(request):
    logout(request)
    return redirect('login')