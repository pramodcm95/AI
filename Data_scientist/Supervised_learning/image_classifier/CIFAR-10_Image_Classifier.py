#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this project, you will build a neural network of your own design to evaluate the CIFAR-10 dataset.
# 
# To meet the requirements for this project, you will need to achieve an accuracy greater than 45%. 
# If you want to beat Detectocorp's algorithm, you'll need to achieve an accuracy greater than 70%. 
# (Beating Detectocorp's algorithm is not a requirement for passing this project, but you're encouraged to try!)
# 
# Some of the benchmark results on CIFAR-10 include:
# 
# 78.9% Accuracy | [Deep Belief Networks; Krizhevsky, 2010](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf)
# 
# 90.6% Accuracy | [Maxout Networks; Goodfellow et al., 2013](https://arxiv.org/pdf/1302.4389.pdf)
# 
# 96.0% Accuracy | [Wide Residual Networks; Zagoruyko et al., 2016](https://arxiv.org/pdf/1605.07146.pdf)
# 
# 99.0% Accuracy | [GPipe; Huang et al., 2018](https://arxiv.org/pdf/1811.06965.pdf)
# 
# 98.5% Accuracy | [Rethinking Recurrent Neural Networks and other Improvements for ImageClassification; Nguyen et al., 2020](https://arxiv.org/pdf/2007.15161.pdf)
# 
# Research with this dataset is ongoing. Notably, many of these networks are quite large and quite expensive to train. 
# 
# ## Imports

# In[1]:


## This cell contains the essential imports you will need – DO NOT CHANGE THE CONTENTS! ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ## Load the Dataset
# 
# Specify your transforms as a list first.
# The transforms module is already loaded as `transforms`.
# 
# CIFAR-10 is fortunately included in the torchvision module.
# Then, you can create your dataset using the `CIFAR10` object from `torchvision.datasets` ([the documentation is available here](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)).
# Make sure to specify `download=True`! 
# 
# Once your dataset is created, you'll also need to define a `DataLoader` from the `torch.utils.data` module for both the train and the test set.

# In[2]:


import torch
from torchvision import datasets, transforms
# Define transforms
## YOUR CODE HERE ##
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.ColorJitter()
     ])
# Create training set and define training dataloader
## YOUR CODE HERE ##
trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Create test set and define test dataloader
## YOUR CODE HERE ##
testset = datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# The 10 classes in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:





# ## Explore the Dataset
# Using matplotlib, numpy, and torch, explore the dimensions of your data.
# 
# You can view images using the `show5` function defined below – it takes a data loader as an argument.
# Remember that normalized images will look really weird to you! You may want to try changing your transforms to view images.
# Typically using no transforms other than `toTensor()` works well for viewing – but not as well for training your network.
# If `show5` doesn't work, go back and check your code for creating your data loaders and your training/test sets.

# In[3]:


def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(classes[labels[i]])
    
        image = images[i].numpy()
        plt.imshow(np.rot90(image.T, k=3))
        plt.show()


# In[4]:


torch.cuda.is_available()


# In[5]:


# Explore data
## YOUR CODE HERE ##
show5(trainloader)


# ## Build your Neural Network
# Using the layers in `torch.nn` (which has been imported as `nn`) and the `torch.nn.functional` module (imported as `F`), construct a neural network based on the parameters of the dataset. 
# Feel free to construct a model of any architecture – feedforward, convolutional, or even something more advanced!

# In[6]:


## YOUR CODE HERE ##

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x


# Specify a loss function and an optimizer, and instantiate the model.
# 
# If you use a less common loss function, please note why you chose that loss function in a comment.

# In[7]:


## YOUR CODE HERE ##
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ## Running your Neural Network
# Use whatever method you like to train your neural network, and ensure you record the average loss at each epoch. 
# Don't forget to use `torch.device()` and the `.to()` method for both your model and your data if you are using GPU!
# 
# If you want to print your loss during each epoch, you can use the `enumerate` function and print the loss after a set number of batches. 250 batches works well for most people!

# In[8]:


## YOUR CODE HERE ##
# TODO: Train the network here
epochs = 25
training_loss_per_epoch = []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss}")
        training_loss_per_epoch.append(running_loss/len(trainloader))


# Plot the training loss (and validation loss/accuracy, if recorded).

# In[9]:


## YOUR CODE HERE ##
plt.plot(training_loss_per_epoch)
plt.tight_layout()
plt.grid()
plt.title('Training loss v/s Epochs')
plt.ylabel('Training loss')
plt.xlabel('Epochs')


# ## Testing your model
# Using the previously created `DataLoader` for the test set, compute the percentage of correct predictions using the highest probability prediction. 
# 
# If your accuracy is over 70%, great work! 
# This is a hard task to exceed 70% on.
# 
# If your accuracy is under 45%, you'll need to make improvements.
# Go back and check your model architecture, loss function, and optimizer to make sure they're appropriate for an image classification task.

# In[10]:


"""
dataiter = iter(testloader)
images, labels = dataiter.next()
#img = images[1]

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(images))

top_p, top_class = ps.topk(1, dim=1)
equals = top_class == labels.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')
"""
## YOUR CODE HERE ##
test_acc = []
for images, labels in testloader:
    ps = torch.exp(model(images))

    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_acc.append(accuracy.item()*100)
    
print(f'Accuracy: {sum(test_acc)/len(test_acc)}%')


# ## Saving your model
# Using `torch.save`, save your model for future loading.

# In[11]:


## YOUR CODE HERE ##
checkpoint = {'input_size': 3072,
              'output_size': 10,
              'hidden_layers': [128,64,32],
              'state_dict': model.state_dict()}

torch.save(model.state_dict(), 'checkpoint.pth')


# ## Make a Recommendation
# 
# Based on your evaluation, what is your recommendation on whether to build or buy? Explain your reasoning below.
# 
# Some things to consider as you formulate your recommendation:
# * How does your model compare to Detectocorp's model?
# * How does it compare to the far more advanced solutions in the literature? 
# * What did you do to get the accuracy you achieved? 
# * Is it necessary to improve this accuracy? If so, what sort of work would be involved in improving it?

# 
# 1) This solution doesn't overcome Detectocorp's model, it can be significatly improved
# 
# 2) IT is almost half of it's performance
# 
# 3) Since I am in a rush aginst time(due to my large unavailability due to health concerns), I am initially trying to finish all projects to minimum level(to pass) and then go for beautification). Here, I got my results by tuning two important hyperparameters, learning rate and network size
# 
# 4) Yes this can be improved significantly, a feed forward fully connected network may not be sufficient in this case, may we can switch towards CNN, Resnet's and others for better accuracy. Changing loss function and activation functions.(highly data dependent)
# 
# 

# ## Submit Your Project
# 
# When you are finished editing the notebook and are ready to turn it in, simply click the **SUBMIT PROJECT** button in the lower right.
# 
# Once you submit your project, we'll review your work and give you feedback if there's anything that you need to work on. If you'd like to see the exact points that your reviewer will check for when looking at your work, you can have a look over the project [rubric](https://review.udacity.com/#!/rubrics/3077/view).
