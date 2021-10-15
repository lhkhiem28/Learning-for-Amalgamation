"""Main script for Maximal Correlation
"""

mode = "cifar" #options are "cifar," "dogs," "tiny_imagenet"
num_source_samps = 500 #recommend 500 for Cifar, 50 for Dogs, and 500 for tiny_imagenet
num_target_samps = 1

if mode == "cifar":
    num_classes = 2
elif mode == "dogs":
    num_classes = 5
elif mode == "tiny_imagenet":
    num_classes = 5
else:
     raise Exception('Invalid dataset type')


import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import nets
import datasets


def compute_max_corr(net,data,labels,num_classes):
    """Computes maximal correlation and also returns the associated g(y)"""
    outputs = net(data)
    outputs -= outputs.mean(dim=0)
    outputs /= get_std_devs(net,data)
    g_y = torch.zeros((num_classes,outputs.shape[1]))
    for idx,row in enumerate(outputs.split(1)):
        g_y[labels[idx]] += row.detach().reshape(-1)
    g_y/=labels.shape[0]
    sigma = torch.zeros(outputs.shape[1])
    for idx,row in enumerate(outputs.split(1)):
        sigma += row.detach().reshape(-1)*g_y[labels[idx],:]
    sigma/=labels.shape[0]
    #make sure signs are positive
    g_y *= sigma.sign()
    sigma *= sigma.sign()
    return sigma, g_y

def weighted_network_output(net,sigma,g,inputs):
    """Output weighted sum(sigma*f*g) for both values of g"""
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    outputs *= sigma.reshape(1,-1)
    outputs= torch.mm(outputs, g.permute(1,0))
    return outputs

def get_means(net,inputs):
    outputs = net(inputs)
    return outputs.mean(dim=0)

def get_std_devs(net,inputs):
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    stds = torch.sqrt(torch.diag(torch.mm(outputs.permute(1,0),outputs)))
    stds[stds==0] = 1

    return stds

def compute_square_difference(net,means,stds,g,inputs,target_class):
    """Output sum((f - g)^2) for target value of g"""
    outputs = net(inputs)
    outputs -= means
    difference = torch.sum(outputs*g[target_class,:],dim=1)#torch.sum((outputs - g[target_class,:])**2)
    return difference

def lookup_classes(class_list):
    path = os.path.join("datasets","tiny-imagenet-200","words.txt")
    with open(path) as f:
        d = dict(x.rstrip().split(None, 1) for x in f)
    return [d[x] for x in class_list]

print("Extracting datasets...")
trainloader_source, trainloader_target, testloader = datasets.generate_dataset(mode, num_source_samps, num_target_samps)

all_nets = []
all_criteria = []
all_optimizers = []
for i in range(len(trainloader_source)):
    all_nets.append(nets.generate_net(mode)) #net[i][0] is just the first part up to the penultimate layer
    all_criteria.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.SGD(all_nets[i].parameters(), lr=0.001, momentum=0.9))

print("Training networks...")

for epoch in range(1, 101):  # loop over the dataset multiple times
    for j in range(len(trainloader_source)):
        for i, data in enumerate(trainloader_source[j], 0):
            # get the inputs
            inputs, labels = data
    
            # zero the parameter gradients
            all_optimizers[j].zero_grad()

            # forward + backward + optimize
            outputs = all_nets[j](inputs)
            loss = all_criteria[j](outputs, labels.long())
            loss.backward()
            all_optimizers[j].step()

    if epoch%10 == 0:
        print("Finished training epoch " + str(epoch))

print('Finished Training')

sigma = torch.zeros((len(trainloader_source),84))
g = torch.zeros((num_classes,84,len(trainloader_source)))

cl = 0

with torch.no_grad():
    #Compute scores
    running_probs_train = torch.zeros((len(trainloader_target.dataset),num_classes))
    running_probs_test = torch.zeros((len(testloader.dataset),num_classes))
    for i in range(len(trainloader_source)):
        for data in trainloader_target:
            # get the inputs
            inputs, labels = data
            sigma[i,:], g[:,:,i] = compute_max_corr(all_nets[i][0],inputs,labels,num_classes)
            running_probs_train += weighted_network_output(all_nets[i][0],sigma[i,:],g[:,:,i],inputs)
        for data in testloader:
            inputs, labels = data
            running_probs_test += weighted_network_output(all_nets[i][0],sigma[i,:],g[:,:,i],inputs)
    all_feats_train = torch.zeros((len(trainloader_target.dataset),84*len(trainloader_source)))
    all_feats_test = torch.zeros((len(testloader.dataset),84*len(trainloader_source)))
    for i in range(len(trainloader_source)):
        for data in trainloader_target:
            # get the inputs
            inputs, labels_train = data
            all_feats_train[:,i*84:(i+1)*84] = all_nets[i][0](inputs)
        for data in testloader:
            inputs, labels_test = data
            all_feats_test[:,i*84:(i+1)*84] = all_nets[i][0](inputs)

def classify_svm(x_train,y_train, x_test, y_test):
    clf = LinearSVC()#dual=False)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return sum(np.array(y_test)==y_test_pred)/len(y_test)

def classify_logistic(x_train,y_train, x_test, y_test):
    clf = LogisticRegression()#dual=False)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return sum(np.array(y_test)==y_test_pred)/len(y_test)

# acc_svm = classify_svm(all_feats_train.numpy(),labels_train.numpy().astype(int),all_feats_test.numpy(),labels_test.numpy().astype(int))
#acc_svm2 = classify_logistic(all_feats_train.numpy(),labels_train.numpy().astype(int),all_feats_test.numpy(),labels_test.numpy().astype(int))

_, predicted_train = torch.max(running_probs_train, 1)
_, predicted_test = torch.max(running_probs_test, 1)


for data in testloader:
    inputs, labels = data
    acc_test = (predicted_test == labels.long()).sum().item()/labels.size(0)

# print("svm accuracy")
# print(acc_svm)
print("num_target_samps:", num_target_samps)
print("max corr accuracy")
print(acc_test)
