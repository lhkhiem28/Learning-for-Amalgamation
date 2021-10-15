"""
Datasets for testing
"""

import torch
import torchvision.transforms as transforms
import os
import numpy as np
import pickle
import random
from PIL import Image

dataset_path = os.path.join('datasets','cifar-10-batches-py')

class CifarBinaryDataset(torch.utils.data.Dataset):
    """Binary Cifar Dataset"""
    
    def __init__(self, filepath, num_samps=None, train=True, transform=None, target_transform=None, offset = 0):
        self.transform = transform
        self.target_transform = target_transform
        self.offset = offset

        self.train = train  # training set or test set

        self.data = []
        self.targets = []
        self.num_samps = num_samps

        # now load the picked numpy arrays
        with open(filepath, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry[0])
            self.targets.extend(entry[1])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.true_len = self.data.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
#        if self.num_samps:
#            img, target = self.data[index*(self.true_len//self.num_samps)], self.targets[index*(self.true_len//self.num_samps)]
#        else:
        img, target = self.data[index + self.offset], self.targets[index + self.offset]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
            return self.num_samps if self.num_samps else self.true_len

class DogsBinaryDataset(torch.utils.data.Dataset):
    """Binary Cifar Dataset"""
    
    def __init__(self, folder_base, num_samps=10, classes = [0,1], train=True, transform=None, target_transform=None, offset = 0):
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set
        self.offset = offset

        self.data = []
        self.targets = []
        self.num_samps = num_samps #num_samples per class
        self.classes = classes

        # Load images
        folders = sorted(os.listdir(folder_base))
        for i in range(len(self.classes)):
            folderpath = os.path.join(folder_base,folders[i])
            image_list = sorted(os.listdir(folderpath))
            for j in range(num_samps):
                if self.train:
                    im = Image.open(os.path.join(folderpath,image_list[j+offset]))
                else:
                    im = Image.open(os.path.join(folderpath,image_list[-j-offset]))                    
                im = im.resize((144,144))
                self.data.append(im)
                self.targets.append(i)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
            return self.num_samps * len(self.classes)

class ImageNetBinaryDataset(torch.utils.data.Dataset):
    """Binary Cifar Dataset"""
    
    def __init__(self, folder_base, num_samps=10, classes = [0,1], mode="train", transform=None, target_transform=None, offset=0):
        self.transform = transform
        self.target_transform = target_transform

        self.mode = mode  # training set or test set
        
        self.offset = offset
        self.data = []
        self.targets = []
        self.num_samps = num_samps #num_samples per class
        self.classes = classes

        # Load images
        for i in range(len(self.classes)):
            folderpath = os.path.join(folder_base,'train',self.classes[i],"images")
            image_list = sorted(os.listdir(folderpath))
            counter=0
            j=0
            while counter < num_samps:
                if self.mode == "train":
                    im = Image.open(os.path.join(folderpath,image_list[j + offset]))
                else:
                    im = Image.open(os.path.join(folderpath,image_list[-j - offset]))
                if im.mode == "RGB":
                    counter+=1
                    im = im.resize((64,64))
                    self.data.append(im)
                    self.targets.append(i)
                j+=1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
            return self.num_samps * len(self.classes)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def generate_dataset(mode, num_source_samps, num_target_samps):
   
    if mode == "cifar":
        #Cifar100
        #num_train_samps = 500 is the recommended number
        cf100_folder = os.path.join('datasets','cifar-100-python')
    
        #Get target set
        trainset_target = CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_0_1.p'), num_samps=num_target_samps, train=False,
                                               transform=transform, offset = random.randint(0,100))
        trainloader_target = torch.utils.data.DataLoader(trainset_target, batch_size=len(trainset_target),
                                                 shuffle=False, num_workers=0)    
        testset = CifarBinaryDataset(filepath=os.path.join(cf100_folder,'test_batch_0_1.p'), train=False,
                                               transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                 shuffle=False, num_workers=0)

        #compile source datasets
        trainset_source = []
        
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_5_6.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_10_11.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_15_16.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_20_21.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_25_26.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_30_31.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_35_36.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_40_41.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_45_46.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_50_51.p'),num_samps=num_source_samps, train=False,
                                               transform=transform))
#        trainset_source.append(CifarBinaryDataset(filepath=os.path.join(cf100_folder,'data_batch_55_56.p'),num_samps=num_source_samps, train=False,
#                                               transform=transform))
        
        trainloader_source = []
        for i in range(len(trainset_source)):
            trainloader_source.append(torch.utils.data.DataLoader(trainset_source[i], batch_size=len(trainset_source[i])//100,
                                                 shuffle=True, num_workers=0))
    
        return trainloader_source, trainloader_target, testloader
    
    elif mode == "dogs":
        #Dogs
        #num_samps=50 is the recommended number
        #Get target set
        trainset_target = DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_target_samps, classes=[51,9,10,11,12], train=True,
                                               transform=transform, offset=random.randint(0,60))
        trainloader_target = torch.utils.data.DataLoader(trainset_target, batch_size=len(trainset_target),
                                                 shuffle=False, num_workers=0)    
        testset = DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=100, classes=[51,9,10,11,12], train=False,
                                               transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                 shuffle=False, num_workers=0)

        #compile source datasets
        trainset_source = []
        
        trainset_source.append(DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_source_samps, classes=[0,1,2,3,4], train=True,
                                               transform=transform))
        trainset_source.append(DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_source_samps, classes=[70,75,76,77,78], train=True,
                                               transform=transform))
        trainset_source.append(DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_source_samps, classes=[32,33,41,54,60], train=True,
                                               transform=transform))
        trainset_source.append(DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_source_samps, classes=[73,17,18,19,20], train=True,
                                               transform=transform))
        trainset_source.append(DogsBinaryDataset(folder_base=os.path.join("datasets","dogs","Images"),num_samps=num_source_samps, classes=[14,21,29,23,24], train=True,
                                               transform=transform))

        trainloader_source = []
        for i in range(len(trainset_source)):
            trainloader_source.append(torch.utils.data.DataLoader(trainset_source[i], batch_size=len(trainset_source[i])//100,
                                                 shuffle=True, num_workers=0))
        
        return trainloader_source, trainloader_target, testloader

    elif mode == "tiny_imagenet":
        #TinyImageNet
        #num_samps=250 is the recommended number
        #Get target set
        trainset_target = ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_target_samps, classes=['n02814860', 'n04099969', 'n02788148', 'n01910747', 'n02999410'], mode="train",
                                               transform=transform)
        trainloader_target = torch.utils.data.DataLoader(trainset_target, batch_size=len(trainset_target),
                                                 shuffle=False, num_workers=0)    
        testset = ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=250, classes=['n02814860', 'n04099969', 'n02788148', 'n01910747', 'n02999410'], mode="test",
                                               transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                 shuffle=False, num_workers=0)

        #compile source datasets
        trainset_source = []
        
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n01983481', 'n02165456', 'n02699494', 'n07871810', 'n04275548'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n03617480', 'n04366367', 'n02841315', 'n09193705', 'n03026506'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n02669723', 'n07768694', 'n03814639', 'n07749582', 'n03649909'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n04074963', 'n02099712', 'n03444034', 'n02410509', 'n03977966'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n03970156', 'n07695742', 'n02909870', 'n02226429', 'n04070727'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n02123394', 'n01774750', 'n02395406', 'n02279972', 'n04486054'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n02364673', 'n03976657', 'n04259630', 'n06596364', 'n02129165'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n02281406', 'n04596742', 'n04398044', 'n02099601', 'n02769748'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n09428293', 'n02892201', 'n02002724', 'n02123045', 'n03544143'], mode="train",
                                               transform=transform))
        trainset_source.append(ImageNetBinaryDataset(folder_base=os.path.join("datasets","tiny-imagenet-200"),num_samps=num_source_samps, classes=['n01443537', 'n03670208', 'n01984695', 'n03179701', 'n01629819'], mode="train",
                                               transform=transform))
        
        trainloader_source = []
        for i in range(len(trainset_source)):
            trainloader_source.append(torch.utils.data.DataLoader(trainset_source[i], batch_size=len(trainset_source[i])//100,
                                                 shuffle=True, num_workers=0))

        return trainloader_source, trainloader_target, testloader

    else:
        raise Exception('Invalid dataset type')
