# -*- coding: utf-8 -*-
"""
==========================
**Author**: Qian Wang, qian.wang173@hotmail.com
"""


from __future__ import print_function, division

from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.transform import rescale
from torch.optim import lr_scheduler
from torch.nn import functional
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
# from skimage import io, transform
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
plt.ion()   # interactive mode
from myInception_segLoss import headCount_inceptionv3
# from generate_density_map import generate_multi_density_map,generate_density_map
from density import generate_density_map
from segmentation import generate_segmentation_map


IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    """
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
    """
    d = os.path.join(dir,'images')
    for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                # print("fname: ",fname)
                if has_file_allowed_extension(fname, extensions):
                    pure_fname = fname.split('.')[0]
                    id = pure_fname.split('_')[-1]
                    image_path = os.path.join(root, fname)
                    head,tail = os.path.split(root)
                    label_path = os.path.join(head,'ground-truth','GT_'+fname[:-4]+'.mat')
                    dmap_path = os.path.join(head,'Dmap','DMAP_'+id+'.mat')
                    # print("dmap_path: ",dmap_path)
                    pmap_path = os.path.join(head,'Pmap','PMAP_'+id+'.mat')
                    item = [image_path, label_path,dmap_path,pmap_path]
                    images.append(item)

    return images

def ReadImage(imPath,mirror = False,scale=1.0):
    """
    Read gray images.
    """
    imArr = np.array(Image.open(imPath))#.convert('L'))
    if(scale!=1):
        imArr = rescale(imArr, scale,preserve_range=True)
    if (len(imArr.shape)<3):
        imArr = imArr[:,:,np.newaxis]
        imArr = np.tile(imArr,(1,1,3))

    return imArr

def ReadMap(mapPath,name):
    """
    Load the density map from matfile.
    """
    map_data = scipy.io.loadmat(mapPath)
    return map_data[name]

def load_data_pairs(img_path, dmap_path, pmap_path):

    img_data = ReadImage(img_path)
    dmap_data = ReadMap(dmap_path,'dmap')
    pmap_data = ReadMap(pmap_path,'pmap')

    img_data = img_data.astype('float32')
    dmap_data = dmap_data.astype('float32')
    pmap_data = pmap_data.astype('int32')

    dmap_data = dmap_data*100.0
    img_data = img_data/255.0

    return img_data, dmap_data, pmap_data

def get_batch_patches(img_path, dmap_path, pmap_path, patch_dim, batch_size):
    rand_img, rand_dmap, rand_pmap = load_data_pairs(img_path, dmap_path, pmap_path)

    if np.random.random() > 0.5:
        rand_img=np.fliplr(rand_img)
        rand_dmap=np.fliplr(rand_dmap)
        rand_pmap=np.fliplr(rand_pmap)

    w, h, c = rand_img.shape

    patch_width = int(patch_dim[0])
    patch_heigh = int(patch_dim[1])

    batch_img = np.zeros([batch_size, patch_width, patch_heigh, c]).astype('float32')
    batch_dmap = np.zeros([batch_size, patch_width, patch_heigh]).astype('float32')
    batch_pmap = np.zeros([batch_size, patch_width, patch_heigh]).astype('int32')
    # batch_num = np.zeros([batch_size]).astype('int32')

    rand_img = rand_img.astype('float32')
    rand_dmap = rand_dmap.astype('float32')
    rand_pmap = rand_pmap.astype('int32')

    for k in range(batch_size):
        # randomly select a box anchor
        w_rand = randint(0, w - patch_width)
        h_rand = randint(0, h - patch_heigh)

        pos = np.array([w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(rand_img[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh, :])
        dmap_temp = copy.deepcopy(rand_dmap[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])
        pmap_temp = copy.deepcopy(rand_pmap[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])

        batch_img[k, :, :, :] = img_norm
        batch_dmap[k, :, :] = dmap_temp
        batch_pmap[k, :, :] = pmap_temp
        # global density step siz, L which is estimated by equation 5 in the paper
        # L = 8
        # batch_num[k] = dmap_temp.sum()/L

    return batch_img, batch_dmap, batch_pmap

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train',extensions=IMG_EXTENSIONS,patch_size=128,num_patches_per_image=4):
        self.samples = make_dataset(data_dir,extensions)
        self.image_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.patch_size = patch_size
        self.numPatches = num_patches_per_image
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        # img_file,label_file = self.samples[idx]
        img_path,label_path, dmap_path ,pmap_path = self.samples[idx]
        # image = cv2.imread(img_file)
        # height, width, channel = image.shape
        # annPoints = scipy.io.loadmat(label_file)
        # annPoints = annPoints['image_info'][0][0][0][0][0]
        # density = cv2.imread(density_file)
        # segmentation = cv2.imread(segmentation_file)
        # positions = generate_density_map(shape=image.shape,points=annPoints,f_sz=15,sigma=4)
        # fbs = generate_segmentation_map(shape=image.shape,points=annPoints,f_sz=25,sigma=1)
        #
        # positions = generate_density_map(image_path=img_file,position_head_path=label_file)
        # fbs = generate_segmentation_map(image_path=img_file,position_head_path=label_file)
        #
        # fbs = np.int32(fbs>0)
        rand_img, rand_dmap, rand_pmap = load_data_pairs(img_path, dmap_path, pmap_path)
        targetSize = [self.patch_size,self.patch_size]
        height, width, channel = rand_img.shape
        if height < targetSize[0] or width < targetSize[1]:
            rand_img = cv2.resize(rand_img,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count = rand_dmap.sum()
            max_value = rand_dmap.max()
            # down density map
            rand_dmap = cv2.resize(rand_dmap, (np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count2 = rand_dmap.sum()
            rand_dmap = np.minimum(rand_dmap*count/(count2+1e-8),max_value*10)
            rand_pmap = cv2.resize(rand_pmap,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            rand_pmap = np.int32(rand_pmap>0)
        if len(rand_img.shape)==2:
            rand_img = np.expand_dims(rand_img,2)
            rand_img = np.concatenate((rand_img,rand_img,rand_img),axis=2)
        # transpose from h x w x channel to channel x h x w
        image = rand_img.transpose(2,0,1)
        numPatches = self.numPatches
        if self.phase == 'train':
            # batch_img, batch_dmap, batch_pmap = get_batch_patches(img_path, dmap_path, pmap_path, patch_size, num_patches_per_image)
            batch_img, batch_dmap, batch_pmap = getRandomPatchesFromImage(image,rand_dmap,rand_pmap,targetSize,numPatches)
            x = np.zeros((batch_img.shape[0],3,targetSize[0],targetSize[1]))
            if self.transform:
              for i in range(batch_img.shape[0]):
                #transpose to original:h x w x channel
                x[i,:,:,:] = self.transform(np.uint8(batch_img[i,:,:,:]).transpose(1,2,0))
            patchSet = x
            return patchSet, batch_dmap, batch_pmap
        if self.phase == 'val' or self.phase == 'test':
            batch_img, batch_dmap, batch_pmap = getAllFromImage(image, rand_dmap,rand_pmap)
            batch_img[0,:,:,:] = self.transform(np.uint8(batch_img[0,:,:,:]).transpose(1,2,0))
        return batch_img, batch_dmap, batch_pmap

def getRandomPatchesFromImage(image,positions,fbs,target_size,numPatches):
    # generate random cropped patches with pre-defined size, e.g., 224x224
    imageShape = image.shape
    if np.random.random()>0.5:
        for channel in range(3):
            image[channel,:,:] = np.fliplr(image[channel,:,:])
        positions = np.fliplr(positions)
        fbs = np.fliplr(fbs)
    total_anno = positions.sum()
    height = image.shape[0]
    width = image.shape[1]
    total_pixel = width*height
    max_anno = 0
    patchSet = np.zeros((numPatches,3,target_size[0],target_size[1]))
    # generate density map
    countSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    fbsSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    # classSet = np.zeros([numPatches])
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1]-target_size[0]+1)#x-height
        topLeftY = np.random.randint(imageShape[2]-target_size[1]+1)#y-width
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        patchSet[i,:,:,:] = thisPatch
        height = thisPatch.shape[0]
        width = thisPatch.shape[1]
        thisPixel = width*height
        # density map
        position = positions[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        fb = fbs[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        position = position.reshape((1, position.shape[0], position.shape[1]))
        fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
        countSet[i,:,:,:] = position
        fbsSet[i,:,:,:] = fb
        # max_anno = max(max_anno,total_anno*thisPixel/total_pixel)
        # max_anno = max(max_anno,countSet[i,:,:,:].sum())
        #classSet[i] = position.sum()/L

    # L = 8
    # for i in range(numPatches):
    #     classSet[i] = countSet[i,:,:,:].sum()/L

    return patchSet, countSet, fbsSet

def getAllPatchesFromImage(image,positions,target_size):
    # generate all patches from an image for prediction
    nchannel,height,width = image.shape
    nRow = np.int(height/target_size[1])
    nCol = np.int(width/target_size[0])
    target_size[1] = np.int(height/nRow)
    target_size[0] = np.int(width/nCol)
    patchSet = np.zeros((nRow*nCol,3,target_size[1],target_size[0]))
    for i in range(nRow):
      for j in range(nCol):
        patchSet[i*nCol+j,:,:,:] = image[:,i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
    return patchSet

def getAllFromImage(image,positions,fbs):
    nchannel, height, width = image.shape
    patchSet =np.zeros((1,3,height, width))
    patchSet[0,:,:,:] = image[:,:,:]
    countSet = positions.reshape((1,1,positions.shape[0], positions.shape[1]))
    fbsSet = fbs.reshape((1,1,fbs.shape[0], fbs.shape[1]))
    return patchSet, countSet, fbsSet

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def focal_loss_num_func(logits, labels, alpha=0.25, gamma=2.0):
    """
    Loss = weighted * -target*log(softmax(logits))
    :param logits: probability score
    :param labels: ground_truth
    :return: softmax-weighted loss
    """
    entroy=nn.CrossEntropyLoss()

    labels=labels.view(24,1)
    # print(labels.long().squeeze())
    logits=logits.view(24,5)
    # print(logits)

    # print(labels.size())
    # print(logits.size())
    # labels = labels.astype(np.int)
    output=entroy(logits,labels.long().squeeze()) #采用CrossEntropyLoss计算的结果。
    # print(output)
    # print("logits: ",logits)
    # gt = functional.one_hot(labels,5)
    # softmaxpred = functional.softmax(logits)
    # loss = 0
    # for i in range(5):
    #     gti = gt[:,i]
    #     predi = softmaxpred[:,i]
    #     loss = loss+ torch.mean(gti*torch.pow(1 - predi, gamma)* torch.log(torch.clip_(predi, 0.005, 1)))
    return output


def focal_loss_func(logits, labels, alpha=0.25, gamma=2.0):
    """
    Loss = weighted * -target*log(softmax(logits))
    :param logits: probability score
    :param labels: ground_truth
    :return: softmax-weighted loss
    """
    # print(labels)
    # gt = functional.one_hot(torch.from_numpy(labels),2)
    gt = labels
    # print(logits)
    softmaxpred = functional.softmax(logits)
    # print(softmaxpred)
    loss = 0
    for i in range(2):
        gti = gt[:,:,i].clone()
        predi = softmaxpred[:,:,i].clone()
        weighted = 1-(torch.sum(gti)/torch.sum(gt))
        loss = loss+ torch.mean(weighted *gti* torch.pow(1 - predi, gamma)* torch.log(torch.clip_(predi, 0.005, 1)))
    return -loss/2

def l1_loss(prediction, ground_truth, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: mean of the l1 loss.
    """
    # print("pre: ",prediction,ground_truth)
    absolute_residuals = torch.abs(torch.subtract(prediction[:,:,:].clone(), ground_truth))
    if weight_map is not None:
        absolute_residuals = torch.multiply(absolute_residuals, weight_map)
        sum_residuals = torch.sum(absolute_residuals)
        sum_weights = torch.sum(weight_map)
    else:
        sum_residuals = torch.sum(absolute_residuals)
        # print(sum_residuals,'\n\n')
        sum_weights = absolute_residuals.size(0)
    return torch.true_divide(float(sum_residuals),float(sum_weights))


def l2_loss(prediction, labels):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: sum(differences squared) / 2 - Note, no square root
    """
    criterion1 = nn.MSELoss(reduction='sum') # for density map loss
    residuals = torch.subtract(prediction[:,:,:].clone(), labels)
    sum_residuals = criterion1(prediction[:,:,:].clone(),labels)
    # print(sum_residuals)
    sum_weights = residuals.size(0)
    return torch.true_divide(float(sum_residuals),float(sum_weights))


def train_model(model, optimizer, scheduler, num_epochs=100, seg_loss=False, cl_loss=False, test_step=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae_val = 1e6
    best_mae_by_val = 1e6
    best_mae_by_test = 1e6
    best_mse_by_val = 1e6
    best_mse_by_test = 1e6
    criterion1 = nn.MSELoss(reduce=False) # for density map loss
    criterion2 = nn.BCELoss() # for segmentation map loss
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0        
        # Iterate over data.
        for index, (inputs, labels, fbs) in enumerate(dataloaders['train']):
            # labels = labels*100
            labels = skimage.measure.block_reduce(labels.cpu().numpy(),(1,1,1,4,4),np.sum)
            fbs = skimage.measure.block_reduce(fbs.cpu().numpy(),(1,1,1,4,4),np.max)
            # fbs = np.float32(fbs>0)
            labels = torch.from_numpy(labels)
            fbs = torch.from_numpy(fbs)
            labels = labels.to(device)
            fbs = fbs.to(device)
            # density_labels = density_labels.to(device)
            inputs = inputs.to(device)
            inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
            labels = labels.view(-1,labels.shape[3],labels.shape[4])

            fbs = fbs.view(-1,fbs.shape[3],fbs.shape[4])
            inputs = inputs.float()
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                output,fbs_out = model(inputs)
                # print(output)
                # print(labels)
                pred_count = output.sum()
                true_count = labels.to(torch.device("cpu")).numpy().sum()
                # print(pred_count," vs ",true_count)
                # print("output vs labels: ",output.size()," ",labels.size())
                # print("fbs_out vs fbs : ",fbs_out.size()," ",fbs.size())
                # loss_den = criterion1(output, labels)
                # loss_seg = criterion2(fbs_out, fbs)
                # =========density estimation loss=========
                density_loss = 10*l1_loss(prediction=output, ground_truth=labels)+l2_loss(output, labels)
                # =========segmentation loss=========
                segment_loss = 10*focal_loss_func(fbs_out, fbs)
                # =========global density prediction loss=========
                # global_density_loss = focal_loss_num_func(density_class,density_labels)
                # print("test loss: ", density_loss,segment_loss)
                # total_loss = density_loss + segment_loss + global_density_loss

                # if cl_loss:
                #     th = 0.1*epoch+5 #cl2
                # else:
                #     th=1000 # no curriculum loss when th is set a big number
                # weights = th/(F.relu(labels-th)+th)
                # density_loss = density_loss*weights
                # density_loss = density_loss.sum()/weights.sum()
                # =========density estimation loss=========
                if seg_loss:
                    # print(density_loss)
                    # print(segment_loss)
                    # print(global_density_loss)
                    loss = density_loss + 50*segment_loss
                else:
                    loss = density_loss

                loss.backward()
                # print("loss backward end")
                optimizer.step()
            # print("calculate running loss")
            running_loss += loss.item() * inputs.size(0)
               

        scheduler.step()
        print("dataset_sizes : ",dataset_sizes['train'])
        epoch_loss = running_loss / dataset_sizes['train']


        print('Train Loss: {:.4f}'.format(epoch_loss))
        print()
        if epoch%test_step==0:
            tmp,epoch_mae,epoch_mse,epoch_mre=test_model(model,optimizer,'val')
            tmp,epoch_mae_test,epoch_mse_test,epoch_mre_test = test_model(model,optimizer,'test')
            if  epoch_mae < best_mae_val:
                best_mae_val = epoch_mae
                best_mae_by_val = epoch_mae
                best_mse_by_val = epoch_mse
                best_epoch_val = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch_mae_test < best_mae_by_test:
                best_mae_by_test = epoch_mae_test
                best_mse_by_test = epoch_mse_test
                best_epoch_test = epoch
            print()
            print('best MAE and MSE by val:  {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_val,best_mse_by_val, best_epoch_val))
            print('best MAE and MSE by test: {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_test,best_mse_by_test, best_epoch_test))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    # Iterate over data.
    for index, (inputs, labels, fbs) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,labels.shape[3],labels.shape[4])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(False):
            outputs,fbs_out = model(inputs)
            outputs = outputs.to(torch.device("cpu")).numpy()/100
            pred_count = outputs.sum()
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        # backward + optimize only if in training phase
        mse = mse + np.square(pred_count-true_count)
        mae = mae + np.abs(pred_count-true_count)
        mre = mre + np.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = np.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(f'MAE:{mae:2.2f}, RMSE:{mse:2.2f}, MRE:{mre:2.4f}')
    time_elapsed = time.time() - since
    return pred,mae,mse,mre

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
#####################################################################
# set parameters here
seg_loss = True
cl_loss = True
test_step = 1
batch_size = 6
num_workers = 10
patch_size = 128
num_patches_per_image = 4
data_dir = './ShanghaiTech/part_B/'

# define data set
image_datasets = {x: ShanghaiTechDataset(data_dir+x+'_data', 
                        phase=x, 
                        transform=data_transforms[x],
                        patch_size=patch_size,
                        num_patches_per_image=num_patches_per_image)
                    for x in ['train','test']}
image_datasets['val'] = ShanghaiTechDataset(data_dir+'train_data',
                            phase='val',
                            transform=data_transforms['val'],
                            patch_size=patch_size,
                            num_patches_per_image=num_patches_per_image)
## split the data into train/validation/test subsets
indices = list(range(len(image_datasets['train'])))
split = np.int(len(image_datasets['train'])*0.2)

val_idx = np.random.choice(indices, size=split, replace=False)
train_idx = indices#list(set(indices)-set(val_idx))
test_idx = range(len(image_datasets['test']))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetSampler(test_idx)

train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'],batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'],batch_size=1,sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'],batch_size=1,sampler=test_sampler, num_workers=num_workers)

dataset_sizes = {'train':len(train_idx),'val':len(val_idx),'test':len(image_datasets['test'])}
dataloaders = {'train':train_loader,'val':val_loader,'test':test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# define models and training
model = headCount_inceptionv3(pretrained=True)
# model = MCNN()
# model = SANet()
# model = TEDNet(use_bn=True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

model = train_model(model, optimizer, exp_lr_scheduler,
                    num_epochs=30,
                    seg_loss=seg_loss, 
                    cl_loss=cl_loss, test_step=test_step)
                    
pred,mae,mse,mre = test_model(model,optimizer,'test')
scipy.io.savemat('./results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
model_dir = './'
torch.save(model.state_dict(), model_dir+'saved_model.pt')

