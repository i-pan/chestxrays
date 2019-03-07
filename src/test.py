############
# ARGPARSE #
############

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('load_model', type=str) 
parser.add_argument('checkpoint', type=str) 
parser.add_argument('train_data_group', type=str) 
parser.add_argument('test_data_group', type=str) 
parser.add_argument('data_dir', type=str) 
parser.add_argument('results_file', type=str) 
parser.add_argument('data_splits', type=str) 
parser.add_argument('--gpu', type=int, nargs='?',
                    const=0, default=0)
parser.add_argument('--nb-classes', type=int, nargs='?',
                    const=2, default=2)
parser.add_argument('--batch-size', type=int, nargs='?',
                    const=128, default=128)
parser.add_argument('--augment-p', type=float, nargs='?',
                    const=0.5, default=0.5)
parser.add_argument('--dropout-p', type=float, nargs='?',
                    const=0., default=0.)

args = parser.parse_args()

###########
# IMPORTS #
###########

from functools import partial 
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset, DataLoader 
from torch import nn, optim

from tqdm import tqdm 

import pretrainedmodels 
import pandas as pd 
import numpy as np
import datetime 
import torch 
import cv2
import os

from albumentations import (
    Compose, OneOf, HorizontalFlip, ShiftScaleRotate, JpegCompression, Blur, CLAHE, RandomGamma, RandomContrast, RandomBrightness
)

##################
# ALBUMENTATIONS #
##################

def simple_aug(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=10, scale_limit=0.15, p=0.5),
        OneOf([
            JpegCompression(quality_lower=80),
            Blur(),
        ], p=0.5),
        OneOf([
            CLAHE(),
            RandomGamma(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.5)
    ], p=p)

train_aug  = simple_aug(p=args.augment_p)

##################
# DATA FUNCTIONS #
##################

def channels_last_to_first(img):
    img = np.swapaxes(img, 0,2)
    img = np.swapaxes(img, 1,2)
    return img 

def preprocess_input(img, model): 
    # assume image is RGB 
    img = img[..., ::-1].astype('float32')
    model_min = model.input_range[0] ; model_max = model.input_range[1] 
    img_min = float(np.min(img)) ; img_max = float(np.max(img))
    img_range = img_max - img_min 
    model_range = model_max - model_min 
    img = (((img - img_min) * model_range) / img_range) + model_min 
    img[..., 0] -= model.mean[0] 
    img[..., 1] -= model.mean[1] 
    img[..., 2] -= model.mean[2] 
    img[..., 0] /= model.std[0] 
    img[..., 1] /= model.std[1] 
    img[..., 2] /= model.std[2] 
    return img

class CXRDataset(Dataset): 
    #
    def __init__(self, imgfiles, labels, resize=None, preprocess=None, transform=None): 
        self.imgfiles   = imgfiles
        self.labels     = labels 
        self.preprocess = preprocess
        self.resize     = resize
        self.transform  = transform
    #
    def __len__(self): 
        return len(self.imgfiles) 
    # 
    def __getitem__(self, i): 
        X = cv2.imread(self.imgfiles[i])
        if self.resize: X = self.resize(X)
        if self.transform: X = self.transform(image=X)['image']
        X = channels_last_to_first(X)
        y = np.asarray(self.labels[i])
        if self.preprocess: X = self.preprocess(X) 
        return torch.from_numpy(X).type('torch.FloatTensor'), torch.from_numpy(y).type('torch.FloatTensor')  



##########
# SCRIPT #
##########
print (">>CNNs for CXRs<<")

if args.train_data_group not in ['rih', 'nih']: 
    raise Exception('{} is not a valid option for train_data_group. Please select from : [nih, rih]'.format(args.train_data_group))

if args.test_data_group not in ['rih', 'nih']: 
    raise Exception('{} is not a valid option for test_data_group. Please select from : [nih, rih]'.format(args.test_data_group))

torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 

cxr_df = pd.read_csv(args.data_splits) 
if args.test_data_group == 'rih': 
    cxr_df['label'] = [1 if _ == 4 else 0 for _ in cxr_df.radcat]     
elif args.test_data_group == 'nih': 
    cxr_df['Finding'] = 1 - cxr_df['No Finding'] 

test_df = cxr_df[cxr_df.split0 == 'test'].reset_index(drop=True)
print ('TEST : n = {}'.format(test_df.shape[0]))

params = {'batch_size':  args.batch_size, 
          'shuffle':     False, 
          'num_workers': 4}

# Run model script 
print ('Loading pretrained model [{}] ...'.format(args.model)) 
execfile(args.load_model) 
model.load_state_dict(torch.load(args.checkpoint))
model.eval().cuda()

# Set up preprocessing function with model 
pp = partial(preprocess_input, model=model) 

print ('Setting up data loaders ...')
if args.test_data_group == 'rih': 
    test_images = [os.path.join(args.data_dir, _.replace('dcm', 'jpg')) for _ in test_df.dicom_file]
    test_labels = np.asarray(test_df.label)
elif args.test_data_group =='nih': 
    findings = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 
                'Cardiomegaly', 'Pleural_Thickening', 'Consolidation', 
                'Pneumothorax', 'Mass', 'Nodule', 'Atelectasis', 'Effusion', 
                'Infiltration', 'Finding']
    test_images = [os.path.join(args.data_dir, _.replace('dcm', 'jpg')) for _ in test_df.pid] 
    test_labels = np.asarray(test_df[findings]) 

test_set = CXRDataset(imgfiles=test_images, labels=test_labels, preprocess=pp) 
test_gen = DataLoader(test_set, **params)

start_time = datetime.datetime.now() 
print ('TESTING : START')

with torch.no_grad(): 
    test_y_pred = [] ; test_y_true = [] 
    for i, data in tqdm(enumerate(test_gen), total=len(test_gen)): 
        batch, labels = data 
        output = model(batch.cuda())
        if args.train_data_group == args.test_data_group: 
            if args.train_data_group == 'rih': 
                test_y_pred.extend(torch.sigmoid(output).cpu().numpy()[:,-1])
            elif args.test_data_group == 'nih': 
                test_y_pred.extend(torch.sigmoid(output).cpu().numpy())
            test_y_true.extend(labels.numpy())
        elif args.train_data_group == 'nih' and args.test_data_group == 'rih': 
            test_y_pred.extend(torch.sigmoid(output).cpu().numpy()[:,-1]) 
            test_y_true.extend(labels.numpy())
        elif args.train_data_group == 'rih' and args.test_data_group == 'nih': 
            test_y_pred.extend(torch.sigmoid(output).cpu().numpy()[:,-1])
            test_y_true.extend(labels.numpy()[:,-1])

test_y_pred = np.vstack(test_y_pred)
test_y_true = np.vstack(test_y_true) 

if args.test_data_group == 'rih': 
    results_df = pd.DataFrame({'pid': [_.split('/')[-1].split('-')[0] for _ in test_images],
                               'y_pred': test_y_pred[:,0],
                               'y_true': test_y_true[:,0]}) 
    results_df.to_csv(args.results_file, index=False) 
elif args.train_data_group == 'nih' and args.test_data_group == 'nih': 
    results_df = pd.DataFrame({'pid': [_.split('/')[-1] for _ in test_images]})
    for i, find in enumerate(findings): 
        results_df['y_pred_{}'.format(find)] = test_y_pred[:,i] 
        results_df['y_true_{}'.format(find)] = test_y_true[:,i] 
    results_df.to_csv(args.results_file, index=False) 
elif args.train_data_group == 'rih' and args.test_data_group == 'nih': 
    results_df = pd.DataFrame({'pid': [_.split('/')[-1] for _ in test_images],
                               'y_pred': test_y_pred[:,0],
                               'y_true': test_y_true[:,0]})
    results_df.to_csv(args.results_file, index=False) 

print ('TESTING : END') 
print ('Testing took {}\n'.format(datetime.datetime.now() - start_time))
