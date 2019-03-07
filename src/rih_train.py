############
# ARGPARSE #
############

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('load_model', type=str) 
parser.add_argument('data_dir', type=str) 
parser.add_argument('save_dir', type=str) 
parser.add_argument('data_splits', type=str) 
parser.add_argument('val_split', type=int) 
parser.add_argument('--gpu', type=int, nargs='?',
                    const=0, default=0)
parser.add_argument('--nb-classes', type=int, nargs='?',
                    const=2, default=2)
parser.add_argument('--batch-size', type=int, nargs='?',
                    const=16, default=16)
parser.add_argument('--augment-p', type=float, nargs='?',
                    const=0.5, default=0.5)
parser.add_argument('--dropout-p', type=float, nargs='?',
                    const=0., default=0.)
parser.add_argument('--max-epochs', type=int, nargs='?',
                    const=100, default=100) 
parser.add_argument('--steps-per-epoch', type=int, nargs='?',
                    const=1000, default=1000)
parser.add_argument('--initial-lr', type=float, nargs='?',
                    const=1e-4, default=1e-4)
parser.add_argument('--weight-decay', type=float, nargs='?',
                    const=1e-6, default=1e-6) 
parser.add_argument('--lr-patience', type=int, nargs='?',
                    const=2, default=2) 
parser.add_argument('--stop-patience', type=int, nargs='?',
                    const=6, default=6) 
parser.add_argument('--annealing-factor', type=float, nargs='?',
                    const=0.5, default=0.5)
parser.add_argument('--min-delta', type=float, nargs='?',
                    const=1e-3, default=1e-3) 
parser.add_argument('--verbosity', type=int, nargs='?',
                    const=100, default=100)

args = parser.parse_args()

###########
# IMPORTS #
###########

from functools import partial 
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset, DataLoader 
from torch import nn, optim

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
        return torch.from_numpy(X).type('torch.FloatTensor'), torch.from_numpy(y).type('torch.LongTensor')  

##########
# SCRIPT #
##########

print (">>CNNs for CXRs<<")
torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 

if not os.path.exists(args.save_dir): os.makedirs(args.save_dir) 

cxr_df = pd.read_csv(args.data_splits) 
cxr_df['label'] = [1 if _ == 4 else 0 for _ in cxr_df.radcat] 

split = 'split{}'.format(args.val_split)

train_df = cxr_df[cxr_df[split] == 'train'].reset_index(drop=True)
valid_df = cxr_df[cxr_df[split] == 'valid'].reset_index(drop=True)
print ('TRAIN : n = {}'.format(train_df.shape[0]))
print ('VALID : n = {}'.format(valid_df.shape[0]))

params = {'batch_size':  args.batch_size, 
          'shuffle':     True, 
          'num_workers': 4}

# Run model script 
print ('Loading pretrained model [{}] ...'.format(args.model)) 
execfile(args.load_model) 

# Set up preprocessing function with model 
pp = partial(preprocess_input, model=model) 

print ('Setting up data loaders ...')
train_images = [os.path.join(args.data_dir, _.replace('dcm', 'jpg')) for _ in train_df.dicom_file] 
train_set = CXRDataset(imgfiles=train_images,
                       labels=train_df.label,
                       preprocess=pp, 
                       transform=train_aug)
train_gen = DataLoader(train_set, **params) 

valid_images = [os.path.join(args.data_dir, _.replace('dcm', 'jpg')) for _ in valid_df.dicom_file] 
valid_set = CXRDataset(imgfiles=valid_images,
                       labels=valid_df.label,
                       preprocess=pp)
valid_gen = DataLoader(valid_set, **params) 

# Calculate inverse frequency weights based on training data distribution 
weights = [] 
weights.append(1. / (1. - np.mean(train_df.label)))
weights.append(1. / np.mean(train_df.label))

weights = np.asarray(weights) 
#weights *= args.nb_classes / float(np.sum(weights))

criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).type('torch.FloatTensor')).cuda()
optimizer = optim.Adam(model.parameters(), 
                       lr=args.initial_lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                 factor=args.annealing_factor, 
                                                 patience=args.lr_patience, 
                                                 threshold=args.min_delta, 
                                                 threshold_mode='abs', 
                                                 verbose=True)

best_auc   = 0. 
stopping   = 0
num_epochs = 0 ; steps = 0 ; steps_per_epoch = args.steps_per_epoch
start_time = datetime.datetime.now() 
print ('TRAINING : START')
while num_epochs < args.max_epochs: 
    running_loss = 0.
    for i, data in enumerate(train_gen):
        batch, labels = data  
        optimizer.zero_grad()
        output = model(batch.cuda())
        loss = criterion(output, labels.cuda())
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()
        steps += 1
        if steps % args.verbosity == 0:  # print every 100 mini-batches
            print('epoch {epoch}, batch {batch} : loss = {train_loss:.4f}'.format(epoch=str(num_epochs + 1).zfill(3), batch=steps, train_loss=running_loss / args.verbosity))
            running_loss = 0.
        if steps % steps_per_epoch == 0: 
            # Validate 
            with torch.no_grad():
                model = model.eval().cuda()
                val_loss = 0.
                val_y_pred = [] ; val_y_true = []
                for i, data in enumerate(valid_gen): 
                    batch, labels = data  
                    output = model(batch.cuda())
                    loss = criterion(output, labels.cuda())
                    val_loss += loss.item()
                    val_y_pred.extend(output.cpu().numpy())
                    val_y_true.extend(labels.numpy())
            val_y_pred = np.asarray(val_y_pred) 
            val_y_true = np.asarray(val_y_true) 
            val_loss /= float(len(valid_gen))
            val_auc_binary = roc_auc_score(val_y_true, val_y_pred[:,-1])
            print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, auc = {auc:.4f}'.format(epoch=str(num_epochs + 1).zfill(3), loss=val_loss, auc=val_auc_binary))
            scheduler.step(val_auc_binary)
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{arch}_{epoch}-{val_loss:.4f}-{val_auc:.4f}.pth'.format(arch=args.model.upper(), epoch=str(num_epochs + 1).zfill(3), val_loss=val_loss, val_auc=val_auc_binary)))
            model = model.train().cuda()
            # Early stopping
            if val_auc_binary > (best_auc + args.min_delta): 
                best_auc = val_auc_binary
                stopping = 0 
            else: 
                stopping += 1 
            if stopping >= args.stop_patience: 
                num_epochs = args.max_epochs
                break 
            num_epochs += 1
            steps = 0 
print ('TRAINING : END') 
print ('Training took {}\n'.format(datetime.datetime.now() - start_time))

           

