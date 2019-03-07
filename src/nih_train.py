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
                    const=15, default=15)
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

train_aug = simple_aug(p=args.augment_p)

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

def pad_and_resize_image(img, size):
    """
    Resizes image to new_length x new_length and pads with 0. 
    """
    # First, get rid of any black border
    nonzeros = np.nonzero(img) 
    x1 = np.min(nonzeros[0]) ; x2 = np.max(nonzeros[0])
    y1 = np.min(nonzeros[1]) ; y2 = np.max(nonzeros[1])
    img = img[x1:x2, y1:y2, ...]
    pad_x  = img.shape[0] < img.shape[1] 
    pad_y  = img.shape[1] < img.shape[0] 
    no_pad = img.shape[0] == img.shape[1] 
    if no_pad: return cv2.resize(img, (size,size)) 
    grayscale = len(img.shape) == 2
    square_size = np.max(img.shape[:2])
    x, y = img.shape[:2]
    if pad_x: 
        x_diff = (img.shape[1] - img.shape[0]) / 2
        y_diff = 0 
    elif pad_y: 
        x_diff = 0
        y_diff = (img.shape[0] - img.shape[1]) / 2
    if grayscale: 
        img = np.expand_dims(img, axis=-1)
    pad_list = [(x_diff, square_size-x-x_diff), (y_diff, square_size-y-y_diff), (0,0)] 
    img = np.pad(img, pad_list, 'constant', constant_values=0)
    assert img.shape[0] == img.shape[1] 
    img = cv2.resize(img, (size, size))
    assert size == img.shape[0] == img.shape[1]
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
torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 

if not os.path.exists(args.save_dir): os.makedirs(args.save_dir) 

cxr_df = pd.read_csv(args.data_splits) 
cxr_df['Finding'] = 1 - cxr_df['No Finding'] 

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

findings = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 
            'Cardiomegaly', 'Pleural_Thickening', 'Consolidation', 
            'Pneumothorax', 'Mass', 'Nodule', 'Atelectasis', 'Effusion', 
            'Infiltration', 'Finding']

print ('Setting up data loaders ...')
train_images = [os.path.join(args.data_dir, _) for _ in train_df.pid] 
train_set = CXRDataset(imgfiles=train_images,
                       labels=np.asarray(train_df[findings]), 
                       preprocess=pp, 
                       transform=train_aug)
train_gen = DataLoader(train_set, **params) 

valid_images = [os.path.join(args.data_dir, _) for _ in valid_df.pid] 
valid_set = CXRDataset(imgfiles=valid_images,
                       labels=np.asarray(valid_df[findings]),
                       preprocess=pp)
valid_gen = DataLoader(valid_set, **params) 

# Calculate inverse frequency weights based on training data distribution 
weights = [] 
for find in findings: 
    weights.append(1. / np.mean(train_df[find]))

weights = np.asarray(weights) 
#weights *= args.nb_classes / float(np.sum(weights))

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).type('torch.FloatTensor')).cuda()
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
            val_mean_auc14 = []
            for vma in xrange(args.nb_classes - 1): 
                val_mean_auc14.append(roc_auc_score(val_y_true[:,vma], val_y_pred[:,vma]))
            val_mean_auc14 = np.mean(val_mean_auc14)
            val_auc_binary = roc_auc_score(val_y_true[:,args.nb_classes-1], val_y_pred[:,args.nb_classes-1])
            print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, auc14 = {auc14:.4f}, auc2 = {auc2:.4f}'.format(epoch=str(num_epochs + 1).zfill(3), loss=val_loss, auc14=val_mean_auc14, auc2=val_auc_binary))
            scheduler.step(val_mean_auc14+val_auc_binary)
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{arch}_{epoch}-{val_loss:.4f}-{val_auc14:.4f}-{val_auc2:.4f}.pth'.format(arch=args.model.upper(), epoch=str(num_epochs + 1).zfill(3), val_loss=val_loss, val_auc14=val_mean_auc14, val_auc2=val_auc_binary)))
            model = model.train().cuda()
            # Early stopping
            if np.mean((val_mean_auc14, val_auc_binary)) > (best_auc + args.min_delta): 
                best_auc = np.mean((val_mean_auc14, val_auc_binary)) 
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

           

