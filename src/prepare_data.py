import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('old_image_dir', type=str) 
parser.add_argument('new_image_dir', type=str)
parser.add_argument('imsize', type=int) 

args = parser.parse_args() 

#####

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

#####

from tqdm import tqdm 

import numpy as np 
import scipy.misc 
import glob
import cv2 
import os 

if not os.path.exists(args.new_image_dir): 
    print ('Creating directory [{}] ...'.format(args.new_image_dir))
    os.makedirs(args.new_image_dir) 

print ('Saving to directory [{}] ...'.format(args.new_image_dir))

imfiles = glob.glob(os.path.join(args.old_image_dir, '*')) 

blacklist = [] 
for imfile in tqdm(imfiles, total=len(imfiles)):
    name = imfile.split('/')[-1]  
    # Force to grayscale
    im = cv2.imread(imfile, 0) 
    if im is None: 
        print ('Could not load [{}] ...'.format(name))
        blacklist.append(name)
        continue
    # Pad and resize 
    im = pad_and_resize_image(im, args.imsize) 
    cv2.imwrite(os.path.join(args.new_image_dir, name), im) 

if len(blacklist) > 0: 
    with open('blacklist.txt', 'w') as f: 
        for im in blacklist: f.write('{}\n'.format(im))
