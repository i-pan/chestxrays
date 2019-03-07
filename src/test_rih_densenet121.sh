###################
# RIH DENSENET121 #
###################

### 
# RIH on NIH 
###

# Split 0
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split0/DENSENET121_004-0.4941-0.9322.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_densenet121_split0.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 1
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split1/DENSENET121_003-0.2909-0.9467.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_densenet121_split1.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 2
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split2/DENSENET121_004-0.3660-0.9396.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_densenet121_split2.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

### 
# RIH on RIH
###

# Split 0
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split0/DENSENET121_004-0.4941-0.9322.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_densenet121_split0.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 1
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split1/DENSENET121_003-0.2909-0.9467.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_densenet121_split1.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 2
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/rih/densenet121/split2/DENSENET121_004-0.3660-0.9396.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_densenet121_split2.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2
