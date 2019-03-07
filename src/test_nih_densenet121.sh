###################
# NIH DENSENET121 #
###################

### 
# NIH on NIH 
###

# Split 0
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split0/DENSENET121_009-1.2198-0.8325-0.9008.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_densenet121_split0.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 1
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split1/DENSENET121_007-1.0604-0.8367-0.8961.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_densenet121_split1.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 2
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split2/DENSENET121_008-1.0181-0.8286-0.9013.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_densenet121_split2.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

### 
# NIH on RIH
###

# Split 0
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split0/DENSENET121_009-1.2198-0.8325-0.9008.pth \
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_densenet121_split0.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 1
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split1/DENSENET121_007-1.0604-0.8367-0.8961.pth \
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_densenet121_split1.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 2
python test.py densenet121 \
    ../train/load_densenet121.py \
    ../../checkpoints/nih/densenet121/split2/DENSENET121_008-1.0181-0.8286-0.9013.pth\
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_densenet121_split2.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15


