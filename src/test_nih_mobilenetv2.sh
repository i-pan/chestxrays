###################
# NIH MOBILENETV2 #
###################

### 
# NIH on NIH 
###

# Split 0
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split0/MOBILENETV2_008-1.2075-0.8236-0.8931.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_mobilenetv2_split0.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 1
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split1/MOBILENETV2_005-0.9765-0.8264-0.8868.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_mobilenetv2_split1.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 2
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split2/MOBILENETV2_006-0.9863-0.8242-0.8945.pth \
    nih \
    nih \
    ../../data/resize-256 \
    nih_on_nih_mobilenetv2_split2.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

### 
# NIH on RIH
###

# Split 0
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split0/MOBILENETV2_008-1.2075-0.8236-0.8931.pth \
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_mobilenetv2_split0.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 1
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split1/MOBILENETV2_005-0.9765-0.8264-0.8868.pth \
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_mobilenetv2_split1.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15

# Split 2
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/nih/mobilenetv2/split2/MOBILENETV2_006-0.9863-0.8242-0.8945.pth \
    nih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    nih_on_rih_mobilenetv2_split2.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 15


