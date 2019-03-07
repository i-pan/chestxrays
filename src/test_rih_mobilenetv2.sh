###################
# RIH MOBILENETV2 #
###################

### 
# RIH on NIH 
###

# Split 0
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split0/MOBILENETV2_003-0.4625-0.9302.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_mobilenetv2_split0.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 1
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split1/MOBILENETV2_002-0.3152-0.9381.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_mobilenetv2_split1.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 2
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split2/MOBILENETV2_002-0.3549-0.9372.pth \
    rih \
    nih \
    ../../data/resize-256 \
    rih_on_nih_mobilenetv2_split2.csv \
    ../../nih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

### 
# RIH on RIH
###

# Split 0
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split0/MOBILENETV2_003-0.4625-0.9302.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_mobilenetv2_split0.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 1
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split1/MOBILENETV2_002-0.3152-0.9381.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_mobilenetv2_split1.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2

# Split 2
python test.py mobilenetv2 \
    ../train/load_mobilenetv2.py \
    ../../checkpoints/rih/mobilenetv2/split2/MOBILENETV2_002-0.3549-0.9372.pth \
    rih \
    rih \
    /users/ipan/dmerck/anon/chest_cr_anon/resize-256 \
    rih_on_rih_mobilenetv2_split2.csv \
    ../../rih_data_splits.csv \
    --gpu 0 \
    --nb-classes 2
