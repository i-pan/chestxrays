python train.py densenet121 \
    load_densenet121.py \
    ../../data/resize-256 \
    ../../checkpoints/densenet121/split0 \
    ../../nih_data_splits.csv \
    0 \
    --gpu 3 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400

python train.py densenet121 \
    load_densenet121.py \
    ../../data/resize-256 \
    ../../checkpoints/densenet121/split1 \
    ../../nih_data_splits.csv \
    1 \
    --gpu 3 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400
  
python train.py densenet121 \
    load_densenet121.py \
    ../../data/resize-256 \
    ../../checkpoints/densenet121/split2 \
    ../../nih_data_splits.csv \
    2 \
    --gpu 3 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400

    