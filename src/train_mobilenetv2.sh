python train.py mobilenetv2 \
    load_mobilenetv2.py \
    ../../data/resize-256 \
    ../../checkpoints/mobilenetv2/split0 \
    ../../nih_data_splits.csv \
    0 \
    --gpu 2 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400

python train.py mobilenetv2 \
    load_mobilenetv2.py \
    ../../data/resize-256 \
    ../../checkpoints/mobilenetv2/split1 \
    ../../nih_data_splits.csv \
    1 \
    --gpu 2 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400
  
python train.py mobilenetv2 \
    load_mobilenetv2.py \
    ../../data/resize-256 \
    ../../checkpoints/mobilenetv2/split2 \
    ../../nih_data_splits.csv \
    2 \
    --gpu 2 \
    --batch-size 32 \
    --steps-per-epoch 2000 \
    --verbosity 400

    