#!/bin/sh

# dataset dir
DATASET_DIR='datasets'
# TODO:
RAW='XXX'
MUTI_DIGIT='muti_digit'
SINGLE_DIGIT='single_digit'

# model dir
MODEL_DIR='model'

CheckCode() {
  if [[ $1 != 0 ]]
  then
    echo "step$2: train model fails, code: $1"
    exit 1
  else
    echo "train step$2 success"
  fi
}

# Create Train Dataset

# Step1: 获取候选框

# TODO:
# args inputdir outputdir

# Step2: 分割单字
python3 src/digit_segment.py \
  --indir $DATASET_DIR/$MUTI_DIGIT \
  --outdir $DATASET_DIR/$SINGLE_DIGIT \
  > log/train_step2.log
CheckCode $? 2

# Step3: 训练模型
python3 src/digit_reg.py \
  --inputdir $DATASET_DIR/$SINGLE_DIGIT \
  --outputdir $MODEL_DIR \
  --modeldir $MODEL_DIR \
  --mode "train" > log/train_step3.log
CheckCode $? 3
