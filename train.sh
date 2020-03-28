#!/bin/sh

# dataset dir
DATASET_DIR='datasets'
RAW='raw'
MUTI_DIGIT='multi_digit'
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

# Step1: 获取人工标注候选框
echo "train step 1 skip, use selected figures"

# Step1: 获取候选框
# python3 src/get_CNY_area.py \
#   --indir $DATASET_DIR/$RAW \
#   --outdir $DATASET_DIR/$MULTI_DIGIT \
#   > log/train_step1.log
# CheckCode $? 1

# Step2: 分割单字

echo "errors in log/train_step2.log"
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
