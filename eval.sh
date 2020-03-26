#!/bin/sh

DATASET_DIR='datasets'

# eval dir
EVAL_DIR='datasets/train'
ID=1

RAW='raw'
MULTI_DIGIT='multi_digit'
SINGLE_DIGIT='single_digit'
RESULT='result'

# model dir
MODEL_DIR='model'

CheckCode() {
  if [[ $1 != 0 ]]
  then
    echo "step$2: infer model fails, code: $1"
    exit 1
  else
    echo "infer step$2 success"
  fi
}

if [[ $1 != "" ]]
then
  ID=$1
else
  echo "error: no ID provided"
  exit 1
fi

if [[ $2 != "" && $3 != "" ]]
then
  RAW=$2
  OUTPUT_DIR=$3
  # step1 :
  if [[ $1 == "1" ]]
  then
  python3 src/get_CNY_area.py \
    --indir $DATASET_DIR/$RAW \
    --outdir $DATASET_DIR/$MULTI_DIGIT \
    > log/infer_step1.log
  elif [[ $1 == "2" ]]
  then
    python3 src/get_barcode_area.py \
    --indir $DATASET_DIR/$RAW \
    --outdir $DATASET_DIR/$MULTI_DIGIT \
    > log/infer_step1.log
  fi
  CheckCode $? 1

  # step2
  python3 src/digit_segment.py \
    --indir $DATASET_DIR/$MULTI_DIGIT \
    --outdir $DATASET_DIR/$SINGLE_DIGIT \
    --mode ID \
    > log/infer_step2.log
  CheckCode $? 2

  # step3
  python3 src/digit_reg.py \
    --inputdir $EVAL_DIR/$SINGLE_DIGIT \
    --outputdir $OUTPUT_DIR/$RESULT \
    --modeldir $MODEL_DIR \
    --mode "eval" > log/infer_step3.log
  CheckCode $? 3
else
  python3 src/digit_reg.py \
    --inputdir $EVAL_DIR \
    --outputdir $MODEL_DIR \
    --modeldir $MODEL_DIR \
    --mode "eval" > log/eval.log
fi
