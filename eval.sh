#!/bin/sh

# eval dir
EVAL_DIR='dataset/train'

RAW='XXX'
MUTI_DIGIT='YYY'
SINGLE_DIGIT='ZZZ'
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

if [[ $1 != "" && $2 != "" ]]
then
  RAW_DIR=$1
  OUTPUT_DIR=$2
  # step1 TODO:

  # step2
  python3 src/digit_segment.py \
    --indir $DATASET_DIR/$MUTI_DIGIT \
    --outdir $DATASET_DIR/$SINGLE_DIGIT \
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
