# ticket_reg

## Setup
```
git clone https://github.com/xinshoulzc/ticket_reg.git
cd ticket_reg
sh setup.sh
```

## Train DataSet
please contact us to get image files
```
cp {downloaded files} dataset/raw
```

## Train
```
sh train.sh
```

## Eval
```
sh eval.sh 1 # price recognition
sh eval.sh 2 # barcode recognition
```

## infer
```
sh eval.sh 1 YOUR_IMAGE_DIR OUTPUT_DIR # price recognition
sh eval.sh 2 YOUR_IMAGE_DIR OUTPUT_DIR # barcode recognition
```
