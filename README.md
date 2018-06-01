# Captcha

A small coding challenge to break a 5-digit captcha.

## Requirements

pytorch 	>= 4.0  
numpy 		>= 1.14  
PIL 		>= 5.0  
python 		>= 3.6  

Optional (confusion matrix plot):  
sklearn		>= 0.19

## Getting started

### Train

To train the model simply run:  
`$ train.py -d <dataset> -b 32`

### Test

To evaluate a trained model run:  
`$ predict.py -m model.pth <dataset/test>`

Add -c to plot a confusion matrix.

### Resources

Clearned and splitted dataset: [captchas.tar.xz](https://drive.google.com/file/d/1PCzMXj25fGg5W6w3RedEzbgHSHEXG4R3/view?usp=sharing)  
A pretrained CNN model with sample error rate `0.9710`, and character error rate `0.9923`: [model.tar.xz](https://drive.google.com/file/d/1ykCusCNghM2D2PHGag0OoOFL4gz9Ulvn/view?usp=sharing)

