# Captcha

A small coding challenge to break a 5-digit captcha.

## Requirements

pytorch 	>= 4.0  
numpy 		>= 1.14  
PIL 		>= 5.0  
python 		>= 3.6  

## Getting started

### Train

To train the model simply run:  
`$ train.py -d <dataset> -b 32`

### Test

To evaluate a trained model run:  
`$ predict.py -m model.pth <dataset/test>`
