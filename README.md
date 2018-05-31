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

### Resources

Clearned and splitted dataset: [captchas.tar.xz](https://drive.google.com/file/d/1ri__TUgm5Hfpu0ygnH5PSJpSsyWx4Z6O/view?usp=sharing)  
Simple CNN model with sample error rate `0.954`, and char error rate `0.991`: [model.tar.xz](https://drive.google.com/file/d/1NFGBfmIe2oSq7TSJbfjVZCP2y9IsGQke/view?usp=sharing)

