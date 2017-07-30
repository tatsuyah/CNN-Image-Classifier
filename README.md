# Keras CNN Image Classifier

## Overview

A Deep Neural Network to classify multiclass image made with Keras.

<img src="https://github.com/tatsuyah/Keras-CNN-Image-Classifier/blob/master/images/accuracy.png">

## Requirements

  * Keras = 2.0.5 (TensorFlow backend)
  * Numpy = 1.12.1

## Usage

First, collect training and validation data and deploy it like this,
```
./data/
  train/
    pizza/
      pizza1.jpg
      pizza2.jpg
      ...
    poodle/
      poodle1.jpg
      poodle2.jpg
      ...
    rose/
      rose1.jpg
      rose2.jpg
      ...
  validation/
    pizza/
      pizza1.jpg
      pizza2.jpg
      ...
    poodle/
      poodle1.jpg
      poodle2.jpg
      ...
    rose/
      rose1.jpg
      rose2.jpg
      ...
```

and then run train script.

```
python src/train.py
```

Train script makes model and weights file to `./output/`.

To test another images, run

```
python src/predict.py
```

After training, you'll have tensorboard log in `./tf-log/`
So you can see the result

```
tensorboard --logdir=./tf-log
```
