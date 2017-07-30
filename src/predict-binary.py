import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    print("Predicted answer: Pizza")
    answer = 'pizza'
  else:
    print("Predicted answer: Poodle")
    answer = 'poodle'

  return answer

tp = 0
tn = 0
fp = 0
fn = 0

for i, ret in enumerate(os.walk('./test-data/poodle')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Poodle")
    result = predict(ret[0] + '/' + filename)
    if result == "poodle":
      tn += 1
    else:
      fp += 1

for i, ret in enumerate(os.walk('./test-data/pizza')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Pizza")
    result = predict(ret[0] + '/' + filename)
    if result == "pizza":
      tp += 1
    else:
      fn += 1

"""
Check metrics
"""
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)  # important
print("False Negative: ", fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Precision: ", precision)
print("Recall: ", recall)

f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
