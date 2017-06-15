from keras.applications.vgg19 import VGG19
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import Model

image_path = input('Enter image path\n')

img = Image.open(image_path)
print(img)