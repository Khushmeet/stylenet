from keras.applications.vgg19 import VGG19
from PIL import Image
import numpy as np
import tensorflow as tf
from pprint import pprint
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b

img_height, img_width, channel = 224, 224, 3


image_path = input('Enter content image path\n')
content_img = Image.open(image_path)
content_img = content_img.resize((img_width, img_height))

image_path = input('Enter style image path\n')
style_img = Image.open(image_path)
style_img = style_img.resize((img_width, img_height))

# preprocessing
content_img = np.asarray(content_img, dtype=np.float32)
content_img = np.expand_dims(content_img, axis=0)

style_img = np.asarray(style_img, dtype=np.float32)
style_img = np.expand_dims(style_img, axis=0)

content_img[:, :, :, 0] -= 103.939
content_img[:, :, :, 1] -= 116.779
content_img[:, :, :, 2] -= 123.68

style_img[:, :, :, 0] -= 103.939
style_img[:, :, :, 1] -= 116.779
style_img[:, :, :, 2] -= 123.68

content_img = content_img[:, :, :, ::-1]
style_img = style_img[:, :, :, ::-1]

content_var = tf.Variable(content_img)
style_var = tf.Variable(style_img)
output_img = tf.placeholder(shape=[1, img_width, img_height, 3], dtype=tf.float32)

input = tf.concat([content_var, style_var, output_img], axis=0)

vgg19 = VGG19(input_tensor=input, include_top=False, weights='imagenet')
