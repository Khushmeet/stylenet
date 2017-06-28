import numpy as np
import PIL.Image


def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    with open(filename, 'wb') as f:
        PIL.Image.fromarray(image).save(f, 'png')

