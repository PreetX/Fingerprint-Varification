import os
import numpy as np
import tensorflow as tf
import cv2
from prepare import prepare_image

def eval(model, img, saved_images, n=1):
    '''
    model: model to extract features
    img: input image (2D grey scale image)
    saved_images: target images (3D array)
    n: number of features
    '''
    confidance = 0
    for i in range(n):
        confidance += model.predict([img, saved_images[i]])

    confidance /=n
    print(confidance)
    similar = False
    if confidance > 0.8:
        similar = True
    return similar

if __name__=='__main__':
    image_width = image_height = 128
    # prepare model
    model = tf.keras.models.load_model('../model/result/fp128_128.h5')

    print('Loading images')
    filepath = '../dataset/original/00000_00.bmp'
    img1 = prepare_image(filepath, image_width, image_height).reshape((1, image_width, image_height, 1)).astype(np.float32) / 255.

    feature = np.array([img1])

    filepath = '../dataset/original/00000_03.bmp'
    img2 = prepare_image(filepath, image_width, image_height).reshape((1, image_width, image_height, 1)).astype(np.float32) / 255.

    print(eval(model, img2, feature))
