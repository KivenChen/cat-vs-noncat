''' this file adds some preprocessing features to images '''

import numpy as np
import tensorflow as tf
from tensorflow import image as tfimg
from keras.preprocessing import image as kimg
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from wheels import *
import random as rd

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb


def random_img_preproc_pil(x, bright=True, contrast=True, lr_flip=True, hue=True, saturn=True):
    if x is not None:
        rv = None
        num = x.shape[0]
        counter = 1
        blue("Kreprocessing: started to process " + str(num) + ' images')
        sess = tf.Session()
        for i in x:
            img = Image.fromarray(i)
            if bright:
                img = ImageEnhance.Brightness(img).enhance(1.5-rd.random())
            if contrast:
                img = ImageEnhance.Contrast(img).enhance(1.5-rd.random())
            if lr_flip and rd.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if hue:
                arr = np.asarray(img.convert('rgba'))
                img = shift_hue(arr, rd.random())
                img = Image.fromarray(img, 'rgba')
            if saturn:
                # img = Image.fromarray(img)
                img = ImageEnhance.Color(img).enhance(1.5-rd.random())
            raw = np.asarray(img).reshape(1, 64, 64, 3)
            ''' if rv is None:
                rv = raw.copy()
            else:
                rv = np.concatenate((rv, raw), axis=0)'''
            if rv is None:
                rv = [raw.copy()]
            else:
                rv.append(raw)
            flush(counter, ' / ', num)
            counter += 1
        print()
        sess.close()
    return np.concatenate(tuple(rv), axis=0)

def random_img_preproc(x, bright=True, contrast=True, lr_flip=True, hue=True, saturn=True):
    if x is not None:
        rv = None
        num = x.shape[0]
        counter = 1
        blue("Kreprocessing: started to process " + str(num) + ' images')
        config = tf.ConfigProto(device_count={"CPU": 6, "GPU": 0},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=12,
                                log_device_placement=True)
        sess = tf.Session(config=config)
        for i in x:
            img = kimg.array_to_img(i)
            if bright:
                brighten = tfimg.random_brightness(img, 0.2)
                img = sess.run(brighten)
            if contrast:
                contr = tfimg.random_contrast(img, 0.5, 1.5)
                img = sess.run(contr)
            if lr_flip:
                flip = tfimg.random_flip_left_right(img)
                img = sess.run(flip)
            if hue:
                adj_hue = tfimg.random_hue(img, 0.2)
                img = sess.run(adj_hue)
            if saturn:
                sat = tfimg.random_saturation(img, 0.3, 1.7)
                img = sess.run(sat)
            raw = np.asarray(img).reshape(1, 64, 64, 3)
            ''' if rv is None:
                rv = raw.copy()
            else:
                rv = np.concatenate((rv, raw), axis=0)'''
            if rv is None:
                rv = [raw.copy()]
            else:
                rv.append(raw)
            flush(counter, ' / ', num)
            counter += 1
        print()
        sess.close()
    return np.concatenate(tuple(rv), axis=0)


def random_img_preproc_gpu(x, bright=True, contrast=True, lr_flip=True, hue=True, saturn=True):
    if x is not None:
        rv = None
        num = x.shape[0]
        counter = 1
        blue("Kreprocessing: started to process " + str(num) + ' images')
        sess = tf.Session()
        for i in x:
            img = kimg.array_to_img(i)
            if bright:
                brighten = tfimg.random_brightness(img, 0.2)
                img = sess.run(brighten)
            if contrast:
                contr = tfimg.random_contrast(img, 0.5, 1.5)
                img = sess.run(contr)
            if lr_flip:
                flip = tfimg.random_flip_left_right(img)
                img = sess.run(flip)
            if hue:
                adj_hue = tfimg.random_hue(img, 0.2)
                img = sess.run(adj_hue)
            if saturn:
                sat = tfimg.random_saturation(img, 0.3, 1.7)
                img = sess.run(sat)
            raw = np.asarray(img).reshape(1, 64, 64, 3)
            ''' if rv is None:
                rv = raw.copy()
            else:
                rv = np.concatenate((rv, raw), axis=0)'''
            if rv is None:
                rv = [raw.copy()]
            else:
                rv.append(raw)
            flush(counter, ' / ', num)
            counter += 1
        print()
        sess.close()
    return np.concatenate(tuple(rv), axis=0)

def test():
    totest = Image.open("images\\Z.jpeg")
    totest.load()
    totest = totest.resize((64, 64), Image.ANTIALIAS)
    raw = kimg.img_to_array(totest).reshape((1,64,64,3))
    raw = random_img_preproc(raw, bright=False)
    plt.figure()
    plt.imshow(raw[0])
    plt.show()
    plt.close()


if __name__ == "__main__":
    test()
