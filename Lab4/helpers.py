#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Thanks to https://martin-thoma.com/classify-mnist-with-pybrain/
import pickle
import gzip
import os
from struct import unpack

from numpy import zeros, uint8, float32
from pylab import imshow, show, cm


def get_labeled_data(image_file, label_file, pickle_name):
    """
    Read input-vector (image) and target class (label, 0-9).
    Returns dict
    """
    if os.path.isfile('%s.pickle' % pickle_name):
        return pickle.load(open('%s.pickle' % pickle_name, "rb"))

    # Open the images with gzip in read binary mode
    images = gzip.open(image_file, 'rb')
    labels = gzip.open(label_file, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('The number of labels did not match '
                        'the number of images.')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = (float(tmp_pixel) / 255)
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
    pickle.dump(data, open("%s.pickle" % pickle_name, "wb"))
    return data


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()
