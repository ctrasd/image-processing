from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt

im=array(Image.open('6.jpg').convert('L'))
im2=filters.gaussian_filter(im,5)
im3=filters.gaussian_filter(im,10)
im4=filters.gaussian_filter(im,15)
#im22=contour(im2,origin='image')
#im33=contour(im3,origin='image')
#print(type(im3))
#cv2.imshow('5',im3)

figure()
gray()
contour(im3,origin='image')
figure(num=3)
axis('equal')
axis('off')
contour(im2,origin='image')
show()


