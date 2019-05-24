from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt
import pydot
from scipy import ndimage
from math import *
#import matplotlib.delaunay as md
from scipy.spatial import Delaunay
import matplotlib.tri as mt
import homography
from scipy import linalg
import sift
import camera
from mpl_toolkits.mplot3d import axes3d


if __name__ == '__main__':
	im1=cv2.imread('ctr1.jpg',-1)
	size=(int(im1.shape[1]*0.25),int(im1.shape[0]*0.25))
	print(size)
	im11=cv2.resize(im1,size,interpolation=cv2.INTER_AREA)
	cv2.imshow('asd',im11)
	cv2.waitKey(0)
	cv2.imwrite('ctr111.jpg',im11, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

	im1=cv2.imread('ctr2.jpg',-1)
	size=(int(im1.shape[1]*0.25),int(im1.shape[0]*0.25))
	print(size)
	im11=cv2.resize(im1,size,interpolation=cv2.INTER_AREA)
	cv2.imshow('asd',im11)
	cv2.waitKey(0)
	cv2.imwrite('ctr211.jpg',im11, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
	'''
	im1=array(Image.open('book11.jpg').convert('L'))
	im2=array(Image.open('book21.jpg').convert('L'))
	print(im1.shape)
	print(im2.shape)
	'''
