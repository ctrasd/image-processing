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
from PIL import Image, ImageDraw


if __name__ == '__main__':
	'''
	im1=cv2.imread('2.jpg',-1)
	cv2.line(im1, (0, 0), (300, 300),(1,21,2)) #5
	#cv2.imshow('asd',im1)
	#cv2.waitKey(0)
	cv2.imwrite('nps.jpg',im1,[int( cv2.IMWRITE_JPEG_QUALITY), 100])
	'''
	im=Image.open("2.jpg")
	draw =ImageDraw.Draw(im)
	draw.line((0,10,50,50),fill=128)
	im.show()
	im.save('nps.jpg',quality=100)

