from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt


#第o族第i幅：    sigma=1.6   (2^(o + i/3))*sigma
 

if __name__ == '__main__':
	im=array(Image.open('6.jpg').convert('L'))
	imx=zeros(im.shape)
	imy=zeros(im.shape)
	imm=zeros(im.shape)
	'''
	filters.gaussian_filter(im,(5,5),(0,1),imx)
	filters.gaussian_filter(im,(5,5),(1,0),imy)
	
	imm=imx+imy
	cv2.namedWindow("imm", cv2.WINDOW_NORMAL);
	cv2.imshow('imm',imm)
	cv2.namedWindow("imx", cv2.WINDOW_NORMAL);
	cv2.imshow('imx',imx)
	cv2.namedWindow("imy", cv2.WINDOW_NORMAL);
	cv2.imshow('imy',imy)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	'''
	imm=filters.gaussian_filter(im,5)
	cv2.namedWindow("imm", cv2.WINDOW_NORMAL);
	cv2.imshow('imm',imm)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 