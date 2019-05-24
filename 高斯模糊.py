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
print(type(im2))
print(np.shape(im2)[0])
print(np.shape(im))
imm=zeros(im2.shape)

'''

for i in range(0,np.shape(im2)[0]):
	for j in range(0,np.shape(im2)[1]):
		#imm[i][j]=im[i][j]
		imm[i][j]=im[i][j]-im2[i][j]

cv2.namedWindow("h", cv2.WINDOW_NORMAL);
cv2.imshow('h',im)
cv2.namedWindow("y", cv2.WINDOW_NORMAL);
cv2.imshow('y',imm)
cv2.waitKey(0)
cv2.destroyAllWindows()  

'''

imx,imy=grad(im)
'''
cv2.namedWindow("gradx", cv2.WINDOW_NORMAL);
cv2.imshow('gradx',imx)
cv2.namedWindow("grady", cv2.WINDOW_NORMAL);
cv2.imshow('grady',imy)
'''
img=zeros(imm.shape)
flag=1
print(np.shape(imm)[0],np.shape(imm)[1])
for i in range(0,np.shape(imm)[0]):
	for j in range(0,np.shape(imm)[1]):
		img[i][j]=imx[i][j]*imx[i][j]
		if img[i][j]!=0 and flag==1:
			print(img[i][j])
		img[i][j]=sqrt(img[i][j])
		if img[i][j]!=0 and flag==1:
			print(img[i][j])
			flag=0

cv2.namedWindow("grad", cv2.WINDOW_NORMAL);
cv2.imshow('grad',img)
cv2.waitKey(0)
cv2.destroyAllWindows()  
'''
cv2.namedWindow("a", cv2.WINDOW_NORMAL);
cv2.imshow('a',im)

cv2.namedWindow("b",cv2.WINDOW_NORMAL);
cv2.imshow('b',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()  
'''

