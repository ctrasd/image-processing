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
def alpha_for_triangle(points,m,n):
	alpha=zeros((m,n))
	# 判断点是否在三角形内
	# 扫描包含三角形的矩形区域
	for i in range(floor(min(points[0])),floor(max(points[0]))):
		for j in range(floor(min(points[1])),floor(max(points[1]))):
			x=linalg.solve(points,[i,j,1])
			if min(x)>0:
					alpha[i,j]=1
	return alpha	
		

def pw_affine(fromim,toim,fp,tp,tri):
	'''从一幅图像中扭曲矩形图像块
		fromim 原始图像
		toim 目标图像
		fp 扭曲前的点
		tp 扭曲后的点
		tri 三角剖分
	 '''

	im=toim.copy()
	is_color=(len(fromim.shape)==3)
	im_t=zeros(im.shape,'uint8')
	for t in tri:
		H=homography.Haffine_from_points(tp[:,t],fp[:,t])

		if is_color:
			for col in range(fromim.shape[2]):
				im_t[:,:,col]=ndimage.affine_transform(fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
		else:
			im_t=ndimage.affine_transform(fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])	
		alpha=alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])
		im[alpha>0]=im_t[alpha>0]
	return im

if __name__ == '__main__':
	'''
	x,y=array(np.random.standard_normal((2,10)))
	print(x,y)
	md=mt.Triangulation(x,y)
	figure()

	for t in md.edges:
		plot(x[[t[0],t[1]]],y[[t[0],t[1]]])
	
	tri=md.get_masked_triangles()
	for t in tri:
		t_ext=[t[0],t[1],t[2],t[0]]
		plot(x[t_ext],y[t_ext])

	plot(x,y,'*')
	show()
	'''

	fromim=array(Image.open('1.png'))
	x,y=meshgrid(range(3),range(4))
	x=(fromim.shape[1]/2)*x.flatten()
	y=(fromim.shape[0]/3)*y.flatten()
	md=mt.Triangulation(x,y)
	tri=md.get_masked_triangles()
	figure()
	
	for t in tri:
		t_ext=[t[0],t[1],t[2],t[0]]
		plot(x[t_ext],y[t_ext],'r')
	
	imshow(fromim)
	show()


	
	im=array(Image.open('chd2.jpg'))
	imshow(im)
	
	a=ginput(12)
	show()
	tp=zeros((12,2))
	tp[2,:]=1
	for i in range(12):
		for j in range(2):
			tp[i,j]=a[i][j]  

	fp=vstack((y,x,ones((1,len(x)))))
	tp=vstack((tp[:,1],tp[:,0],ones((1,len(tp)))))
	im=pw_affine(fromim,im,fp,tp,tri)
	figure()
	imshow(im)
	show()
	