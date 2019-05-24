from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt

def compute_harris_response(im,sigma=3):
	'''计算响应函数  '''
	imx=zeros(im.shape)
	imy=zeros(im.shape)
	filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
	filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)
	wxx=filters.gaussian_filter(imx*imx,sigma)
	wyy=filters.gaussian_filter(imy*imy,sigma)
	wxy=filters.gaussian_filter(imx*imy,sigma)

	wdet=wxx*wyy-2*wxy*wxy
	wtr=wxx+wyy

	return wdet/wtr

def get_harris_points(harrisim,min_dist=10,threshold=0.1):
	'''
	harrisim :响应图
	min_dist :分割角点与边界的最少像素数目
	'''
	corner_threshold=harrisim.max()*threshold
	harrisim_t=(harrisim>corner_threshold)*1 # 将大于门限值的点记为1

	coords=array(harrisim_t.nonzero()).T
	#print(coords)
	candidate_values=[harrisim[c[0],c[1]] for c in coords]

	index=argsort(candidate_values)
	aa=[]
	allow_ha=zeros(harrisim.shape)
	for k in index:
		x=coords[k][0]
		y=coords[k][1]
		if allow_ha[x][y]==0:
			aa.append([x,y])
			for i in range(-min_dist,min_dist+1):
					for j in range(-min_dist,min_dist+1):
							if(i>=0 and i<harrisim.shape[0] and j>=0 and j<harrisim.shape[1]):
								allow_ha[i][j]=1
	return aa

def plot_harris_points(image,coords):
	
	figure()
	gray()
	imshow(image)
	plot([p[1] for p in coords],[p[0] for p in coords],'.') #画图时的坐标与矩阵相反
	axis('off')
	show()
	
	'''
	for [x,y] in coords:
		cv2.circle(image,(y,x),5,(120))
	#cv2.circle(image,(1500,1500),60,(120),-1)
	cv2.namedWindow("grad", cv2.WINDOW_NORMAL);
	cv2.imshow('grad',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()  
	'''

def get_description(image,filtered_coords,wid=5):
	
	desc=[]
	for coords in filtered_coords:
		patch=image[coords[0]-wid:coords[0]+wid+1,coords[1]-wid:coords[1]+wid+1].flatten()
		desc.append(patch)
	return desc


def match(desc1,desc2,threshold=0.5):

	n=len(desc1[0])
	d=-ones((len(desc1),len(desc2)))
	for i in range(len(desc1)):
		for j in range(len(desc2)):
			d1=(desc1[i]-mean(desc1[i]))/std(desc1[i])
			d2=(desc2[j]-mean(desc2[j]))/std(desc2[j])
			ncc_value=sum(d1*d2)/(n-1)
			if ncc_value>threshold:
				d[i,j]=ncc_value
	ndx=argsort(-d)
	matchscores=ndx[:,0]
	#print(ndx.shape)

	return matchscores

def match_twosided(desc1,desc2,threshold=0.5):
	
	matches_12=match(desc1,desc2,threshold)
	matches_21=match(desc2,desc1,threshold)

	ndx_12=where(matches_12>0)[0]	
	for n in ndx_12:
		if matches_21[matches_12[n]]!=n:
			matches_12[n]=-1
	return matches_12

def appendimages(im1,im2):
	
	rows1=im1.shape[0]
	rows2=im2.shape[0]
	''' 填充不同大小的地方'''
	if rows1<rows2:
		im1=concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
	elif rows2<rows1:
		im2=concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
	
	return concatenate((im1,im2),axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
	
	im3=appendimages(im1,im2)
	if show_below:
		im3=vstack((im3,im3))

	imshow(im3)

	cols1=im1.shape[1]
	for i,m in enumerate(matchscores):
		if m>0:
			plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')



if __name__ == '__main__':
	im1=array(Image.open('nmb.jpg').convert('L'))
	harrisim1=compute_harris_response(im1)
	#print(type(harrisim))
	coords1=get_harris_points(harrisim1,10,0.5)
	#print(len(coords))
	#plot_harris_points(im,coords)
	im2=array(Image.open('nmb2.jpg').convert('L'))

	#print(im1.shape,zeros())

	harrisim2=compute_harris_response(im2)
	coords2=get_harris_points(harrisim2,10,0.5)

	desc1=get_description(im1,coords1)
	desc2=get_description(im2,coords2)
	
	print("start")
	matches=match_twosided(desc1,desc2)

	figure()
	gray()
	plot_matches(im1,im2,coords1,coords2,matches)
	show()
	