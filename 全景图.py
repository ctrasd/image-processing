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
import sift
import homography
'''
l[i] x,y ......
'''
def convert_points(j):
	ndx=matches[j].nonzero()[0]  #j+1图 中的有对应点的点
	fp=homography.make_homog(l[j+1][ndx,:2].T)
	ndx2=[int(matches[j][i]) for i in ndx] # fp的对应点
	tp=homography.make_homog(l[j][ndx2,:2].T)
	return fp,tp

def panorama(H,fromim,toim,padding=2400,delta=2400):
	'''
	结果为一幅和toim具有相同高度的图像,
	padding指定填充像素的数目，delta指定平移量
	'''

	is_color=len(fromim.shape)==3
	def transf(p):
		p2=dot(H,[p[0],p[1],1])
		return (p2[0]/p2[2],p2[1]/p2[2])

	if H[1,2]<0: # fromim在右边
		print('warp-right')
		if is_color:
			#图像右边填充0
			toim_t=hstack((toim,zeros((toim.shape[0],padding,3))))
			fromim_t=zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
			
			for col in range(3):
				fromim_t[:,:,col]=ndimage.geometric_transform(fromim[:,:,col]
					,transf,(toim.shape[0],toim.shape[1]+padding))
		else:
			#在目标图像右边填充0
			toim_t=hstack((toim,zeros((toim.shape[0],padding))))
			fromim_t=zeros((toim.shape[0],toim.shape[1]+padding))
			fromim_t=ndimage.geometric_transform(fromim
					,transf,(toim.shape[0],toim.shape[1]+padding))
	else:
		print('warp-left')
		#为了补偿填充效果,在左边加入平移量
		H_dalta=array([[1,0,0],[0,1,-delta],[0,0,1]])
		H=dot(H,H_dalta)
		if is_color:
			#在图像左边填充0
			toim_t=hstack((zeros((toim.shape[0],padding,3)),toim))
			fromim_t=zeros((toim.shape[0],toim.shape[1]+padding,3))
			for col in range(3):
				fromim_t[:,:,col]=ndimage.geometric_transform(fromim[:,:,col],
					transf,(toim.shape[0],toim.shape[1]+padding))
		else:
			#在图像左边填充0
			fromim_t=zeros((toim.shape[0],toim.shape[1]+padding))
			toim_t=hstack((zeros((toim.shape[0],padding)),toim))
			fromim_t=ndimage.geometric_transform(fromim,
					transf,(toim.shape[0],toim.shape[1]+padding))
	im=toim_t.copy()
	if is_color:
		alpha=((fromim_t[:,:,0]*fromim_t[:,:,1]*fromim_t[:,:,2]>0))
		for col in range(3):
			im[:,:,col]=toim_t[:,:,col]*(1-alpha)
			toim_t[:,:,col]=fromim_t[:,:,col]*alpha+toim_t[:,:,col]*(1-alpha)
			
	else:
		alpha=(fromim_t>0)
		im=toim_t[:,:]*(1-alpha)
		toim_t=fromim_t*alpha+toim_t*(1-alpha)
		
	return toim_t,im

if __name__ == '__main__':
	featname=['Univ'+str(i+1)+'11.sift' for i in range(3)]
	imname=['Univ'+str(i+1)+'11.jpg' for i in range(3)]
	l={}
	d={}
	for i in range(3):
		#sift.process_image(imname[i],featname[i])
		l[i],d[i]=sift.read_features_from_file(featname[i])
	matches={}
	for i in range(2):
		matches[i]=sift.match(d[i+1],d[i])
		'''
		matches[0] d[1]与d[0]的对应点
		'''

	im={}
	for i in range(3):
		im[i]=array(Image.open(imname[i]).convert('L'))
	
	'''
	#画图
	matchess={}
	for i in range(3):
		#matches[i]=sift.match(d[i],d[(i+1)%3])
		matchess[i]=sift.match(d[i],d[(i+1)%3])
		figure()
		sift.plot_matches(im[i],im[(i+1)%3],l[i],l[(i+1)%3],matchess[i])
		show()
	'''
	model=homography.RansacModel()

	fp,tp=convert_points(0)
	#H_01=homography.H_from_ransac(fp,tp,model)[0] # im0 到im1的单应性矩阵
	H_01, status = cv2.findHomography(fp[:][:2].T.reshape(-1,1,2),tp[:][:2].T.reshape(-1,1,2),cv2.RANSAC,4)
	tp,fp=convert_points(1)
	#H_21=homography.H_from_ransac(fp,tp,model)[0] # im2到im1的单应性矩阵
	H_21, status = cv2.findHomography(fp[:][:2].T.reshape(-1,1,2),tp[:][:2].T.reshape(-1,1,2),cv2.RANSAC,4)
	print(H_01)
	print(H_21)

	delta=2000
	im_01,imt=panorama(H_01,im[0],im[1],delta,delta)
	figure()
	imshow(im_01)
	show()
	figure()
	imshow(imt)
	show()
	im_21,imt=panorama(H_21,im[2],im_01,delta,delta)
	figure()
	#imshow(im_01)
	imshow(im_21)
	show()
	figure()
	imshow(imt)
	show()