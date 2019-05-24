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

def normalize(points):
	''' 在齐次坐标下，对点进行归一化，使最后一行为1'''
	rt=zeros(points.shape)
	idx=0
	for row in points:
		row=row/points[-1]
		#print(row)
		rt[idx]=row
		idx=idx+1
	return rt

def make_homog(points):
	'''将点集(dim*n的数组)转化为齐次坐标 '''
	return vstack((points,ones((1,points.shape[1]))))		

def H_from_points(fp,tp):
	''' 使用DLT方法，计算单应性矩阵H，使FP映射到TP。点自动进行归一化'''

	if fp.shape!=tp.shape:
		raise RuntimeError('number of points do not match')

	#对点进行归一化 ：减去均值，处以标准差
	#映射起始点
	m=mean(fp[:2],axis=1) # dim中的前两维,n个点取均值
	maxstd=max(std(fp[:2],axis=1))+1e-9
	c1=diag([1/maxstd,1/maxstd,1])
	c1[0][2]=-m[0]/maxstd
	c1[1][2]=-m[1]/maxstd

	'''
	fp
	[x1	x2	x3
	 y1	y2	y3
	 w1	w2	w3]

	c1
	[1/maxstd	0			-m[0]/maxstd;
		0		1/maxstd	-m[1]/maxstd;
		0		0			1]
	'''
	fp=dot(c1,fp)

	#映射对应点
	m=mean(tp[:2],axis=1)
	maxstd=max(std(tp[:2],axis=1))+1e-9
	c2=diag([1/maxstd,1/maxstd,1])
	c2[0][2]=-m[0]/maxstd
	c2[1][2]=-m[1]/maxstd
	tp=dot(c2,tp)

	#创建用于线性方法的矩阵，对于每个对应点对，在矩阵中会出现两行数值
	nbr_correspondences=fp.shape[1]
	A=zeros((2*nbr_correspondences,9))
	for i in range(nbr_correspondences):
		A[2*i]=[-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]
			*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
		A[2*i+1]=[0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]
			*fp[1][i],tp[1][i]]

	U,S,V=linalg.svd(A)
	H=V[8].reshape((3,3))
	#反归一化
	#print(H)
	H=dot(linalg.inv(c2),dot(H,c1))	
	return H/H[2,2]

def Haffine_from_points(fp,tp):
	'''计算H,仿射变换,使得tp是fp经过仿射变换H得到的'''
	if fp.shape!=tp.shape:
		raise RuntimeError('number of points do not match')

	#对点进行归一化
	#---映射起始点
	m=mean(fp[:2],axis=1)
	maxstd=max(std(fp[:2],axis=1))+1e-9
	c1=diag([1/maxstd,1/maxstd,1])
	c1[0][2]=-m[0]/maxstd
	c1[1][2]=-m[1]/maxstd
	fp_cond=dot(c1,fp)
	#---映射对应点
	m=mean(tp[:2],axis=1)
	c2=c1.copy() #两个点集,必须都进行相同的缩放
	c2[0][2]=-m[0]/maxstd
	c2[1][2]=-m[1]/maxstd
	tp_cond=dot(c2,tp)

	A=concatenate((fp_cond[:2],tp_cond[:2]),axis=0)
	U,S,V=linalg.svd(A.T)

	tmp=V[:2].T
	B=tmp[:2]
	C=tmp[2:4]

	tmp2=concatenate((dot(C,linalg.pinv(B)),zeros((2,1))),axis=1)
	H=vstack((tmp2,[0,0,1]))

	#反归一化
	H=dot(linalg.inv(c2),dot(H,c1))
	return H/H[2,2]

def image_in_image(im1,im2,tp):
	''' 使用仿射变换将im1放在im2上,使im1的角和tp尽可能靠近,
		tp是齐次表示的,并且是按照从左上角逆时针计算的
	'''
	#扭曲的点
	m,n=im1.shape[:2]
	fp=array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

	#计算仿射变换,并且将其应用于图像im1
	H=Haffine_from_points(tp,fp)
	im1_t=ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
	alpha=(im1_t>0)
	return (1-alpha)*im2+alpha*im1_t

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
		
def image_in_image_pro(im1,im2,tp):
	m,n=im1.shape[:2]
	fp=array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

	#分成两个三角形进行拼接
	tp2=tp[:,:3]
	fp2=fp[:,:3]

	#计算H
	H=Haffine_from_points(tp2,fp2)
	im1_t=ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])

	#三角形的alpha
	alpha=alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
	im3=(1-alpha)*im2+alpha*im1_t

	#第二个三角形
	tp2=tp[:,[0,2,3]]
	fp2=fp[:,[0,2,3]]
	#计算H
	H=Haffine_from_points(tp2,fp2)
	im1_t=ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
	alpha=alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
	im4=(1-alpha)*im3+alpha*im1_t
	return im4


class RansacModel(object):
	"""docstring for RansacModel"""
	def __init__(self, debug=False):
		self.debug = debug
	def fit(self,data):
		''' 计算选取的4个对应的单应性矩阵'''
		data=data.T
		fp=data[:3,:4]
		tp=data[3:,:4]
		return Haffine_from_points(fp,tp)

	def get_error(self,data,H):
		''' 计算单应性矩阵，对每个变换后的点计算误差'''
		data=data.T
		fp=data[:3]
		tp=data[3:]
		fp_transformed=dot(H,fp)
		for i in range(3):
			fp_transformed[i]=fp_transformed[i]/fp_transformed[2]

		#print(type(tp),type(fp_transformed))
		#sqrt 会默认使用math里的，不能对矩阵进行操作，改写为np中的开方
		return np.sqrt(np.sum((tp-fp_transformed)**2,axis=0))

def H_from_ransac(fp,tp,model,maxiter=1000,match_theshold=10):
	#下载地址 https://scipy-cookbook.readthedocs.io/_downloads/ransac.py
	# 输入 fp,tp 3*n的矩阵
	import ransac

	data=vstack((fp,tp))
	#print(data.shape)
	H,ransac_data=ransac.ransac(data.T,model,4,maxiter,match_theshold,10,return_all=True)
	return H,ransac_data['inliers']


if __name__ == '__main__':
	c=zeros((3,3))
	a=array([[2,4,6],[1,2,3],[1,1,1]])
	b=array([[1,2,3],[2,4,6],[1,1,1]])
	'''
	print(a)
	print('')
	print(b)
	print('')
	print(c)
	print(Haffine_from_points(a,b))
	print(H_from_points(a,b))
	'''
	imname='2.jpg'
	im1=array(Image.open(imname).convert('L'))
	#H=array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
	#H=array([[2,0,-500],[0,3,0],[0,0,1]])
	#print(H)
	#im2=ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))
	imname='chd2.jpg'
	im2=array(Image.open(imname).convert('L'))
	figure()
	gray()
	imshow(im2)
	
	a=ginput(4)
	print(a)
	print(a[0][0])
	tp=zeros((3,4))
	tp[2,:]=1
	for i in range(2):
		for j in range(4):
			tp[i,j4]=a[j][i^1]  # matlab画图是xy相反的
	show()
	#直接变换
	im2=image_in_image_pro(im1,im2,tp)
	
	

	figure()
	gray()
	imshow(im2)
	show()