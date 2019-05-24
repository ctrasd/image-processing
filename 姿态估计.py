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
import pickle

def cube_points(c,wid):
	''' 创建用于绘制方体的点列表 '''
	p=[]

	#底面
	p.append([c[0]-wid,c[1]-wid,c[2]-wid])
	p.append([c[0]-wid,c[1]+wid,c[2]-wid])
	p.append([c[0]+wid,c[1]+wid,c[2]-wid])
	p.append([c[0]+wid,c[1]-wid,c[2]-wid])
	p.append([c[0]-wid,c[1]-wid,c[2]-wid])

	#顶部
	p.append([c[0]-wid,c[1]-wid,c[2]+wid])
	p.append([c[0]-wid,c[1]+wid,c[2]+wid])
	p.append([c[0]+wid,c[1]+wid,c[2]+wid])
	p.append([c[0]+wid,c[1]-wid,c[2]+wid])
	p.append([c[0]-wid,c[1]-wid,c[2]+wid])

	#竖直边
	p.append([c[0]-wid,c[1]-wid,c[2]+wid])
	p.append([c[0]-wid,c[1]+wid,c[2]+wid])
	p.append([c[0]-wid,c[1]+wid,c[2]-wid])
	p.append([c[0]+wid,c[1]+wid,c[2]-wid])
	p.append([c[0]+wid,c[1]+wid,c[2]+wid])
	p.append([c[0]+wid,c[1]-wid,c[2]+wid])
	p.append([c[0]+wid,c[1]-wid,c[2]-wid])

	return array(p).T

'''185 260   dz=270  2100 2700 '''
def my_calibratio(sz):
	row,col=sz
	fx=3064*col/3648
	fy=2803*row/2736
	K=diag([fx,fy,1])
	K[0,2]=0.5*col
	K[1,2]=0.5*row
	return K

if __name__ == '__main__':
	'''
	im=array(Image.open('book.jpg').convert('L'))
	figure()
	imshow(im)
	a1=ginput()
	a2=ginput()
	show()
	print(a1,a2)
	'''
	#绘制3d图
	'''
	p=cube_points([0,0,0],2)
	fig=plt.figure()
	ax=Axes3d(fig)
	ax.plot(p[0],p[1],p[2],'r')
	'''
	
	#sift.process_image('book11.jpg','book1.sift')
	l0,d0=sift.read_features_from_file('book1.sift')
	im1=array(Image.open('book11.jpg').convert('L'))
	im2=array(Image.open('book21.jpg').convert('L'))
	#sift.process_image('book21.jpg','book2.sift')
	l1,d1=sift.read_features_from_file('book2.sift')
	'''
	print(im1.shape)
	print(im2.shape)
	print(l1.shape)
	print(l0,shape)
	'''
	matches=sift.match_twosided(d0,d1)
	ndx=matches.nonzero()[0]
	fp=homography.make_homog(l0[ndx,:2].T)   # x1 x2……
	ndx2=[int(matches[i])for i in ndx]       # y1 y2……
	tp=homography.make_homog(l1[ndx2,:2].T)
	print(fp.shape)
	print(tp.shape)
	figure()
	sift.plot_matches(im1,im2,l0,l1,matches)
	show()

	model=homography.RansacModel()
	#H,qnm=homography.H_from_ransac(fp,tp,model)
	print(fp[:][:2].shape)
	H, status = cv2.findHomography(fp[:][:2].T.reshape(-1,1,2),tp[:][:2].T.reshape(-1,1,2),cv2.RANSAC,4)
	print(H)
	H=array(H)
	print(H.shape)
	print(type(H))
	
	K=my_calibratio((912,684))
	box=cube_points([0,0,0.1],0.1)
	#投影第一幅图像
	cam1=camera.Camera(hstack((K,dot(K,array([[0],[0],[-1]])))))
	#底部正方形上的点
	box_cam1=cam1.project(homography.make_homog(box[:,:5]))
	#从第一幅图像变换到第二幅图像
	print(type(box_cam1))
	print('h*box_cam1')
	print(dot(H,box_cam1))
	box_trans=homography.normalize(dot(H,array(box_cam1)))

	#从cam1和H中计算第二个照相机矩阵
	cam2=camera.Camera(dot(H,cam1.P))
	A=dot(linalg.inv(K),cam2.P[:,:3])
	A=array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
	K=array(K)
	cam2.P[:,:3]=array(dot(array(K),array(A)))

	#使用第二个照相机矩阵投影
	box_cam2=cam2.project(homography.make_homog(box))
	'''
	point=array([1,1,0,1]).T
	print(homography.normalize(dot(dot(H,cam1.P),point)))
	print(cam2.project(point))
	'''
	figure()
	imshow(im1)
	plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
	show()
	#print(box_cam1)

	figure()
	imshow(im2)
	plot(box_trans[0,:],box_trans[1,:],linewidth=3)
	show()
	
	#print(box_trans)
	
	figure()
	imshow(im2)
	plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
	show()
	#print(box_cam2)
	print(K)
	#print(A)
	print(dot(inv(K),cam2.P))
	with open('ar_camera_rubbit.pkl','wb')as f:
		pickle.dump(K,f)
		pickle.dump(dot(inv(K),cam2.P),f)