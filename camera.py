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
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame,pygame.image
from pygame.locals import *



class Camera(object):
	"""docstring for Camera"""
	def __init__(self, P):
		self.P = P
		self.K=None
		self.t=None
		self.c=None

	def project(self,X):
		'''x的投影点,并且进行坐标归一化'''
		x=dot(self.P,X)
		for i in range(3):
			x[i]/=x[2]
		return x
	def factor(self):
		''' 将矩阵分解为K,R,t,其中，P=K[R|t]'''
		#分解前3*3的部分

		K,R=linalg.rq(self.P[:,:3])
		#将K的对角线元素设为正值
		T=diag(sign(diag(K)))  # 转化为对角线位+-1的对角阵
		if linalg.det(T)<0:
			T[1,1]*=-1
		self.K=dot(K,T)
		self.R=dot(T,R)# K'*R'得与K*R相等 所以K*T*T^(-1)*R,然后T的逆与T相等
		self.t=dot(linalg.inv(self.K),self.P[:,3])

		return self.K,self.R,self.t

def rotation_matrix(a):
	R=eye(4);
	R[:3,:3]=linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
	return R;


	

def center(self):
	''' 计算照相机的中心位置'''

	if self.c is not None:
		return self.c
	else:
		self.factor()
		self.c=-dot(self.R,self.t)
		return self.c

def set_projection_from_camera(K):
	''' 从照相机标定矩阵中获得视图'''

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()

	fx=K[0,0]
	fy=K[1,1]
	fovy=2*arctan(0.5*height/fy)*180/pi
	aspect=(width*fy)/(height*fx)

	#定义近的和远的剪裁平面
	near=0.1
	far=100.0

	#设定透视
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
	''' 从照相机姿态中获得模拟试图矩阵'''
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	#围绕x轴将茶壶旋转90度，使Z轴向上
	Rx=array([[1,0,0],[0,0,-1],[0,1,0]])
	
	#获得旋转的最佳逼近
	R=Rt[:,:3]
	U,S,V=linalg.svd(R)
	R=dot(U,V)
	R[0,:]=-R[0,:] #改变x轴的符号

	#获得平移量
	t=Rt[:,3]

	#获得4*4的模拟视图矩阵
	M=eye(4)
	M[:3,:3]=dot(R,Rx)
	M[:3,3]=t

	#转置并压平以获取系列数值
	M=M.T
	m=M.flatten()
	#将模拟视图矩阵替换为新的矩阵
	glLoadMatrixf(m)

def draw_background(imname):
	'''使用四边形绘制背景图像'''
	#载入背景图像(.bmp格式),转换为opengl纹理
	bg_image=pygame.image.load(imname).convert()
	bg_data=pygame.image.tostring(bg_image,"RGBX",1)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

	#绑定纹理
	glEnalbe(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
	glTexImage2D(GL_TEXTURE_2D,0,GLRGBA,width,height,0,GL_RGBA,GL_unsigned_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D,GL_Texture_MAG_ILTER,GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D,GL_Texture_MIN_ILTER,GL_NEAREST)

	#创建四方形填充整个窗口
	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,0.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,1.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(0.0,1.0);glVertex3f(-1.0,-1.0,-1.0)
	glEnd()

	#清除纹理
	glDeleteTextures(1)

def draw_teapot(size):
	'''在原点处绘制红色茶壶 '''
	glEnalbe(GL_lIGHTING)
	glEnalbe(Gl_LIGHT0)
	glEnalbe(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)

	#绘制红色茶壶
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialfv(GL_FRONT,GLSHININESS,0.25*128.0)
	glutSolidTeapot(size)


	
if __name__ == '__main__':

	fromim=array(Image.open('house.001.pgm'))
	cv2.imshow('asd',fromim)

	points=loadtxt('house.p3d').T
	print(points.shape)
	points=vstack((points,ones(points.shape[1])))

	P=hstack((eye(3),array([[0],[0],[-10]])))
	print(eye(3))
	cam=Camera(P)
	x=cam.project(points)

	figure()
	plot(x[0],x[1],'k.')
	show()

	r=0.5*np.random.rand(3)
	rot=rotation_matrix(r)
	figure()
	for i in range(10):
		cam.P=dot(cam.P,rot)
		print(cam.P)
		x=cam.project(points)
		plot(x[0],x[1],'k.')
		show()
