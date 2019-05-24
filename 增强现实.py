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
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
import pygame,pygame.image
from pygame.locals import *
import pickle
import objloader
def set_projection_from_camera(K):
	''' 从照相机标定矩阵中获得视图'''

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()

	fx=K[0,0]
	fy=K[1,1]
	fovy=4*arctan(0.5*height/fy)*180/pi
	aspect=(width*fy)/(height*fx)
	print(height)
	#定义近的和远的剪裁平面
	near=10
	far=100.0
	print(fovy)
	#print(aspect)
	#设定透视
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
	''' 从照相机姿态中获得模拟试图矩阵'''
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	#围绕x轴将茶壶旋转90度，使Z轴向上
	
	Rx=array([[1,0,0],[0,1,0],[0,0,1]])
	
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
	print(M)
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
	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)

	#创建四方形填充整个窗口
	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0);glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,0.0);glVertex3f(1.0,-1.0,-1.0)
	glTexCoord2f(1.0,1.0);glVertex3f(1.0,1.0,-1.0)
	glTexCoord2f(0.0,1.0);glVertex3f(-1.0,1.0,-1.0)
	glEnd()

	#清除纹理
	glDeleteTextures(1)

def draw_teapot(size):
	#glRotatef(0.1, 5, 5, 0)
	#print(K)
	'''在原点处绘制红色茶壶 '''
	
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)

	#绘制红色茶壶
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialfv(GL_FRONT,GL_SHININESS,0.25*128.0)
	
	glutSolidTeapot(0.02)
	
	#glutWireTeapot(size)

def setup():
	'''设置窗口和pygame环境'''
	pygame.init()
	pygame.display.set_mode((width,height),OPENGL|DOUBLEBUF)
	pygame.display.set_caption('OpenGL AR demp')

def load_and_draw_model(filename):
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)

	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.75,1.0,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)

	obj=objloader.OBJ(filename,swapyz=True)
	glCallList(obj.gl_list)

if __name__ == '__main__':
	#载入照相机数据
	fobjdir='D:\\机器学习\\图像处理学习\\'
	fobj='toyplane.obj'
	with open('ar_camera.pkl','rb')as f:
		clock = pygame.time.Clock()
		width=912
		height=684
		K=pickle.load(f)
		print(K)
		Rt=pickle.load(f)
		print(Rt)
		setup()
		#glutInit(sys.argv)
		'''
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
		glutInitWindowPosition(0,0)
		glutInitWindowSize(width,height)
		glutCreateWindow(b"First")
		glutDisplayFunc(draw_teapot)
		glutIdleFunc(draw_teapot)
		
		glutMainLoop()
		'''
		draw_background('book21.bmp')
		set_projection_from_camera(K)
		set_modelview_from_camera(Rt)
		#draw_teapot(0.02)
		load_and_draw_model('obj__fern1.obj')
		pygame.display.flip()
		pygame.time.delay(20000)
		'''
		while True:
			#clock.tick(30)
			pygame.time.delay(1000)
			event=pygame.event.poll()
			if event.type in (QUIT,KEYDOWN):
				break
			pygame.display.flip()
		'''
			