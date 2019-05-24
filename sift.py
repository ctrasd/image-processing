from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt



def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	""" process an image and save the results in a file"""
 
	if imagename[-3:] != 'pgm':
		#create a pgm file
		im = Image.open(imagename).convert('L')
		im.save('tmp.pgm')
		imagename = 'tmp.pgm'
 
	cmmd = str("C:/Users/ctr/AppData/Local/Programs/Python/Python36/Lib/site-packages/win64vlfeat/sift.exe "+imagename+" --output="+resultname+
				" "+params)
	os.system(cmmd)
	print ('processed', imagename, 'to', resultname)

def read_features_from_file(filename):
	''' 读取特征属性值,然后以矩阵形式返回 '''

	f=np.loadtxt(filename)
	return f[:,:4],f[:,4:]  # 位置与描述子

def write_features_to_file(filename,locs,desc):
	''' 将特征位置与描述子保存至文件'''
	np.savetxt(filename,hstack((locs,desc)))

def plot_features(im,locs,circle=True):
	''' 输入:图像，每个特征的行列尺幅和方向角度'''

	def draw_circle(c,r):
		t=arange(0,1.01,.01)*2*pi
		x=r*cos(t)+c[0]
		y=r*sin(t)+c[1]
		plot(x,y,'b',linewidth=2)

	imshow(im)
	if circle:
		for p in locs:
			draw_circle(p[:2],p[2])
	else:
		plot(locs[:,0],locs[:,1],'ob')

	axis('off')

def match(desc1,desc2):
	desc1=array([d/linalg.norm(d) for d in desc1])
	desc2=array([d/linalg.norm(d) for d in desc2])
	dist_ratio=0.9
	desc1_size=desc1.shape

	matchscores=zeros((desc1.shape[0],1),'int')
	desc2t=desc2.T
	for i in range(desc1.shape[0]):
		dotprods=dot(desc1[i,:],desc2t)
		dotprods=0.99*dotprods
		indx=argsort(arccos(dotprods))

		if arccos(dotprods)[indx[0]]<dist_ratio*arccos(dotprods)[indx[1]]:
			matchscores[i]=int(indx[0])
	return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
	
	matches_12=match(desc1,desc2)
	matches_21=match(desc2,desc1) 

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
	#print(locs1.shape)
	cols1=im1.shape[1]
	for i,m in enumerate(matchscores):
		if m>0:
			plot([locs1[i,0],locs2[m,0]+cols1],[locs1[i,1],locs2[m,1]],'c')


if __name__ == '__main__':
	imname='V_0144_06.jpg'
	im1=array(Image.open(imname).convert('L'))
	process_image(imname,'1.sift')
	l1,d1=read_features_from_file('1.sift')
	print(im1.shape)
	imname='V_0144_14.jpg'
	im2=array(Image.open(imname).convert('L'))
	process_image(imname,'2.sift')
	l2,d2=read_features_from_file('2.sift')
	print(l2.shape)
	print('start')
	matchscores=match_twosided(d1,d2) 
	#print(matchscores)
	figure()
	gray()
	#plot_features(im1,l1,circle=True)
	plot_matches(im1,im2,l1,l2,matchscores)
	show()

	'''
   
	
	
	
	'''