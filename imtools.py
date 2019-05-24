from PIL import Image
import os
import numpy as np
from scipy.ndimage import filters
from pylab import *
''' 返回jpg结尾的文件名'''
def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im,sz):
	''' 重新定义图像大小'''
	#im=im.reshape(1,-1)
	#print(np.shape(im))
	pil_im=Image.fromarray(im)
	print(sz)
	return array(pil_im.resize(sz))

def histeq(im,nbr_bins=265):
	'''直方图均衡化 '''
	imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)
	cdf=imhist.cumsum()
	cdf=255*cdf/cdf[-1]
	im2=interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape),cdf

def compute_average(imlist):
	'''计算图像列表的平均图像'''

	#打开第一幅图像，将其存储在浮点型数组中
	averageim=np.array(Image.open(imlist[0]),'f')

	for imname in imlist[1:]:
		try:
			im2=np.array(Image.open(imname))
			averageim=averageim+im2
		except :
			print(imname,'...skiped')
		
	averageim/=len(imlist)

	return array(averageim,'uint8')

def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
	''' 
	ROF去噪
	输入:含有噪声的灰度图，U的初始值，TV正则权值，步长，停业条件
	输出:去噪和去除纹理后的图像，纹理残留
	'''

	n,m=im.shape
	U=U_init
	Px=im # 对偶域的x分量
	Py=im # 对偶域的y分量
	error=1
	while (error>tolerance):
		Uold=U
		GradUx=roll(U,-1,axis=1)-U # 变量U梯度的x分量
		GradUy=roll(U,-1,axis=0)-U # 变量U梯度的y分量
		# 更新对偶变量
		PxNew=Px+(tau/tv_weight)*GradUx
		PyNew=Py+(tau/tv_weight)*GradUy
		NormNew=maximum(1,sqrt(PxNew**2+PyNew**2))

		Px=PxNew/NormNew
		Py=PyNew/NormNew
		RxPx=roll(Px,1,axis=1)# 对x分量进行右x轴平移
		RyPy=roll(Py,1,axis=0)#	对y分量进行右y轴平移
		
		DivP=(Px-RxPx)+(Py-RyPy) # 对偶域的散度
		U=im+tv_weight*DivP # 更新原始变量

		# 更新误差
		error=linalg.norm(U-Uold)/sqrt(n*m)
	return U,im-U

def grad(im):
	'''
	返回图像梯度
	'''
	imx=zeros(im.shape)
	filters.sobel(im,1,imx)
	imy=zeros(im.shape)
	filters.sobel(im,0,imy)
	return imx,imy

	