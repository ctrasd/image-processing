from PIL import Image
import os
from imtools import *
from pylab import *

im= array(Image.open('1.jpg').convert('L'))
print(im[0][0])
im2,cdf=histeq(im)
imshow(im2)
x=ginput(3)
#im= Image.fromarray(im)
#print(type(im))
#print(shape(im)[0],shape(im)[1])
#im=imresize(im,[shape(im)[1],shape(im)[0]])
#imshow(im)
print(im[1,1])
'''
画线与点
x=[100,100,400,400]
y=[200,500,100,333]

plot(x,y,'r*')
plot(x[:2],y[:2])
plot([300,200],[100,500])
show()
'''

'''
轮廓，直方图
figure()
gray()
contour(im,origin='image')
axis('equal')
axis('off')
show()

figure()
hist(im.flatten(),128)
show()
'''

'''
交互式操作
imshow(im)
print('kick 3 points')
x=ginput(3)
print('you kick :',x)
show()
'''

im2=255-im

im3=(100.0/255)*im+100

im4=255.0*(im/255.0)**2
#print(im4[1,1])
imshow(im3)
x=ginput(3)
print(int(im3.min()),int(im.min()))