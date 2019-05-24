from PIL import Image
import os
import numpy as np
from numpy import random
from pylab import *
from imtools import *
from scipy.ndimage import filters
im=zeros((500,500))
im[100:400,100:400]=128
im[200:300,200:300]=255

im=im+30*np.random.standard_normal((500,500))

U,T=denoise(im,im)
G=filters.gaussian_filter(im,10)

imshow(im)
show()

imshow(U)
show()

imshow(T)
show()