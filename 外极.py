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

exec(open("./load_vggdata.py").read())

X=vstack((points3D,ones(points3D.shape[1])))
x=P[0].project(X)

figure()
imshow(im1)
plot(points2D[0][0],points2D[0][1],'*')

figure()
imshow(im1)
plot(x[0],x[1],'r.')

show()
