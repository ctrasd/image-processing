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

im1=array(Image.open('waiji/images/001.jpg'))
im2=array(Image.open('waiji/images/002.jpg'))

points2D=[loadtxt('waiji/2D/00'+str(i+1)+'.corners').T for i in range(3)]

points3D=loadtxt('waiji/3D/p3d').T

corr = genfromtxt('waiji/2D/nview-corners',dtype='int')

P=[camera.Camera(loadtxt('waiji/2D/00'+str(i+1)+'.P')) for i in range(3)]

