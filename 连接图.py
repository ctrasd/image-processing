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

g=pydot.Dot(graph_type='graph',fontname='FangSong')

g.add_node(pydot.Node(str(0),fontname='FangSong'))
'''
for i in range(5):
	g.add_node(pydot.Node(str(i+1)))
	g.add_edge(pydot.Edge(str(0),str(i+1)))
	for j in range(5):
		g.add_node(pydot.Node(str(i+1)+'-'+str(j+1)))
		g.add_edge(pydot.Edge(str(i+1),str(i+1)+'-'+str(j+1)))
'''
g.add_node(pydot.Node(str("2"),fontname='FangSong'))

g.write_png('g.png',prog='neato')

