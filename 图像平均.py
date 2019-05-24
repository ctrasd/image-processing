from PIL import Image
import os
from imtools import *
from pylab import *

path=os.getcwd()
imlist=get_imlist(path)
print(imlist)
im=compute_average(imlist)
imshow(im)
show()