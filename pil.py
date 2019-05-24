from PIL import Image
import os
from imtools import get_imlist 
pil_im = Image.open('1.jpg')
#pil_im.show()
path=os.getcwd()
a=get_imlist(path)
print(a)
filelist=os.listdir()
'''
for infile in filelist:
	outfile=os.path.splitext(infile)[0]+".jpg"
	if infile != outfile:
		try:
			Image.open(infile).save(outfile)
		except IOError:
			print("Cannot convert",infile)
'''

#pil_im.thumbnail((128,120))
pil_im.show()
box=(300,100,600,400)
region=pil_im.crop(box)
region=region.transpose(Image.ROTATE_90)
pil_im.paste(region,box)
pil_im.show()