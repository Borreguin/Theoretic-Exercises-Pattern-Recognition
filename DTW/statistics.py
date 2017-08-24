import glob
import numpy as np
from scipy import stats
from PIL import Image, ImageDraw, ImageChops
import sys

if len(sys.argv) <= 1:
	directory = './words'
else:
	directory = sys.argv[1]

search = directory + "/*.png"
files = glob.glob(search)
if len(files) == 0:
	print "There are no .png files in the directory"
	sys.exit(0)

collection = [name.split('/') for name in files]
names = [item[-1] for item in collection] 

sizes = []
heights  = []
for file in files:
	image_size = Image.open(file).size
	sizes.append(image_size)
	heights.append(image_size[1])

im_sizes = np.asarray(sizes)
mode, count = stats.mode(im_sizes)
mean_height = mode[0].__getitem__(1)
mean_size = [int(x) for x in np.mean(im_sizes, axis = 0) ]

print "Some statistics:"	 
print "Mean size  Image: ", mean_size 
print "Min size Image: ", np.amin(im_sizes, axis = 0), \
	"\t name:", names[np.argmin(heights)]
print "Max size Image: ", np.amax(im_sizes, axis = 0), \
	"\t name:", names[np.argmax(heights)]
print "Mode size :" , mode[0], \
	"\t name:", names[heights.index(mean_height)]