import numpy
import sys
from PIL import Image, ImageDraw, ImageChops
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
from xml.dom import minidom

doc = minidom.parse(sys.argv[2])  # .svg file parseString also exists


#  define the coordinate to crop
def coordinates_to_crop(coordinate_tuple, xMin, yMin, xMax, yMax):
    x, y = coordinate_tuple
    if(x < xMin):
        xMin = x
    if(x > xMax):
        xMax = x
    if(y < yMin):
        yMin = y
    if(y > yMax):
        yMax = y

    return xMin, yMin, xMax, yMax

def gradient_filter ( im ):
	
	"""
	Takes a grayscale img and retuns the Sobel operator on the image
	@im: a image represented in floats
	"""
	print("Computing energy function using the Sobel operator")
	# A grayscale image represented in floats
	im = im.convert("F")
	im_width, im_height = im.size 
	im_arr = numpy.reshape( im.getdata( ), (im_height, im_width) )
	sobel_arr = generic_gradient_magnitude( im, derivative=sobel )
	gradient = Image.new("L", im.size)
	gradient.putdata( list( sobel_arr.flat ) )

	return gradient


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# Parse all the paths in a svg
for path in doc.getElementsByTagName('path'):
    path_strings = path.getAttribute('d')  # The svg path as a string
    img_id = path.getAttribute('id')  # Image id, for file name
    # Maybe not the best method for string parsing but will always work
    # for this kind of paths
    path_strings = path_strings.replace(" Z ", ";")
    path_strings = path_strings.replace(" L ", ";")
    path_strings = path_strings.replace(" M ", ";")
    path_strings = path_strings.replace("Z ", "")
    path_strings = path_strings.replace("L ", "")
    path_strings = path_strings.replace("M ", "")
    path_strings = path_strings.replace(" Z", "")
    path_strings = path_strings.replace(" L", "")
    path_strings = path_strings.replace(" M", "")

    # Split the path string into tuple string
    coordinate_tuples_string = path_strings.split(';')

    # polygon will contain the coordinates
    polygon = []
    xMax = yMax = 0
    xMin = yMin = float('Inf')

    for coordinate_string in coordinate_tuples_string:
        coordinate_list = coordinate_string.split(" ")
        coordinate_tuple = (float(coordinate_list[0]), float(coordinate_list[1]))
        xMin, yMin, xMax, yMax = coordinates_to_crop(coordinate_tuple, xMin, yMin, xMax, yMax)
        polygon.append(coordinate_tuple)

    # read image as RGB and add alpha (transparency)
    im = Image.open(sys.argv[1]).convert("RGBA")  # .jpg file

    # convert to numpy (for convenience)
    imArray = numpy.asarray(im)

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape, dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :, :3] = imArray[:, :, :3]

    # transparency (4th column)
    newImArray[:, :, 3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")

    # alternative way of binarization using the Sobel operator
    # as a filter (this approach takes in count the gradient of the image)
    
    binaryIm = gradient_filter(newIm)   
    binaryIm = binaryIm.point(lambda x: 0 if x > 160 else 255, '1')

    # paste to new image
    jpgSize = (int(round(newIm.size[0])), int(round(newIm.size[1])))
    background = Image.new('L', jpgSize, 255)
    background.paste(binaryIm, mask=newIm.split()[3])  # 3 is the alpha channel
    background = background.convert('1')

    # crop the imagen at size of the word
    cropIntegers = [int(round(x)) for x in (xMin, yMin, xMax, yMax)]
    background = background.crop(cropIntegers)

    # trim
    background = trim(background)

    # resize (respect the original size of the image therefore the next line is commented)
    #background = background.resize((100, 100), Image.ANTIALIAS)

    # jpg version
    #file_name = "words2/" + img_id + ".jpg"
    #background.save(file_name)

    # png version
    file_name = "words/" + img_id + ".png"
    background.save(file_name)

    print("Created : "+file_name)

doc.unlink()
