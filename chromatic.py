import numpy
from PIL import Image, ImageChops
import sys, math

#print("seperating channels")
def chromatic(image, offset):
    try:
        with Image.open(image) as im:

            r, g, b = im.split()
    except IOError as e:
        print()

    #print("offsetting channels")

    offset = int(offset)
    #print(offset)

    g = ImageChops.offset(g, offset, 0)
    b = ImageChops.offset(b, int(math.cos(2 * math.pi / 3) * offset),
                          int(math.sin(2 * math.pi / 3) * offset))
    r = ImageChops.offset(r, int(math.cos(4 * math.pi / 3) * offset),
                          int(math.sin(4 * math.pi / 3) * offset))

    print("merging channels")

    result = Image.merge("RGB", [r, g, b])
    open_cv_image = numpy.array(result)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
    #result.show()

    #filename = sys.argv[1].split(".")[0]

    #print("saving file: " + filename + "_final.png")

    #result.save("final.png")