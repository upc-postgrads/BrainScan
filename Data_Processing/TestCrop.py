from PIL import Image
from numpy import asarray
import matplotlib.image as img
import matplotlib.pyplot as plt

data = img.imread('/Users/aitorjara/Desktop/CleanSlices/imagesTs/P665C1_100.png')

# Mask of non-black pixels (assuming image has a single channel).
mask = data > 0

# Coordinates of non-black pixels.
coords = np.argwhere(mask)

# Bounding box of non-black pixels.
try:
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
except:
    pass

# Get the contents of the bounding box.
cropped = data[x0:x1, y0:y1]
