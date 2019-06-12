import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import imageio

def testCropSave(mode):

    '''PATHS'''
    r_path_train = '/Users/aitorjara/Desktop/CleanSlices/imagesTr'
    r_path_test = '/Users/aitorjara/Desktop/CleanSlices/imagesTs'
    r_path_label = '/Users/aitorjara/Desktop/CleanSlices/labelsTr'
    w_path_train = '/Users/aitorjara/Desktop/CroppedSlices/imagesTr'
    w_path_test = '/Users/aitorjara/Desktop/CroppedSlices/imagesTs'
    w_path_label = '/Users/aitorjara/Desktop/CroppedSlices/labelsTr'

    if mode = 'train':
        r_path = r_path_train
    elif mode = 'test':
        r_path = r_path_test
    elif mode = 'label':
        r_path = r_path_label

    max_hei = 0
    max_wid = 0
    cropped_images = []
    for filename in os.listdir(r_path):
        data = img.imread(r_path + filename)
        mask = data > 0             # Mask of non-black pixels (assuming image has a single channel).
        coords = np.argwhere(mask)  # Coordinates of non-black pixels.
        try:                        # Bounding box of non-black pixels.
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        except:
            pass

        cropped = data[x0:x1, y0:y1]          # Get the contents of the bounding box.
        cropped_images.append(cropped)        # Save the cropped images in a list as data
        if cropped.shape[0] > max_hei:
            max_hei = cropped.shape[0]
        if cropped.shape[1] > max_hei:
            max_wid = cropped.shape[1]

#Insert padding to make the most cropped images be the same size as the bigget image
    black = [0,0,0]
    for image in cropped_images:
        constant=cv2.copyMakeBorder(image, (max_hei - image.shape[0])/2, (max_wid - image.shape[0])/2, (max_wid - image.shape[0])/2, (max_wid - image.shape[0])/2, cv2.BORDER_CONSTANT,value=black)
        imageio.imwrite('/Users/aitorjara/Desktop/P485C1_100,1.png', constant) #convert from data to image and save
