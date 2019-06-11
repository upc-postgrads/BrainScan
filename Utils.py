#import numpy as np
#
#def cropVolume(Volume):
    #''' This helper function removes all the black area around the 2D images
    #parameter volume: It is a volumetric img, typically a nifti img
    #'''
#
#data = data.get_data()
#count = 0
#sum_array = []
#for depth in range(data.shape[2]):
        #values, indexes = np.where(data[:, :, ch] > 0)
        #sum_val = sum(values)
        #count += 1
        #print(sum_val)
        #sum_array.append(sum_val)
#dep_start = np.nonzero(sum_array)[0][0]
#dep_end = np.nonzero(sum_array)[0][-1]
#print('Slice with non-black pixels goes from nº %s to %s' % (dep_start, dep_end))
#print('Count is:', count)
#
#count = 0
#sum_array = []
#for hei in range(data.shape[1]):
        #values, indexes = np.where(data[:, hei, :] > 0)
        #sum_val = sum(values)
        #count += 1
        #print(sum_val)
        #sum_array.append(sum_val)
#hei_start = np.nonzero(sum_array)[0][0]
#hei_end = np.nonzero(sum_array)[0][-1]
#print('Slice with non-black pixels goes from nº %s to %s' % (hei_start, hei_end))
#print('Count is:', count)
#
#count = 0
#sum_array = []
#for wid in range(data.shape[1]):
        #values, indexes = np.where(data[wid, :, :] > 0)
        #sum_val = sum(values)
        #count += 1
        #print(sum_val)
        #sum_array.append(sum_val)
#wid_start = np.nonzero(sum_array)[0][0]
#wid_end = np.nonzero(sum_array)[0][-1]
#print('Slice with non-black pixels goes from nº %s to %s' % (wid_start, wid_end))
#print('Count is:', count)
#
#return dep_start, dep_end, hei_start, hei_end, wid_start, wid_end

import numpy as np
from PIL import Image
import os

#First define paths
r_path_tr =  '/Users/aitorjara/Desktop/CleanSlices/imagesTr/'
r_path_ts = '/Users/aitorjara/Desktop/CleanSlices/imagesTs/'
r_path_lbl = '/Users/aitorjara/Desktop/CleanSlices/labelsTr/'
w_path_tr =  '/Users/aitorjara/Desktop/CroppedSlices/imagesTr/'
w_path_ts =   '/Users/aitorjara/Desktop/CroppedSlices/imagesTs/'
w_path_lbl = '/Users/aitorjara/Desktop/CroppedSlices/labelsTr/'

class CropStoreImages():

    def biggestImage(mode):

        if mode == 'train':
            r_path = r_path_tr
        elif mode == 'test':
            r_path = r_path_ts
        elif mode == 'label':
            r_path = r_path_lbl

        count = 0
        pix_val = 0
        max_val = 0
        slices = 240
        cropped_data = []

        for filename in os.listdir(r_path):
            f_hand = Image.open(r_path + filename)
            count += 1
            if count % div == 0:
                print("Intensity pixels value is: %s \nVolume nº %.0f" % (pix_val, count/div))
                pix_val = 0
            for x in range(slices):
                for y in range(slices):
                    data = f_hand.load()
                    pix_val += data[x,y]
            if pix_val > max_val:
                max_val = pix_val
        print("The biggest img is: %s\nAll the intensity value pixels add up to %.0f pixels" % (filename, max_val))

        #Biggest img is P420_72.png and has 232658 pixels
        #The biggest img is: P191C4_42.png It has 719344521 pixels in the training set
        #The biggest image is: P665C1_134.png It has 721124568 pixels

    def cropSaveImage(mode):
        #new_wid and new_hei must be even numbers for the code to not blow up

        if mode == 'train':
            new_wid = 240
            new_hei = 240
            r_path = r_path_tr
            w_path = w_path_tr
        elif mode == 'test':
            new_wid = 240
            new_hei = 240
            r_path = r_path_ts
            w_path = w_path_ts
        elif mode == 'label':
            new_wid = 240
            new_hei = 240
            r_path = r_path_lbl
            w_path = w_path_lbl

        f_hand = Image.open(r_path)
        wid, hei = f_hand.size   # Get dimensions

        left = (wid - new_wid)/2
        top = (hei - new_hei)/2
        right = (wid + new_wid)/2
        bottom = (hei + new_hei)/2

        img = f_hand.crop((left, top, right, bottom))
        img.save(w_path)

if __name__ == '__main__':
    #CropStoreImages.biggestImage(mode = 'test')
    CropStoreImages.cropSaveImage(mode = 'test')
