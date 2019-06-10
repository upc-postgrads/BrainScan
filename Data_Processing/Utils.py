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
            div = 155 * 4
            r_path = r_path_tr
        elif mode == 'test':
            div = 155 * 4
            r_path = r_path_ts
        elif mode == 'label':
            div = 155
            r_path = r_path_lbl

        #sum_array = []
        count = 0
        pix_val = 0
        max_val = 0
        slices = 240

        #Find the img with max added pixels value,
        #do sanity check against a list with all the stored pixel values
        #from the images in the for loop and then return the filename

        for filename in os.listdir(r_path):
            fle_data = Image.open(r_path + filename)
            count += 1
            if count % div == 0:
                print("Intensity pixels value is: %s \nVolume nº %.0f" % (pix_val, count/div))
                pix_val = 0
            for x in range(slices):
                for y in range(slices):
                    data = fle_data.load()
                    pix_val += data[x,y]
            #sum_array.append(pix_val)
            if pix_val > max_val:
                max_val = pix_val
        print("The biggest img is: %s\nAll the intensity value pixels add up to %.0f pixels" % (filename, max_val))

        #for indices, values in enumerate(sum_array):
        #    check_max_value = max(sum_array)
        #    max_index = sum_array.index(check_max_value)
        #print("Length of array is: ", len(sum_array))
        #print('Biggest img is in position %s and has %.0f pixels' % (max_index, check_max_value))
        #Biggest img is in position 15033 and has 232658 pixels in the labels set
        #Biggest img is P420_72.png and has 232658 pixels

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

        img = Image.open(r_path)
        wid, hei = img.size   # Get dimensions

        left = (wid - new_wid)/2
        top = (hei - new_hei)/2
        right = (wid + new_wid)/2
        bottom = (hei + new_hei)/2

        img = img.crop((left, top, right, bottom))
        img.save(w_path)

if __name__ == '__main__':
    #CropStoreImages.biggestImage(mode = 'test')
    CropStoreImages.cropSaveImage(mode = 'test')
