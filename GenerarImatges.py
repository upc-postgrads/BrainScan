import nibabel as nib
import os
from PIL import Image
import numpy as np
import pylab
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join
import traceback
import sys
import pandas as pd


imagePathTR = "/Users/aitorjara/Desktop/Task01_BrainTumour/imagesTr/"
labelPathTR = "/Users/aitorjara/Desktop/Task01_BrainTumour/labelsTr/"
imagePathTS = "/Users/aitorjara/Desktop/Task01_BrainTumour/imagesTs/"
outputDir = "/Users/aitorjara/Desktop/CleanSlices/Slices_David"


outputFileType = "jpg" # "jpg", "png"
labelsIndex = [0,1,2,3]
framesIndex = [0,1,2,3]
sliceStart =0
sliceEnd = 154
numberOfImagesToProcess=2
createFolderByPatient=True

lst = []
df = pd.DataFrame(columns=['id','type', 'frame','index','valid'])

def addResultLine(id,type,frame,index,valid):
    lst.append(df.append(pd.Series([id, type,frame, index,valid], index=df.columns), ignore_index=True))

def ensure_dir_from_file_path(file_path):
    directory = os.path.dirname(file_path)
    ensure_dir(directory)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getPatientID(fileName):
    id = fileName.split('.')[0]
    id = id[-3:]
    id = str(int(id))
    return id

def get_output_file_name(inputFile, index, frame, origin):
    str_fileName = inputFile.split('.')[0]
    if createFolderByPatient==True:
        str_fileName= os.path.join(getPatientID(inputFile),str_fileName)
    #print(str_fileName)
    str_outputFile = '%s-%s%03d-slice%03d.%s' % (
                                                str_fileName,
                                                origin,
                                                frame, index,
                                                outputFileType)

    return str_outputFile

# The standard work-around: first convert to greyscale
def isValidImage(img):
    value = True
    extrema = img.getextrema()
    if extrema == (0, 0):
        # all black
        value = False
    elif extrema == (255, 255):
        value = False

    return value

"""
def SaveImage(data,outputFile,FL):
    if createFolderByPatient==True:
        ensure_dir_from_file_path(outputFile)
    if outputFileType == "jpg":
        if FL=="F":
            im=Image.fromarray(np.squeeze(255.*data/np.max(np.abs(data))).astype(np.uint8))
        if FL=="L":
            im = Image.fromarray(np.uint8(data)*255)
        if isValidImage(im):
            im.save(outputFile)
    if outputFileType == "png":
        pylab.imsave(outputFile, data, format="png", cmap = cm.Greys_r)
"""

def SaveImage(data,outputFile,label):
    if createFolderByPatient==True:
        ensure_dir_from_file_path(outputFile)
    data = ((np.subtract(data[:,:],data[:,:].min(), dtype=np.float32) / np.subtract(data[:,:].max(),data[:,:].min(), dtype=np.float32))*255.9).astype(np.uint8)
    res=False
    im=Image.fromarray(data)
    if isValidImage(im):
        res=True
        im.save(outputFile)
    return res


def GenerateImagesFromVolume(inputFile,outputFolder,label):
    #try:
        nii_img = nib.load(inputFile)
        data = nii_img.get_data()
        for f in framesIndex:
            data_f = data[:,:,:,f]
            for i in range(sliceStart, sliceEnd):
                data_i = data_f[:, :, i]
                outputFile=get_output_file_name(os.path.basename(inputFile),i,f,label)
                res=SaveImage(data_i,os.path.join(outputFolder,outputFile),label)
                addResultLine(getPatientID(os.path.basename(inputFile)),label,f,i,res)

    #except Exception as e:
    #    print('\nError al processar el fitxer %s\n%s\n' % (inputFile, str(e.with_traceback)))


def GenerateImagesFromLabel(inputFile,outputFolder,label):
    try:
        nii_img = nib.load(inputFile)
        data = nii_img.get_data()
        for l in labelsIndex:
            for i in range(sliceStart, sliceEnd):
                data_i = data[:, :, i]
                data_i = data_i == l
                outputFile=get_output_file_name(os.path.basename(inputFile),i,l,label)
                res=SaveImage(data_i,os.path.join(outputFolder,outputFile),label)
                addResultLine(getPatientID(os.path.basename(inputFile)),label,f,i,res)
    except Exception as e:
        print('\nError al processar el fitxer %s\n%s\n' % (inputFile, str(e)))



def GenerateImages():

    outputImagePathTR=os.path.join(outputDir,"imagesTr")
    outputlabelPathTR=os.path.join(outputDir,"labelsTr")
    outputimagePathTS=os.path.join(outputDir,"imagesTs")

    ensure_dir(outputImagePathTR)
    ensure_dir(outputlabelPathTR)
    ensure_dir(outputimagePathTS)

    print("Generating files training")
    onlyfiles = [f for f in listdir(imagePathTR) if isfile(join(imagePathTR, f)) and f.endswith("nii.gz")]
    for i,f in enumerate(onlyfiles):
        if i<numberOfImagesToProcess:
            GenerateImagesFromVolume(os.path.join(imagePathTR,f),outputImagePathTR,'train')
    print("Done generating training")

    print("Generating files label")
    onlyfiles = [f for f in listdir(labelPathTR) if isfile(join(labelPathTR, f)) and f.endswith("nii.gz")]
    for i,f in enumerate(onlyfiles):
        if i<numberOfImagesToProcess:
            GenerateImagesFromLabel(os.path.join(labelPathTR,f),outputlabelPathTR,'label')
    print("Done generating label")

    print("Generating files test")
    onlyfiles = [f for f in listdir(imagePathTS) if isfile(join(imagePathTS, f)) and f.endswith("nii.gz")]
    for i,f in enumerate(onlyfiles):
        if i<numberOfImagesToProcess:
            GenerateImagesFromVolume(os.path.join(imagePathTS,f),outputimagePathTS,'test')
    print("Done generating test")


    resultData=df.append(lst , ignore_index=True)
    print(resultData)
GenerateImages()
