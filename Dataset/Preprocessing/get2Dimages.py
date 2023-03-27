import skimage
print(skimage.__version__)
from bs4 import BeautifulSoup
import numpy as np
import pydicom as dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True) 
import pydicom.uid
import pandas as pd
import cv2

###########################################
###########################################
# read from xml file
with open('195.xml', 'r') as f: # please change the file you want to read
    data = f.read()

Bs_data = BeautifulSoup(data, "xml")

# Find all coordinate for Z position
b_roi = Bs_data.find_all('roi')
bX = []
bY = []
Zposition =[]
for edge in b_roi:
  cood_name = edge.find_all('edgeMap')
  Zposition.append(float(edge.find('imageZposition').string[1:]))
  x = []
  y = []
  for c in cood_name:
    x.append(int(c.find('xCoord').string))
    y.append(int(c.find('yCoord').string))
  bX.append(x)
  bY.append(y)

  # Tidy up Z position
Zposition_without = list( dict.fromkeys(Zposition) )

# get all the slice name
path = '/content/new'
slice_name = []
for s in os.listdir(path):
  slices = dicom.read_file(path + '/' + s,force=True)
  if -slices.SliceLocation in Zposition_without:
    slice_name.append(s)

# delete the slice and z position where SliceLoation != last element of ImagePositionPatient
for p in slice_name:
  scans = [dicom.read_file('/content/new/'+p,force=True)]
  if scans[0].SliceLocation!= scans[0].ImagePositionPatient[-1]:
    Zposition_without.remove(-scans[0].SliceLocation)
    slice_name.remove(p)

print(len(slice_name))
print(Zposition_without)

###########################################
###########################################
# Load images to numpy

def get_pixels_hu(path):
    scans = [dicom.read_file('/content/new/'+path,force=True)]
    scans[0].PixelSpacing = [0.5,0.5]
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)

      
    return np.array(image, dtype=np.int16)


id=0
imgs = []
for p in slice_name:
  imgs.append(get_pixels_hu(p))

###########################################
###########################################
# Crop images based on z position
# get abnormal center

# read from csv file
df = pd.read_csv('/content/list3.2.csv')

zX = int(df[df['case']==221]['x loc.'].tolist()[0])
zY = int(df[df['case']==221]['y loc.'].tolist()[0])


for t in range(len(Zposition_without)):
  # get all the annotations id for position 0
  ann_id = []
  for idx,z in enumerate(Zposition):
    if z == Zposition_without[t]:
      ann_id.append(idx)

  # get location from the nodule list csv

  crop_img = np.zeros((90,90))
  
  for i in range(90):
    for j in range(90):
      crop_img[i][j]=imgs[t][0][i+zY-45][j+zX-45]

  #im = Image.fromarray(crop_img)
  plt.imsave('/content/new/result/z-'+str(Zposition_without[t])+'.png',crop_img)
  
  #cv2.imwrite('/content/new/result/z-'+str(Zposition_without[t])+'.png',crop_img)
  image = cv2.imread('/content/new/result/z-'+str(Zposition_without[t])+'.png')
  image = cv2.resize(image,(180,180))
  cv2.imwrite('/content/new/result/z-'+str(Zposition_without[t])+'.png',image)
  

  # generate masks
  # mask outlines
  for ann in range(len(ann_id)):
    mask_img = np.zeros((90,90))
    for i in range(len(bX[ann_id[ann]])):
      mask_img[bY[ann_id[ann]][i]-zY+45][bX[ann_id[ann]][i]-zX+45]=255

    plt.imsave('/content/new/result/gt-'+str(Zposition_without[t])+str(ann)+'.png',mask_img)
    mask = cv2.imread('/content/new/result/gt-'+str(Zposition_without[t])+str(ann)+'.png')
    mask = cv2.resize(mask,(180,180))
    cv2.imwrite('/content/new/result/gt-'+str(Zposition_without[t])+str(ann)+'.png',mask)
  
  # add black mask for non annotations
  for ann in range(4-len(ann_id)):
    mask_img = np.zeros((90,90))
    plt.imsave('/content/new/result/gt-'+str(Zposition_without[t])+str(ann + len(ann_id))+'.png',mask_img)
    mask = cv2.imread('/content/new/result/gt-'+str(Zposition_without[t])+str(ann + len(ann_id))+'.png')
    mask = cv2.resize(mask,(180,180))
    cv2.imwrite('/content/new/result/gt-'+str(Zposition_without[t])+str(ann + len(ann_id))+'.png',mask)


 
