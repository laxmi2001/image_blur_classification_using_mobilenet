import streamlit as st 
from tensorflow.keras.models import load_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import keras
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adamax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import MobileNetV2
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
import random
import os
from io import BytesIO
import h5py

st.title('Image Bluriness Occulded')

model_file_path = "mobile_net_occ.h5"

##Blurriness Features

plt. figure(figsize=(10,9))
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def threshold(value, thresh):
    if value > thresh:
        return "Not Blur"
    else:
        return "Blur"  
def blurr_predict(img_iter):
  
  def make_prediction(img_content):
    pil_image = Image.open(img_content)
    imgplot = plt.imshow(pil_image)
    #st.image(pil_image)
    plt.show()
    gray_cvimage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    #print(gray_cvimage)
    variance_laplacian = variance_of_laplacian(gray_cvimage)
    #print(variance_laplacian)
    return variance_laplacian

  variance_score = make_prediction(img_iter)
  thresh = 2000
  variance_score = variance_score/thresh
  predicted_label = threshold(variance_score, 1)
  return predicted_label,variance_score

#image_path = "images_11.jpeg"
f = st.file_uploader('Upload an Image',type=(["jpeg","jpg","png"]))
if f is not None:
    image_path = f.name
    st.image(image_path)
else:
    image_path = None
    
predicted_label,variance_score = blurr_predict(image_path)
st.header(predicted_label)
st.header(str(round(variance_score,2)))
#st.("The image is", '\033[1m' + str(predicted_label) + '\033[0m', "with the score value of" +str(round(variance_score,2)))