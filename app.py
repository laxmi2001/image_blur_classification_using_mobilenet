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
import io
from io import BytesIO,StringIO
from pathlib import Path
import h5py

model_file_path = "mobile_net_occ.h5"


#page_names = ["Blurred or Not Blurred Prediction","Occluded or Not Occluded Prediction"]
#page = st.sidebar.radio('Navigation',page_names)
#st.write("Welcome to the Project")


st.title("""
         Prediction of Image Blurriness
         """)
#st.subheader("Prediction of Blur or NotBlur Image")
st.write("""Blurring refers to the distortion of the definition of objects in an image, resulting in poor spatial resolution.
Image blur is very common in natural photos, arising from different factors such as object motion, camera lens out-of-focus, and camera shake.
To detect if an image is blurred or not, the variance of Laplacian is used. The Laplacian of an image identifies edges, 
and the variance of the same shows how smooth or hard the edge is. Smooth edges mean blurred images, hence sharp images tend to have
large positive and negative Laplacian. We can use this model for filtering blurred images in all kinds of computer vision projects.
 """)
images = ["blur1.png","bird1.jpeg","blurimg3.png","images_11.jpeg"]
with st.sidebar:
    st.write("choose an image")
    st.image(images)
#model_file_path = "mobile_net_occ.h5"

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
file = st.file_uploader('Upload an Image',type=(["jpeg","jpg","png"]))

if file is None:
    st.write("Please upload an image file")
else:
    image= Image.open(file)
    st.image(image,use_column_width = True)
    predicted_label,variance_score = blurr_predict(file)
        #st.header(predicted_label)
        #st.header(str(round(variance_score,2)))
    string = "The image is," + str(predicted_label) + " with the score value of  " + str(round(variance_score,2))
    st.subheader(string)

st.write("""
For a detailed description please look through our Documentation  
""")

url = 'https://huggingface.co/spaces/ThirdEyeData/image_bluriness_prediction/blob/main/README.md'

st.markdown(f'''
<a href={url}><button style="background-color: #668F45;">Documentation</button></a>
''',
unsafe_allow_html=True)