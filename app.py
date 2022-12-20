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


page_names = ["Blurred or Not Blurred Prediction","Occluded or Not Occluded Prediction"]
page = st.sidebar.radio('Navigation',page_names)
st.write("Welcome to the Project")

if page == "Blurred or Not Blurred Prediction":
    st.title("""
         Image Blurriness Occluded
         """)
    st.subheader("Prediction of Blur or NotBlur Image")
    images = ["blur1.png","blurimg2.png","blurimg3.png","images_11.jpeg"]
    with st.sidebar:
        st.write("choose an image")
        st.image(images)
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
else:
    st.title("Prediction of Occluded or not Occluded ")
    plt. figure(figsize=(10,9))
    def occ_predict(img_content):
        im = []
        image=cv2.imread(img_content)
        imgplot = plt.imshow(image)
        plt.show()
        img = Image.fromarray(image, 'RGB') 
        resize_image = img.resize((50, 50))
        im.append(np.array(resize_image))
        fv = np.array(im)
        np_array_img = fv.astype('float32') / 255
        model_gcs = h5py.File(model_file_path, 'r')
        myModel = load_model(model_gcs)
        prediction = myModel.predict(np_array_img)
        score = prediction[0][0].item()
        thresh = 0.5
        if score > thresh:
            return "Not Occluded",score
        else:
            return "Occluded",score

    f = st.file_uploader('Upload an Image',type=(["jpeg","jpg","png"]))
    #st.write(f)
    #st.subheader("Prediction of Occluded or Not Occluded")
    images1 = ["img1.png","img2.png","img3.png","img4.png"]
    with st.sidebar:
        st.write("choose an image")
        st.image(images1)

    if f is None:
        st.write("Please upload an image file")
    else:
        image1= Image.open(f)
        st.image(image1,use_column_width = True)
        predicted_label,variance_score = occ_predict(image1)
        #st.header(predicted_label)
        #st.header(str(round(variance_score,2)))
        string1 = "The image is," + predicted_label + " with the score value of  " + str(round(variance_score,2))
        st.subheader(string1)

#predicted_label, score = occ_predict("/content/drive/MyDrive/Occulded.jpg")
#print("The image is", '\033[1m' + predicted_label1 + '\033[0m', "with the score value of" ,round(score,2))