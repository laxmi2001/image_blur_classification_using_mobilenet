---
title: Image_bluriness_prediction
emoji: 😻
colorFrom: yellow
colorTo: pink
sdk: streamlit
sdk_version: 1.15.2
app_file: app.py
pinned: false
---
### Image Quality Check
In today’s world, filtering out better-quality images from thousands of images for any kind of further inspection process is a humongous problem. 

### Application of the tool:
In this project, we have used a transfer learning approach with mobilenet neural network and Laplacian variance to identify any lower-quality images. Having a good quality picture is required for the inspector to take any preventive measures against any fire or unexpected activity.

Model name: MobileNet-v2 and Laplacian variance
### Description of the Model:
MobileNet-v2 is a convolutional neural network that is 53 layers deep. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database. It is a class of small, low-latency, low-power models that can be used for classification, detection, and other common tasks convolutional neural networks are good for. Because of their small size, these are considered great deep learning models to be used on mobile devices or run in a quick time.
Laplacian Operator is a second-order derivative operator which is used to find edges in an image. allows a natural link between discrete representations, such as graphs, and continuous representations, such as vector spaces and manifolds. The most important application of the Laplacian is spectral clustering which corresponds to a computationally tractable solution to the graph partitioning problem.

### Project Details:
Here in this project, we have collected some open source images of the houses. The idea is to identify if we are able to see the house clearly or not.
The output of the model simply tells if the houses in the images are clearly visible or not




Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
