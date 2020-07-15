
# coding: utf-8

# # Mask R-CNN demo
# 
# This notebook illustrates one possible way of using `maskrcnn_benchmark` for computing predictions on images from an arbitrary URL.
# 
# Let's start with a few standard imports

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import random
import os
import cv2

import sys
sys.path.append("/home/leon/Desktop/mask/maskrcnn-benchmark")


# In[2]:


# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


# Those are the relevant imports for the detection model

# In[3]:


from maskrcnn_benchmark.config import cfg
from mypredictor import COCODemo


# We provide a helper class `COCODemo`, which loads a model from the config file, and performs pre-processing, model prediction and post-processing for us.
# 
# We can configure several model options by overriding the config options.
# In here, we make the model run on the CPU

# In[8]:


config_file = "../configs/my_test_e2e_mask_rcnn_R_101_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


# Now we create the `COCODemo` object. It contains a few extra options for conveniency, such as the confidence threshold for detections to be shown.

# In[30]:


coco_demo = COCODemo(
    cfg,
    min_image_size=720,
    confidence_threshold=0.5,
)


# Let's define a few helper functions for loading images from a URL

# In[31]:


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    #response = requests.get(url)
    #pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    pil_image = Image.open(url).convert("RGB")
    pil_image = pil_image.resize((800,1024))
    # print(pil_image.size)
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# Let's now load an image from the COCO dataset. It's reference is in the comment

# In[36]:


# imgList = os.listdir("./testimg")
# for i in imgList:
#     # image = load("../datasets/coco/val2017/"+no+".jpg")
#     image = load("./testimg/"+i)
#     imshow(image)


#     # ### Computing the predictions
#     # 
#     # We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format), and computes the predictions on them, returning an image with the predictions overlayed on the image.

#     # In[37]:


#     # compute predictions
#     predictions = coco_demo.run_on_opencv_image(image)

#     imshow(predictions)
#     import cv2
#     cv2.imwrite('/home/leon/Desktop/mask/maskrcnn-benchmark/demo/uimgresult/'+i[:-4]+'rst'+'.jpg',predictions)




imgList = os.listdir("./demoImg")

for i in imgList[859:]:

    image = load("./demoImg/"+i)
    # image = load("./testimg/"+i)
    imshow(image)
    # cv2.imwrite('/home/leon/Desktop/mask/maskrcnn-benchmark/demo/demoImg/'+no+'.jpg',image)


    # ### Computing the predictions
    # 
    # We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format), and computes the predictions on them, returning an image with the predictions overlayed on the image.

    # In[37]:


    # compute predictions
    predictions = coco_demo.run_on_opencv_image(image)

    imshow(predictions)
    
    cv2.imwrite('/home/leon/Desktop/mask/maskrcnn-benchmark/demo/udemoresult/'+i[:-4]+'.jpg',predictions)



