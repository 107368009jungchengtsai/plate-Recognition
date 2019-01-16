# plate-Recognition
作法說明
## 流程圖
![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)

＃寫出image file 的路徑供yolo-v2使用
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
 
TrainDir = 'data/train_preprocess'
 
out_file = open('data/train_image_path.txt','w')
for root,dirs,files in os.walk(TrainDir):
    for file in files:
        if '.jpg'==file[-4:]:
            out_file.write('%s/%s\n'%(root,file))
out_file.close()

ValDir = 'data/val_preprocess'
 
out_file = open('data/val_image_path.txt','w')
for root,dirs,files in os.walk(ValDir):
    for file in files:
        if '.jpg'==file[-4:]:
            out_file.write('%s/%s\n'%(root,file))
out_file.close()
TestDir = 'data/test'
 
out_file = open('data/test_image_path.txt','w')
i =1
for root,dirs,files in os.walk(TestDir):
    for file in files:
        if '.jpg'==file[-4:]:
            out_file.write('%s/%s\n'%(root,str(i)+'.jpg'))
            i=i+1
out_file.close()
＃xml轉成txt檔
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

import shutil, random

def XMLtoTXT(train_xml_label_Dir):
    for rootDir,dirs,files in os.walk(train_xml_label_Dir+'0'):  
        for file in files:  
            file_name = file.split('.')[0]  
            out_file = open(train_txt_label_Dir+'%s.txt'%(file_name),'w')  
            in_file = open("%s/%s.xml"%('data/train',file_name))  
            tree = ET.parse(in_file)  
            root = tree.getroot()  
            size = root.find('size')  
            w = int(size.find('width').text)  
            h = int(size.find('height').text)  
            
            for obj in root.iter('object'):  
                xmlbox = obj.find('bndbox')  
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))  
                bb = convert((w,h), b)  
                out_file.write("0" + " " + " ".join([str(a) for a in bb]) + '\n')      #only one class,index 0
            out_file.close()
    for rootDir,dirs,files in os.walk(train_xml_label_Dir+'1'):  
        for file in files:  
            file_name = file.split('.')[0]  
            out_file = open(train_txt_label_Dir+'%s.txt'%(file_name),'w')  
            in_file = open("%s/%s.xml"%('data/train',file_name))  
            tree = ET.parse(in_file)  
            root = tree.getroot()  
            size = root.find('size')  
            w = int(size.find('width').text)  
            h = int(size.find('height').text)  
            
            for obj in root.iter('object'):  
                xmlbox = obj.find('bndbox')  
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))  
                bb = convert((w,h), b)  
                out_file.write("0" + " " + " ".join([str(a) for a in bb]) + '\n')      #only one class,index 0
            out_file.close()

xmlDir = 'data/xml/'


def convert(size, box):        #归一化操作
    dw = 1./size[0]  
    dh = 1./size[1]  
    x = (box[0] + box[1])/2.0  
    y = (box[2] + box[3])/2.0  
    w = box[1] - box[0]  
    h = box[3] - box[2]  
    x = x*dw  
    w = w*dw  
    y = y*dh  
    h = h*dh  
    return (x,y,w,h)  

if not os.path.exists(xmlDir):   
    os.makedirs(xmlDir)

if not os.path.exists(train_txt_label_Dir):   
    os.makedirs(train_txt_label_Dir)  
if not os.path.exists(val_txt_label_Dir):   
    os.makedirs(val_txt_label_Dir)  

XMLtoTXT(xml_Dir)

## use yolo-v2
First, use python transforming your xml file into txt files which are used in yolo-v2, and dividing file into train and validation. 
Second, in your file folder use 'git clone https://github.com/pjreddie/darknet' to download file from github, and use 'wget https://pjreddie.com/media/files/yolo.weights' to download weight .
Third, modify the file for your custom train
Forth, run make to make darknet executed file, in terminal execute './darknet detector train <txt file define> <yolo-v2 network> <weight file>',example './darknet detector train cfg/voc.data cfg/yolov2-voc-custom.cfg darknet53.conv.74'

Finally, after training run './darknet detector test cfg/voc.data cfg/yolov2-voc-custom.cfg backup/yolov2-final.weights <image path>'.

And you will see like below.

image yolo_test.png
# 車牌切割
## 流程圖
![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)
## 1.宣告和定義

    import os
    import sys
    import random
    import math
    import re
    import time
    import numpy as np
    import tensorflow as tf
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import skimage
    from skimage import color,io
    import cv2
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from PIL import Image
    import PIL.ImageOps  
## 2.讀取圖像並做顏色反轉
    image = Image.open('image/42.jpg')
    inverted_image = PIL.ImageOps.invert(image)

## 3.圖像做顏色空間轉換
    imgl = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB) 
    #imgl = cv2.cvtColor(imgl, cv2.IMREAD_GRAYSCALE)
    result = cv2.medianBlur(imgl, 3)
    plt.imshow(result);plt.title('Original')
![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/1.PNG)
## 4.圖像做二值化
    gray_image = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 0)
    plt.imshow(img),plt.title('Mask')
![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/2.PNG)
## 5.圖像轉換成灰值
    image_grey = color.rgb2gray(img)
    img = image_grey
![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/4.PNG)
## 6.分割車牌方式
    def split_x(img):
    io.imshow(img)
    ban = []
    last_i = 0
    for i in range(img.shape[1]):
        c = np.count_nonzero(img[:, i] < .3)
        if c > img.shape[0]*.8:
            if i - last_i > 3:
                plt.plot([last_i,last_i], [0,img.shape[0]], c='yellow')
                plt.plot([i,i], [0,img2.shape[0]], c='green')
                ban.append((last_i,i))

            last_i = i
    if i != last_i:
        plt.plot([last_i,last_i], [0,img.shape[0]], c='yellow')
        plt.plot([i,i], [0,img.shape[0]], c='green')
        ban.append((last_i,i))

    chars = []
    for y0, y1 in ban:
        img_sub = img[:, y0:y1]
        chars.append(img_sub)
    return chars

    def split_y(img):
    plt.figure()
    io.imshow(img)
    
    last_i = 0
    ban = []
    for i in range(img.shape[0]):
        c = np.count_nonzero(img[i,:] < .3)
        if c > img.shape[1]*.9:
            if i - last_i > 3:
                plt.plot([0, img.shape[1]],[last_i,last_i], c='yellow')
                plt.plot([0, img.shape[1]],[i,i], c='green')
                ban.append((last_i, i))
            last_i = i
    if i != last_i:
        plt.plot([0, img.shape[1]],[last_i,last_i], c='yellow')
        plt.plot([0, img.shape[1]],[i,i], c='green')
        ban.append((last_i, i))
    
    for r0, r1 in ban:
        if r1 - r0 < img.shape[0]/3:
            continue

        img2 = img[r0:r1,:]
        s_max = max(img2.shape)

        pad_w = int((s_max - img2.shape[0])/2)
        pad_h = int((s_max - img2.shape[1])/2)

        img2 = np.pad(img2, ((pad_w,pad_w), (pad_h,pad_h)), mode='constant', constant_values=np.mean(img))
        img2 = skimage.transform.resize(img2, (20,20), mode='reflect', anti_aliasing=True)
        
        return img2
        img2 = img
  ## 7.X軸切割
        chars = split_x(img2)
  ![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/x.PNG)
  ## 8.Y軸切割
        chars2 = np.array([split_y(c) for c in chars])
  ![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/y.png)
  ## 9.字符model
        DATASET_DIR = 'dataset/carplate'
    classes = os.listdir(DATASET_DIR + "/ann/")

    num_classes = len(classes)
    img_rows, img_cols = 20, 20


    if K.image_data_format() == 'channels_first':
        input_shape = [1, img_rows, img_cols]
    else:
        input_shape = [img_rows, img_cols, 1]
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    model.load_weights("char_cnn.h5")
 ## 10.切割車牌字符辨任
    def extend_channel(data):
    if K.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        
    return data


    ys = np.unique(classes)

    p_test = model.predict_classes(extend_channel(chars2))
    print(' '.join([ys[p_test[i]] for i in range(len(p_test))]))
![image](https://github.com/107368009jungchengtsai/plate-Recognition/blob/master/13.PNG)
    
