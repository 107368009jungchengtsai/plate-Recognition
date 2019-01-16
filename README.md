# plate-Recognition
作法說明
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
![image]()
## 3.圖像做顏色空間轉換
    imgl = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB) 
    #imgl = cv2.cvtColor(imgl, cv2.IMREAD_GRAYSCALE)
    result = cv2.medianBlur(imgl, 3)
    plt.imshow(result);plt.title('Original')
![image]()
## 4.圖像做二值化
    gray_image = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 0)
    plt.imshow(img),plt.title('Mask')
![image]()
## 5.圖像轉換成灰值
    image_grey = color.rgb2gray(img)
    img = image_grey
![image]()
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
  ![image]()
  ## 8.Y軸切割
        chars2 = np.array([split_y(c) for c in chars])
  ![image]()
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
![image]()
    
