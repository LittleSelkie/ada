import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from encoder_decoder import *

def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images

path_train = "D:\Video\Skyrim\Screens\Training\\**\*.*" # 100img ~= 3m30s

train = img_to_np(path_train)
train = train.astype('float32') / 255.

od = OutlierAE( threshold = 0.001,
                    encoder_net=coder_net(train).encoder,
                    decoder_net=coder_net(train).decoder)

adam = tf.keras.optimizers.Adam(lr=1e-4)

od.fit(train, epochs=100, verbose=True,
        optimizer = adam)

def find(path_test):
    test = img_to_np(path_test)
    test = test.astype('float32') / 255.

    od.infer_threshold(test, threshold_perc=95)

    preds = od.predict(test, outlier_type='instance',
                return_instance_score=True,
                return_feature_score=True)

    for i, fpath in enumerate(glob.glob(path_test)):
        if(preds['data']['is_outlier'][i] == 1):
            source = fpath
            shutil.copy(source, 'img\\')
            
    filenames = [os.path.basename(x) for x in glob.glob(path_test, recursive=True)]

    dict1 = {'Filename': filenames,
        'instance_score': preds['data']['instance_score'],
        'is_outlier': preds['data']['is_outlier']}
        
    df = pd.DataFrame(dict1)
    df_outliers = df[df['is_outlier'] == 1]

    print(df_outliers)

    recon = od.ae(test).numpy()

    plot_feature_outlier_image(preds, test, 
                            X_recon=recon,  
                            max_instances=5,
                            outliers_only=True,
                            figsize=(15,15))