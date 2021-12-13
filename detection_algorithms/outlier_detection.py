import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import shutil
import tensorflow as tf
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_feature_outlier_image
from alibi_detect.utils.saving import save_detector, load_detector
from detection_algorithms.encoder_decoder import *
from report_builders.excel_builder import *
#from controller import study_dir as sd

def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images

sd = "D:\Video\Skyrim\Screens\Training\\**\*.*"

train = img_to_np(sd)
train = train.astype('float32') / 255.

def load_mydetector(path, load=bool):
    if load == 0:
        od = load_detector(path)
    else:
        od = OutlierAE( threshold = 0.001,
                    encoder_net=encoder_net(train),
                    decoder_net=decoder_net(train))

        adam = tf.keras.optimizers.Adam(lr=1e-4)

        od.fit(train, epochs=100, verbose=True,
                optimizer = adam)
        save_detector(od, path)

def find(path_test, path_detector, test_name):
    od = load_detector(path_detector)
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

    if test_name != 0:
        create_report(df, test_name)

    print(df_outliers)

    recon = od.ae(test).numpy()

    plot_feature_outlier_image(preds, test, 
                            X_recon=recon,  
                            max_instances=5,
                            outliers_only=True,
                            figsize=(15,15))