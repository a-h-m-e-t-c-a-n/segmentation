from PIL import Image
from tensorflow import keras as k
from tensorflow.keras.preprocessing.image import load_img
import sys
import os
import numpy as np
import shutil 
from utils import mask_from_sparse
import time

if __name__ == "__main__":
    img_size=(256,256)
    source_dir=sys.argv[1]
    target_dir=sys.argv[2]
    color_map=[[0,0,0],
               [255,0,0],[0,255,0],[0,0,255],
               [64,0,0],[0,64,0],[0,0,64],
               [128,0,0],[0,128,0],[0,0,128],
               [255,0,0],[0,255,0],[0,0,255],
               [255,64,0],[64,255,0],[0,64,255],
               [255,64,64],[64,255,64],[64,64,255]
               ]

    n_class=18
    checkpoint_path="checkpoint/latest_celebreate_mask.h5"

    model = k.models.load_model(checkpoint_path)
    #model.summary()

    fname_list=[f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]
    for index, fname in enumerate(sorted(fname_list)):
        fpath=os.path.join(source_dir,fname)
        
        #subject_rgb = np.array(load_img(fpath, target_size=img_size,interpolation='bicubic',color_mode="grayscale"))
        #subject_rgb=np.stack((subject_rgb,)*3, axis=-1)
        subject_rgb = np.array(load_img(fpath, target_size=img_size,interpolation='bicubic'))

        #rgb_normalization_image = cv2.normalize(subject_rgb, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        subject_rgb_normalized=subject_rgb[np.newaxis,...].astype(np.float)/255
        
        start_time = time.time()
        generated = model(subject_rgb_normalized)
        print("time ms {}\n".format((time.time()-start_time))*1000)

        generated = mask_from_sparse(generated).numpy()
        generated= (generated[0]*255/n_class).astype("uint8")
        
        name,ext=os.path.splitext(fname)
        #path_dir=os.path.join(target_dir,name)
        path_dir=target_dir
        if(os.path.exists(path_dir)==False):
            os.makedirs(path_dir)

        generated=np.concatenate((generated,)*3, axis=-1) #gray scale den rgb ye çevirdik
        output_img=np.concatenate((subject_rgb,generated),axis=1)
        path_file=os.path.join(path_dir,name+".png")
        Image.fromarray(output_img).save(path_file)
        print(fpath+"->"+path_file)



