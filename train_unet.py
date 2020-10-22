import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import backend as kbe
from tensorflow.keras import layers as kl
from PIL import Image
from monitor import TBMonitor,BestCheckPoint,PrintMonitor
from unet import define_unet


# Free up RAM in case the model definition cells were run multiple times
k.backend.clear_session()

img_size=(512,512)
# Build model
model = define_unet(img_size,n_classes=25)
model.summary()

from  datagenerator import get_dataset
tds,vds=get_dataset("/media/ahmet/Workspace/project/ai/data/dataset/celeb",1,img_size=img_size)


model.compile(optimizer="adam",loss='categorical_crossentropy', metrics=['categorical_accuracy'])

import os
checkpoint_dir="checkpoint"
weight_file_name="celebreate_mask.h5"
latest_path=os.path.join(checkpoint_dir,"latest_"+weight_file_name)
cp_path=os.path.join(checkpoint_dir,weight_file_name)
if(os.path.exists(latest_path)):
    model.load_weights(latest_path)

callbacks = [
     BestCheckPoint(cp_path, save_best_only=True,batch_save_freq=1000)
    ,TBMonitor(tds,vds,-1)
    ,PrintMonitor(tds,vds,print_freq=500,one_hot=True,print_dir="samples")

]

model.fit(tds,steps_per_epoch=len(tds), epochs=100,verbose=1,validation_data=vds,validation_steps=len(vds),max_queue_size=10,callbacks=callbacks)


