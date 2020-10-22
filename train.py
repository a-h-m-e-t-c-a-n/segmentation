import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import backend as kbe
from tensorflow.keras import layers as kl
from PIL import Image
from monitor import Checkpoint,PrintSample
from unet import define_unet
from xception import define_unet_xception
from tensorflow.keras.utils import plot_model
import json 
import os
from utils import mask_from_sparse,build_accuracy_for_sparse

# Free up RAM in case the model definition cells were run multiple times
k.backend.clear_session()

img_size=(256,256)
checkpoint_path="checkpoint/celebreate_mask.h5"
state_path="checkpoint/state_celebreate_mask.h5.json"

model=define_unet_xception((256,256),19)
#model=define_unet((256,256,3),19)
model.summary()
#plot_model(model,to_file="unet_xception.png", show_shapes=True,show_layer_names=True)

from  datagenerator import get_dataset
train=get_dataset("/media/ahmet/Workspace/project/ai/data/dataset/celeb",batch_size=2,part_count=2,img_size=img_size)
validation=get_dataset("/media/ahmet/Workspace/project/ai/data/dataset/celeb",batch_size=2,part_count=2,img_size=img_size)
validation.next_part()
printdata=get_dataset("/media/ahmet/Workspace/project/ai/data/dataset/celeb",img_size=img_size,batch_size=1,part_count=1,torturer=False)


model.compile(run_eagerly=False,optimizer="adam",loss=k.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[build_accuracy_for_sparse()])

callbacks = [
     k.callbacks.TensorBoard(log_dir="./logs",update_freq='batch'),
     Checkpoint(checkpoint_path,batch_save_freq=1000,save_loop=False)
    ,PrintSample([("print",printdata)],print_freq=500,one_hot=False,print_dir="samples")

]

state={"epoch":0,"batch":0}
if(os.path.exists(state_path)):
    with open(state_path) as file: 
        state= json.loads(file.read()) 

model.fit(train,steps_per_epoch=len(train), epochs=100,verbose=1,validation_data=validation,validation_steps=500,max_queue_size=10,callbacks=callbacks,initial_epoch=state["epoch"])



