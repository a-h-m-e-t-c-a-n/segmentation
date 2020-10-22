import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from utils import from_onehot,mask_from_sparse
from PIL import Image
import os
import numpy as np
import time
import json 



class PrintSample(keras.callbacks.Callback):
    def __init__(self,datasets=None,print_freq=-1,print_dir=None,one_hot=False):
        super().__init__()
        self.datasets=datasets
        self.print_freq=print_freq
        self.print_dir=print_dir
        self.one_hot=one_hot
        self._epoch=0
        self.metric=tf.keras.metrics.Accuracy()


    def print_images(self,batch,logs=None):
        if(self.print_dir is  None):
            return
        for dataset in self.datasets:
            index=int(np.random.uniform(low=0,high=len(dataset[1])-1))
            source,target=dataset[1][index]

            generated = self.model(source)
            dir_p="{}/{}".format(self.print_dir,dataset[0])

            if(os.path.exists(dir_p)==False):
                os.makedirs(dir_p)

            
            source_rgb = (np.array(source[0])*255).astype("uint8")
            target_rgb = (np.concatenate((target[0],)*3, axis=-1)*255/18).astype("uint8")
            generated = mask_from_sparse(generated)
            generated_rgb = (np.concatenate((generated[0],)*3, axis=-1)*255/18).astype("uint8")
            
            self.metric.update_state(target[0],generated[0])
            accuracy=self.metric.result().numpy()

            """
            if(generated_image.shape[3]==1):
                #imgarr=np.squeeze((np.array(generated_image[i]+1)*128).astype("uint8"),-1)
                imgarr=np.squeeze((np.array(generated_image[0])*255).astype("uint8"),-1)
            else:
                imgarr=(np.array(generated_image[0])*255).astype("uint8")
            """

            output_img=np.concatenate((source_rgb,target_rgb,generated_rgb),axis=1)
            if logs is None:
                loss=0
            else:
                loss="{:.4f}".format(logs.get("loss")).replace(".","_")
            loss=str(loss).replace(".","_")
            accuracy="{:.4f}".format(accuracy).replace(".","_")
            path="{}/{}_{}_{}_{}.jpg".format(dir_p,self._epoch,batch,loss,accuracy)
            Image.fromarray(output_img).save(path,"JPEG")
    
    def print_onehot_images(self,batch,logs={}):
        if(self.print_dir is  None):
            return
        datasets=[]

        if self.train_data is not None:
            datasets.append(("train",self.train_data))
        if self.val_data is not None:
            datasets.append(("val",self.val_data))
        for dataset in datasets:
            index=int(np.random.uniform(low=0,high=len(dataset[1])-11))
            source,target=dataset[1][index]

            generated = self.model(source)
            dir_p="{}/e_{}_b_{}/{}".format(self.print_dir,self._epoch,batch,dataset[0])

            if(os.path.exists(dir_p)==False):
                os.makedirs(dir_p)

            
            source = (np.array(source[0])*255).astype("uint8")
            target = (np.array(target[0])*255).astype("uint8")
            generated= (np.array(generated[0])*255).astype("uint8")
            
            #path="{}/source.jpg".format(dir_p)
            #Image.fromarray(source).save(path,"JPEG")
            for channel_i in range(generated.shape[2]):
                target_img=target[:,:,channel_i]
                generated_img=generated[:,:,channel_i]
                #output_img=np.concatenate((target_img,generated_img),axis=1)
                
                target_img=np.stack((target_img,)*3, axis=-1) #gray scale den rgb ye çevirdik
                generated_img=np.stack((generated_img,)*3, axis=-1)
                output_img=np.concatenate((source,target_img,generated_img),axis=1)
                loss = logs.get("loss") 
                if logs is None:
                    loss=0
                else:
                    loss="{:.4f}".format(logs.get("loss")).replace(".","_")
                loss=str(loss).replace(".","_")
                path="{}/b_{}_c_{}_{}.jpg".format(dir_p,batch,channel_i,loss)
                Image.fromarray(output_img).save(path,"JPEG")
    def on_train_batch_end(self,batch, logs=None):
        super().on_train_batch_end(batch,logs)
        if(self.print_freq>=0):
            if((batch%self.print_freq)==0):
                if(self.one_hot):
                    self.print_onehot_images(batch,logs)
                else:
                    self.print_images(batch,logs)
    def on_epoch_begin(self, epoch, logs={}):
        self._epoch=epoch


class Checkpoint(keras.callbacks.Callback):
    def __init__(self,weigth_file_path,metric='loss',batch_save_freq=-1,load_weights_on_restart=True,save_weights_only=False,save_best=True,load_best=False,save_loop=True):
        super().__init__()
        self.weigth_file_path=weigth_file_path
        self.batch_save_freq=batch_save_freq
        self.save_best=save_best
        self.save_weights_only=save_weights_only
        self.metric=metric
        self.load_best=load_best
        self.load_weights_on_restart=load_weights_on_restart
        self.current=None
        self.best=None
        self.save_loop=save_loop
        self._epoch=0
    def on_epoch_begin(self,epoch, logs=None):
        self._epoch=epoch
        super().on_epoch_begin(epoch,logs=logs)
    def update_filepath(self,batch):
        if self.current is None:
            current_str="0"
        else:
            current_str="{:.4f}".format(self.current).replace(".","_")

        if self.best is None:
            best_str="0"
        else:
            best_str="{:.4f}".format(self.best).replace(".","_")

        dir_path=os.path.dirname(self.weigth_file_path)
        file_name=os.path.basename(self.weigth_file_path)
        if(dir_path!='' and os.path.exists(dir_path)==False):
            os.makedirs(dir_path)
        if dir_path!='':
            self.file_path="{}/epoch_{}_batch_{}_{}_{}_{}".format(dir_path,self._epoch,batch,current_str,best_str,file_name)
            self.latest_file_path="{}/latest_{}".format(dir_path,file_name)
            self.best_file_path="{}/best_{}".format(dir_path,file_name)
            self.state_data="{}/state_{}.json".format(dir_path,file_name)
            self.state_data_epoch="{}/epoch_{}_state_{}.json".format(dir_path,self._epoch,file_name)
        else:
            self.file_path="epoch_{}_batch_{}_{}_{}_{}".format(self._epoch,batch,current_str,best_str,file_name)
            self.latest_file_path="latest_{}".format(file_name)
            self.best_file_path="best_{}".format(file_name)
            self.state_data="state_{}.json".format(file_name)
            self.state_data_epoch="epoch_{}_state_{}.json".format(self._epoch,file_name)
    def on_epoch_end(self,epoch, logs=None):
        self._save(0,logs)
        
        with open(self.state_data_epoch,"w", encoding="utf8") as file: 
            json.dump({"epoch":self._epoch,"batch":None,"best":self.best,"logs":logs},file)  
        super().on_epoch_begin(epoch,logs=logs)
    def update_best(self,logs):
        logs = logs or {}
        if self.save_best:
            self.current = logs.get(self.metric)
            if self.current is not None:
                if self.best is None:
                    self.best=self.current
                    self.best_changed=True
                else:
                    if self.best>self.current:
                        self.best=self.current
                        self.best_changed=True
    def _save(self,batch,logs):
        self.update_best(logs)        
        self.update_filepath(batch)
        if self.save_weights_only:
            if self.save_loop:
                self.model.save_weights(self.file_path,overwrite=True)
            self.model.save_weights(self.latest_file_path,overwrite=True)
        else:
            if self.save_loop:
                self.model.save(self.file_path,overwrite=True)
            self.model.save(self.latest_file_path,overwrite=True)
        if self.save_best:
            if self.best_changed:
                if self.save_weights_only:
                    self.model.save_weights(self.best_file_path,overwrite=True)
                else:
                    self.model.save(self.best_file_path,overwrite=True)
                self.best_changed=False

    def on_train_batch_end(self,batch, logs=None):
        if self.batch_save_freq>0:
            if((batch%self.batch_save_freq)==0):
                self._save(batch,logs)
                with open(self.state_data,"w", encoding="utf8") as file: 
                    json.dump({"epoch":self._epoch,"batch":batch,"best":self.best,"logs":logs},file)  
    def on_train_begin(self, logs=None):
        self.update_filepath(0)
        if self.load_weights_on_restart:
            if self.load_best:
                if os.path.exists(self.best_file_path):
                    self.model.load_weights(self.best_file_path)
                elif os.path.exists(self.latest_file_path):
                    self.model.load_weights(self.latest_file_path)
            else:
                if os.path.exists(self.latest_file_path):
                    self.model.load_weights(self.latest_file_path)
        
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_start_time=0
        self.batch_start_time=0
    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
    def on_batch_end(self, batch, logs=None,):
        batch_diff=time.time() - self.batch_start_time
        print(" bt:{}     ".format(int(batch_diff*1000)),end='\r')
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        epoch_time=time.time() - self.epoch_start_time
        print(" epoch min:",int(epoch_time/60))