import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
import math
from  utils  import to_onehot,from_onehot
from torturer import Torturer

class CelebrateHQ(tf.keras.utils.Sequence):
    def __init__(self, batch_size,img_size, img_dir,mask_dir,dlib_dir,part_count,torturer=True):
        self.index=0
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dlib_dir = dlib_dir
        self.part_count=part_count
        if torturer:
            self.torturer=Torturer(img_size[0],max_scale=10)
        else:
            self.torturer=None
       # else:
       #     self.torturer=None
        self.init_split()
        """
        self.atts_mask = ['skin',  #0
                         'l_brow', #1
                         'r_brow', #2
                         'l_eye',  #3
                         'r_eye',  #4
                         'eye_g',  #5
                         'l_ear',  #6
                         'r_ear',  #7
                         'ear_r',  #8
                         'nose',   #9
                         'mouth',  #10
                         'u_lip',  #11
                         'l_lip',  #12
                         'neck',   #13
                         'neck_l', #14
                         'cloth',  #15
                         'hair',   #16
                         'hat']    #17
        self.atts_dlib=  ['face',          #18
                          'left_eye',      #19 
                          'left_eyebrow',  #20
                          'right_eye',     #21 
                          'right_eyebrow', #22
                          'mouth',         #23
                          'nose']          #24
        """
                         
                                    
        self.attrs = [   [None, None],                #0 empty   
                         ['skin','face'],            #1
                         ['l_brow','left_eyebrow'],  #2
                         ['r_brow','right_eyebrow'], #3
                         ['l_eye','left_eye'],       #4
                         ['r_eye','right_eye'],      #5
                         ['eye_g',None],    #6
                         ['l_ear',None],    #7
                         ['r_ear',None],    #8
                         ['ear_r',None],    #9
                         ['nose','nose'],   #10
                         ["mouth",'mouth'], #11
                         ['u_lip',None],  #12
                         ['l_lip',None],  #13
                         ['neck',None],   #14
                         ['neck_l',None], #15
                         ['cloth',None],  #16
                         ['hair',None],   #17
                         ['hat',None],    #18
                         ]

    def init_split(self):
        self.img_names=sorted([os.path.splitext(fname)[0] for fname in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir,fname))])
        self.part_len=int(math.floor(len(self.img_names) /self.part_count))
        self.dataset_names=[]
        for i in range(self.part_count):
            start_index=i*self.part_len
            end_index=start_index+self.part_len
            self.dataset_names.append(self.img_names[start_index:end_index])
        self.dataset_names_current=self.dataset_names[0]
    def next_part(self,index=None):
        if index is not None:
            self.index=index
        else:
            self.index+=1
        self.part_index=self.index % len(self.dataset_names)
        self.dataset_names_current=self.dataset_names[self.part_index]
    def __len__(self):
        return int(math.floor(len(self.dataset_names_current) / self.batch_size))
    def __getitem__(self, idx):
        img_channel=3
        n_class=len(self.attrs)

        x = np.zeros((self.batch_size,) + self.img_size + (img_channel,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")

        i = idx * self.batch_size
        batch_names = self.dataset_names_current[i : i + self.batch_size]

        for bath_index, name in enumerate(batch_names):
            path=os.path.join(self.img_dir,name+".png")
            source = np.array(load_img(path, target_size=self.img_size,interpolation='bicubic'))
           
            mask=np.zeros(self.img_size)
            for attr_index, attr in enumerate(self.attrs):
                if attr[0] is not None:
                    path=os.path.join(self.mask_dir,name+"_"+attr[0]+".png")
                    if os.path.exists(path):
                        img = np.array(load_img(path, target_size=self.img_size, color_mode="grayscale",interpolation='bicubic'))
                        mask[img>0]=(attr_index)
                if attr[1] is not None:
                    path=os.path.join(self.dlib_dir,name+"_"+attr[1]+".png")
                    if os.path.exists(path):
                        img = np.array(load_img(path, target_size=self.img_size, color_mode="grayscale",interpolation='bicubic'))
                        mask[img>0]=(attr_index)
            
            if self.torturer is not None:
                v_source,v_mask=self.torturer.process(source,mask)
                x[bath_index] = v_source
                y[bath_index,:,:,0] = v_mask
            else:
                x[bath_index] = source
                y[bath_index,:,:,0] = mask
        #normalize et 0..1
        x=x/255
        y=y#-(n_class/2) #  /n_class
        return x, y
    def on_epoch_end(self):
        self.next_part()
        print("dataset epoch end index ->index {} partindex {}".format(self.index,self.part_index))
 

def get_dataset(base_dir,batch_size=1,img_size=(512,512),part_count=2,torturer=True):
    img_dir=os.path.join(base_dir,"img")
    mask_dir=os.path.join(base_dir,"mask")
    dlib_dir=os.path.join(base_dir,"dlib")

    return CelebrateHQ(batch_size,img_size,img_dir,mask_dir,dlib_dir,part_count,torturer=torturer)

