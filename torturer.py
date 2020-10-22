import cv2
import random
import numpy as np
import math

class Torturer():
    def __init__(self,dim,max_scale=10):
        self.dim=dim
        self.max_scale=max_scale
    def random_bool(self):
        return bool(random.getrandbits(1))
    def random_dim(self):
        return random.randint(int(self.dim/self.max_scale), self.dim-1)
    def random_divisible_dim(self):
        rval= random.uniform(self.dim/20,self.dim/10)
        for i in range(int(rval),0,-1):
            if self.dim%i==0:
                return i
    def random_pos(self,margin):
        x= random.randint(0,self.dim-margin)
        y= random.randint(0,self.dim-margin)
        return x,y
    def mirror(self,x,y):
        new_x=np.fliplr(x)
        new_y=np.fliplr(y)
        return new_x,new_y
    
    def random_interpolation(self):
       interpolations=[cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]    
       return interpolations[random.randint(0,len(interpolations)-1)]


    def bad_resulation(self,x,y):
        new_dim= random.randint(int(self.dim/self.max_scale), int(self.dim/self.max_scale*2))
        sx=cv2.resize(x,(new_dim,new_dim),interpolation=self.random_interpolation())
        sx=cv2.resize(sx,(self.dim,self.dim),interpolation=self.random_interpolation())

        return sx,y
    
    def color_noise(self,x,y):
        rgb_factor = np.random.uniform(low=0.3, high=2.5, size=3)
        n_x=(x*rgb_factor).astype("uint8")
        return n_x,y
    
    def hsv_noise(self,x,y):
       hsvImg = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
       #factor =np.random.uniform(low=0.5, high=2, size=3)
       factor=random.uniform(0.5, 2)
       vValue=hsvImg[:,:,2].astype("float32")*factor
       vValue[vValue>255]=255
       hsvImg[:,:,2]=vValue.astype("uint8")
       hsvImg=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
       return hsvImg,y
    """       
    def hsv_noise(self,x,y):
       hsvImg = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
       factor =np.random.uniform(low=0.5, high=2, size=3)
       #vValue=hsvImg[:,:,2].astype("float32")*0.5
       vValue=hsvImg.astype("float32")*factor
       
       vValue[vValue>255]=255

       #hsvImg[:,:,2]=vValue.astype("uint8")
       hsvImg=vValue.astype("uint8")
      
       hsvImg=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
       return hsvImg,y
    """
    def scale(self,x,y):
        new_dim=self.random_dim()
        sx=cv2.resize(x,(new_dim,new_dim),interpolation=cv2.INTER_NEAREST)
        sy=cv2.resize(y,(new_dim,new_dim),interpolation=cv2.INTER_NEAREST)

        #new_x=np.random.randint(255, size=x.shape,dtype=np.uint8)
        new_x=np.zeros(shape=x.shape,dtype=np.uint8)

        patch_dim=self.random_divisible_dim()
        for j in range(0,self.dim,patch_dim):
            for i in range(0,self.dim,patch_dim):
                target_x,target_y=self.random_pos(patch_dim)
                new_x[j:j+patch_dim,i:i+patch_dim]=x[target_x:target_x+patch_dim,target_y:target_y+patch_dim]
                


        new_y=np.zeros(shape=y.shape,dtype=np.uint8)

        diff_dim=self.dim-new_dim
        h_pos=random.randint(0,diff_dim)
        v_pos=random.randint(0,diff_dim)

        new_x[h_pos:h_pos+sx.shape[0],v_pos:v_pos+sx.shape[1]]=sx
        new_y[h_pos:h_pos+sy.shape[0],v_pos:v_pos+sy.shape[1]]=sy
        return new_x,new_y
    def process(self,x,y):
        if self.random_bool():
            x,y=self.hsv_noise(x,y)
        if self.random_bool():
            x,y=self.bad_resulation(x,y)
        if self.random_bool():
            x,y=self.scale(x,y)
        if self.random_bool():
            x,y=self.mirror(x,y)
        return x,y  
