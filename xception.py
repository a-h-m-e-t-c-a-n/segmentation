import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import backend as kbe
from tensorflow.keras import layers as kl

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = kl.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = kl.BatchNormalization()(x)
    if activation == True:
        x = kl.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = kl.LeakyReLU(alpha=0.1)(blockInput)
    x = kl.BatchNormalization()(x)
    blockInput = kl.BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = kl.Add()([x, blockInput])
    return x

def define_unet_xception(input_shape=(None, None),n_class=1,weights="imagenet",dropout_rate=0.2,feature_extraction_train_enable=False):
     
    input_layer = tf.keras.layers.Input(shape=input_shape+(3,))

    #noise_input=kl.GaussianNoise(0.2)(input_layer, training=True)

    backbone = k.applications.Xception(input_tensor=input_layer,weights=weights,include_top=False)

    
    #ac feature extraction için kullanılan encoder 
    #trainable kapalı tutlursa eğitim daha stabil ve hızlı oluyor
    backbone.trainable = feature_extraction_train_enable
    
    input = backbone.input
    start_neurons = n_class

    conv4 = backbone.layers[121].output
    conv4 = kl.LeakyReLU(alpha=0.1)(conv4)
    pool4 = kl.MaxPooling2D((2, 2))(conv4)
    pool4 = kl.Dropout(dropout_rate)(pool4)
    
     # Middle
    convm = kl.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = kl.LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = kl.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = kl.concatenate([deconv4, conv4])
    uconv4 = kl.Dropout(dropout_rate)(uconv4)
    
    uconv4 = kl.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = kl.LeakyReLU(alpha=0.1)(uconv4)
    
    deconv3 = kl.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = kl.concatenate([deconv3, conv3])    
    uconv3 = kl.Dropout(dropout_rate)(uconv3)
    
    uconv3 = kl.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = kl.LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = kl.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = kl.ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = kl.concatenate([deconv2, conv2])
        
    uconv2 = kl.Dropout(0.1)(uconv2)
    uconv2 = kl.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = kl.LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = kl.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = kl.ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = kl.concatenate([deconv1, conv1])
    
    uconv1 = kl.Dropout(0.1)(uconv1)
    uconv1 = kl.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = kl.LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = kl.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = kl.Dropout(dropout_rate)(uconv0)
    
    uconv0 = kl.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    
   
   #sparse catagorial loss aktivasyon çıktısı olmadan daha iyi çalışıyor bu kısımdan sonrası logic olacak
    uconv0 = kl.LeakyReLU(alpha=0.1)(uconv0) 
    
    #uconv0 = kl.Dropout(dropout_rate/2)(uconv0)
    #output_layer = kl.Conv2D(n_class,3,strides=1,padding="same", activation="sigmoid")(uconv0)    
    output_layer = kl.Conv2D(n_class,3,strides=1,padding="same")(uconv0)    

    #output_layer=uconv0

    model = k.Model(input, output_layer)
    model._name = 'u-xception'

    return model