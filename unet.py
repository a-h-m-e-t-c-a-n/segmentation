import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import backend as kbe
from tensorflow.keras import layers as kl

def define_unet(img_size,n_classes=1):
    inputs = k.Input(shape=img_size)

    x = kl.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation(activation=k.activations.swish)(x)

    previous_block_activation = x  # Set aside residual

    for filters in [64, 128, 256]:
        x = kl.Activation(activation=k.activations.swish)(x)
        x = kl.SeparableConv2D(filters, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)
        

        x = kl.Activation(activation=k.activations.swish)(x)
        x = kl.SeparableConv2D(filters, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)
       
        x = kl.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = kl.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = kl.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    for filters in [256, 128, 64, 32]:
        x = kl.Activation(activation=k.activations.swish)(x)
        x = kl.Conv2DTranspose(filters, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)


        x = kl.Activation(activation=k.activations.swish)(x)
        x = kl.Conv2DTranspose(filters, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)
       


        x = kl.UpSampling2D(2)(x)


        residual = kl.UpSampling2D(2)(previous_block_activation)
        residual = kl.Conv2D(filters, 3, padding="same")(residual)
        x = kl.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual



    
    #x = kl.Conv2D(n_classes, 1, activation="sigmoid", padding="same")(x)
    x = kl.Conv2D(n_classes, 3,  padding="same")(x)
  
    outputs=x
    model = k.Model(inputs, outputs)
    return model
