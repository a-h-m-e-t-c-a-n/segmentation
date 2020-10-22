import numpy as np
import tensorflow as tf
from  tensorflow.keras.utils import to_categorical
import tensorflow.keras as k


def to_onehot(data,n_class=None):
   return to_categorical(data,num_classes=n_class) 
   
def from_onehot(data):
    return np.expand_dims(np.argmax(data,axis=-1),-1)

def mask_from_sparse(prediction):
  prediction_mask = tf.argmax(prediction, axis=-1)
  prediction_mask = prediction_mask[..., tf.newaxis]
  return prediction_mask
def mask_from_sparse_as_np(prediction):
  prediction_mask = np.argmax(prediction, axis=-1)
  return prediction_mask

def build_accuracy_for_sparse():
    accuracy_base=k.metrics.Accuracy()
    def accuracy(y_true, y_pred):
        y_pred_mask=mask_from_sparse(y_pred)
        accuracy_base.update_state(y_true,y_pred_mask)
        return accuracy_base.result()
    return accuracy
