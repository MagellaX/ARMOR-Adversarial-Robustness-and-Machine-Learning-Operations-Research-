import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.supported_datasets = ['mnist', 'cifar10']
    
    def load_data(self, dataset_name='mnist', normalize=True, reshape=True, validation_split=0.1):
        if dataset_name not in self.supported_datasets:
            raise ValueError(f'Dataset {dataset_name} not supported')
        
        if dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            img_shape = (28, 28, 1)
            num_classes = 10
        else:  # cifar10
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            img_shape = (32, 32, 3)
            num_classes = 10
            
        if reshape:
            if len(x_train.shape) == 3:
                x_train = x_train.reshape(-1, *img_shape)
                x_test = x_test.reshape(-1, *img_shape)
        
        if normalize:
            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
        
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        if validation_split > 0:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, 
                test_size=validation_split,
                random_state=42
            )
        else:
            x_val, y_val = None, None
            
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'img_shape': img_shape,
            'num_classes': num_classes
        }
