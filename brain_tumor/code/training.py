import sys
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from os import listdir
from multiprocessing import Pool
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv3D, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Add, Reshape, Lambda, Dense
import keras.backend as K
import json





fn_config = 'config.json'
config = json.load(open(fn_config))


metrics = ['accuracy']




max_epochs = 50
patience = 20


save_dir = '../model/'








### Data generator

# Normoalization
d_X = '../data/X/'
d_Y = '../data/Y/'

split_frac = 0.7

IDs = sorted(listdir(d_Y))
IDs = np.array([I.replace('.npy','') for I in IDs])
IDs_train = np.array(IDs[:int(split_frac*len(IDs))])
IDs_test = np.array(IDs[int(split_frac*len(IDs)):])



partition = {}

partition['train'] = IDs_train
partition['test'] = IDs_test





class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=config['BS'], dim=(config['imagesize'],config['imagesize']), n_channels=config['F'], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.d_X = d_X
        self.d_Y = d_Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, 2), dtype=np.float32)

        # Generate data
        for i, IDD in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(self.d_X + IDD + '.npy')

            # Store class
            y[i,] = np.load(self.d_Y + IDD + '.npy')

        return X, y

training_generator = DataGenerator(partition['train'])
test_generator = DataGenerator(partition['test'])






# Model
def CNN_3D(config):

    
    
    
    input_layer = Input((config['imagesize'],config['imagesize'],config['F']))

    conv_layer = input_layer


    conv_layer = keras.layers.Reshape((config['imagesize'],config['imagesize'],config['F'],1))(conv_layer)
    
    ## 3D convolutional layers
    for i in range(len(config['filters_3d'])):
        if config['dropout'] != False:
            if i == 0:
                pass
            else:
                conv_layer = Dropout(config['dropout'])(conv_layer)

        
        if config['activation']=='selu':
            conv_layer = Conv3D(filters=config['filters_3d'][i], kernel_size=tuple(config['kernels_3d'][i]), strides=(1,1,1) ,padding='same', kernel_initializer="lecun_normal", activation='selu')(conv_layer)
        else:
            conv_layer = Conv3D(filters=config['filters_3d'][i], kernel_size=tuple(config['kernels_3d'][i]), strides=(1,1,1) ,padding='same', kernel_initializer="he_normal")(conv_layer)
            conv_layer = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(conv_layer)
        

        if config['batch_norm']:
            conv_layer = BatchNormalization()(conv_layer)
                
        
    # 2D convolutional layers
    conv_layer = keras.layers.Reshape((config['imagesize'],config['imagesize'],config['F']))(conv_layer)
    for i in range(len(config['filters_2d'])):
        if config['dropout'] != False:
            conv_layer = Dropout(config['dropout'])(conv_layer)

        if config['activation']=='selu':
            conv_layer = Conv2D(filters=config['filters_2d'][i], kernel_size=tuple(config['kernels_2d'][i]), strides=(1) ,padding='same', kernel_initializer="lecun_normal",activation='selu')(conv_layer)
        else:
            conv_layer = Conv2D(filters=config['filters_2d'][i], kernel_size=tuple(config['kernels_2d'][i]), strides=(1) ,padding='same')(conv_layer)
            conv_layer = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(conv_layer)

        if config['batch_norm']:
            conv_layer = BatchNormalization()(conv_layer)


    # Dense layers
    conv_layer = keras.layers.Flatten()(conv_layer)
    for i in range(len(config['filters_1d'])):
            
        if config['dropout'] != False:
            conv_layer = Dropout(config['dropout'])(conv_layer)

        if config['activation']=='selu':
            conv_layer = Dense(config['filters_1d'][i], activation='selu')(conv_layer)
        else:
            conv_layer = Dense(config['filters_1d'][i])(conv_layer)
            conv_layer = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(conv_layer)

        if config['batch_norm']:
            conv_layer = BatchNormalization()(conv_layer)

    out = keras.layers.Dense(2, activation='softmax')(conv_layer)

    model = keras.models.Model(inputs=input_layer, outputs=[out])
    return model

        

    
        
model = CNN_3D(config)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['LR']),loss=config['loss_func'],metrics=metrics)


#model.summary()







###################################################################################################


filename = save_dir+'training.log'

def get_callbacks():
    return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
    tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True),\
    tf.keras.callbacks.CSVLogger(
    filename, separator=',', append=True
)
]




history = model.fit(
training_generator,
batch_size=config['BS'],
epochs=max_epochs,
validation_data=test_generator,
callbacks=get_callbacks(),
shuffle=True,
verbose=2)




model.save(save_dir)



