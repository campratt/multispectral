import sys
import numpy as np
from os import listdir
import os
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose, Reshape, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose

    
        
versions_train = np.arange(20,100)
versions_test = np.arange(0,20)


fn_config = 'config.json'
config = json.load(open(fn_config))

metrics = ['MSE','MAE']


max_epochs = 300
patience = 20

save_dir = '../model/'


if os.path.exists(save_dir)==False:
    os.mkdir(save_dir)



### Data generator

d_X = '../data/X/'
d_Y = '../data/Y/'


N_per_map = 250

partition = {}

p_train = []
for i in versions_train:
    s = str(i)
    l = len(s)
    pp = '0'*(5-l)
    w = pp+s
    
    for j in range(N_per_map):
        p_train.append('id-%s-%i'%(w,j))
        


    
    
p_test = []
for i in versions_test:
    s = str(i)
    l = len(s)
    pp = '0'*(5-l)
    w = pp+s
    print(w)
    
    for j in range(N_per_map):
        p_test.append('id-%s-%i'%(w,j))






partition['train'] = p_train
partition['test'] = p_test





class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDD in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(self.d_X + IDD + '.npy')

            # Store class
            y[i,] = np.load(self.d_Y + IDD + '.npy')

        return X, y

training_generator = DataGenerator(partition['train'])
test_generator = DataGenerator(partition['test'])



import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
name = 'model'


###################################################################################################


filename = save_dir+'training.log'
def get_callbacks(name):
    return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
    tf.keras.callbacks.TensorBoard(logdir/name),\
    tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True),\
    tf.keras.callbacks.CSVLogger(
    filename, separator=',', append=True
)]






# Define the model
def CNN_3D(config):
    
    input_layer = Input((config['imagesize'],config['imagesize'],config['F']))

    conv_layer = input_layer

    conv_layer = keras.layers.Reshape((128,128,12,1))(conv_layer)


    for i in range(len(config['filters_3d'])):
        if config['dropout'] != False:
            if i == 0:
                pass
            else:
                conv_layer = Dropout(config['dropout'])(conv_layer)

        conv_layer = Conv3D(filters=config['filters_3d'][i], kernel_size=tuple(config['kernels_3d'][i]), strides=(1,1,1) ,padding='same', kernel_initializer="he_normal")(conv_layer)
        conv_layer = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(conv_layer)
        

        if config['batch_norm']:
            conv_layer = BatchNormalization()(conv_layer)
                
        
    conv_layer = keras.layers.Reshape((128,128,12))(conv_layer)
    for i in range(len(config['filters_2d'])):
        if config['dropout'] != False:
            conv_layer = Dropout(config['dropout'])(conv_layer)


        conv_layer = Conv2D(filters=config['filters_2d'][i], kernel_size=tuple(config['kernels_2d'][i]), strides=(1) ,padding='same')(conv_layer)
        conv_layer = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(conv_layer)

        if config['batch_norm']:
            conv_layer = BatchNormalization()(conv_layer)

    out = keras.layers.Conv2D(1, 1, padding='same', activation='linear')(conv_layer)


    model = keras.models.Model(inputs=input_layer, outputs=[out])
    return model






### Get the model


# Distribute training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = CNN_3D(config)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['LR']),
                        loss=config['loss_func'],
                        metrics=metrics)


#model.summary()



history = model.fit(
training_generator,
batch_size=config['BS'],
epochs=max_epochs,
validation_data=test_generator,
callbacks=get_callbacks(name),
verbose=2)
   

model.save(save_dir)



