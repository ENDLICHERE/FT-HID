# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:26:57 2019

@author: a
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
from keras.optimizers import SGD
from PIL import Image
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Multiply, multiply, add
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
import keras
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers import GRU,Bidirectional,Activation,LSTM, DepthwiseConv2D,SeparableConv2D
from keras.layers import Flatten,RepeatVector,Multiply,Add,Permute,Reshape,Concatenate,Lambda,Average
import scipy.io as sio 
#import cv2 
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  
session = tf.Session(config=config)
KTF.set_session(session)

def lr_scheduler(epoch):

    if epoch > 0.75 * epochs:
        lr = 0.000001
    elif epoch > 0.5 * epochs:
        lr = 0.00001
    elif epoch > 0.25 * epochs:
        lr = 0.0001
    else:
        lr = 0.001 

    print('lr: %f' % lr)
    return lr 

def process_line(path,line):

    tmp = [val for val in line.strip().split(' ')]
    name=tmp[0]
    label=int(tmp[1])
    data_path=path + '/' + name  
    #image=Image.open(data_path)
    image = sio.loadmat(data_path, verify_compressed_data_integrity=False)
    image = image['skel']
    data=np.array(image)
    
    return data, label
 
def generate_arrays_from_file(T, N, label_path, data_path_1, data_path_2, batch_size, num_classes,shuffle='false'):
    train1 = np.empty((batch_size, T, N, 3), dtype = "float32")
    train2 = np.empty((batch_size, T, N, 3), dtype = "float32")

    while 1:
        f = open(label_path)
        lines=f.readlines()
        if shuffle=='true':
            random.shuffle(lines)
        cnt = 0
        Label=[]
        for line in lines:

            data1, label = process_line(data_path_1, line)
            data2, label2 = process_line(data_path_2, line)

            train1[cnt,:,:,:]=data1
            train2[cnt,:,:,:]=data2

            Label.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                one_hot_labels = keras.utils.to_categorical(Label, num_classes=num_classes)
                
                yield ([train1, train2], one_hot_labels)
                train1 = np.empty((batch_size, T, N, 3), dtype = "float32")
                train2 = np.empty((batch_size, T, N, 3), dtype = "float32")

                Label=[]
    f.close()
 
def MVI_block(x1, x2, nb_filter_1, stage, block):
    
    MVIB_name_base = 'mvib' + str(stage) + str(block)
    
    width_in = [55, 28, 14, 7]
    width = width_in[int(stage)-2]
    
    x = layers.concatenate([x1, x2],axis=-1)
    
    x = DepthwiseConv2D((width, width), strides=(1, 1),  depth_multiplier=1, name='gdconv'+MVIB_name_base)(x)

    x = Dense(int(nb_filter_1/16*2), name='fc'+MVIB_name_base+'1')(x)
    x = Activation('relu', name='relu'+MVIB_name_base)(x)
    x = Dense(nb_filter_1*2, name='fc'+MVIB_name_base+'2_1')(x)
    x = Reshape((nb_filter_1, 2))(x)
    y = Activation('softmax', name='softmax'+MVIB_name_base)(x)
    
    y1 = Lambda(lambda y: y[:, :, :1])(y)
    y2 = Lambda(lambda y: y[:, :, 1:2])(y)
    
    y1 = Reshape((1, 1, nb_filter_1))(y1)
    y2 = Reshape((1, 1, nb_filter_1))(y2)
    
    x1 = layers.multiply([x1, y1])
    x2 = layers.multiply([x2, y2])

    return x1, x2


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=None):

    # Determine proper input shape
   
    img_input_1 = Input(shape=(112, 112, 3), name='img_input_1')
    img_input_2 = Input(shape=(112, 112, 3), name='img_input_2')
    
    global concat_axis
    bn_axis = 3
    
    
    x_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv1_01')(img_input_1)
    x_1= BatchNormalization(axis=bn_axis, name='bn_conv1_01')(x_1)
    x_1= Activation('relu')(x_1)
    x_1= MaxPooling2D((3, 3), strides=(2, 2))(x_1)

    x_2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv1_02')(img_input_2)
    x_2= BatchNormalization(axis=bn_axis, name='bn_conv1_02')(x_2)
    x_2= Activation('relu')(x_2)
    x_2= MaxPooling2D((3, 3), strides=(2, 2))(x_2)
    
    x_1 = conv_block(x_1, 3, [64, 64, 256], stage=2, block='a_1', strides=(1, 1))
    x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='b_1')
    x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='c_1')
    
    x_2 = conv_block(x_2, 3, [64, 64, 256], stage=2, block='a_2', strides=(1, 1))
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='b_2')
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='c_2')
    
    x_1, x_2 = MVI_block(x_1, x_2, nb_filter_1=256, stage=2, block='r_0')

    x_1 = conv_block(x_1, 3, [128, 128, 512], stage=3, block='a_1')
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='b_1')
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='c_1')
    x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='d_1')
    
    x_2 = conv_block(x_2, 3, [128, 128, 512], stage=3, block='a_2')
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='b_2')
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='c_2')
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='d_2')

    x_1, x_2 = MVI_block(x_1, x_2, nb_filter_1=512, stage=3, block='r_1')

    x_1 = conv_block(x_1, 3, [256, 256, 1024], stage=4, block='a_1')
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='b_1')
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='c_1')
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='d_1')
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='e_1')
    x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='f_1')
    
    x_2 = conv_block(x_2, 3, [256, 256, 1024], stage=4, block='a_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='b_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='c_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='d_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='e_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='f_2')
    
    x_1, x_2 = MVI_block(x_1, x_2, nb_filter_1=1024, stage=4, block='r_2')

    x_1 = conv_block(x_1, 3, [512, 512, 2048], stage=5, block='a_1')
    x_1 = identity_block(x_1, 3, [512, 512, 2048], stage=5, block='b_1')
    x_1 = identity_block(x_1, 3, [512, 512, 2048], stage=5, block='c_1')
    
    x_2 = conv_block(x_2, 3, [512, 512, 2048], stage=5, block='a_2')
    x_2 = identity_block(x_2, 3, [512, 512, 2048], stage=5, block='b_2')
    x_2 = identity_block(x_2, 3, [512, 512, 2048], stage=5, block='c_2')
    
    x_1, x_2 = MVI_block(x_1, x_2, nb_filter_1=2048, stage=5, block='r_3')

    x_1 = AveragePooling2D((7, 7), name='avg_pool_1')(x_1)
    out_1 = Flatten()(x_1)
    
    x_2 = AveragePooling2D((7, 7), name='avg_pool_2')(x_2)
    out_2 = Flatten()(x_2)
    
    x = keras.layers.concatenate([out_1, out_2], axis=-1)

    x = Dense(classes, activation='softmax', name='fc'+str(classes))(x)
    
    inputs = [img_input_1, img_input_2]
    model = Model(inputs, x, name='resnet50_se')

    return model


if __name__ == '__main__':
    
    # load parameters  
    num_classes = 30
    batch_size = 16
    epochs = 50
    # Due to the loss or corruption of some skeleton file, it ended up with 8079 skeleton samples per view.
    num_train_data = 4365 
    num_test_data = 3714 
    T, N = 112, 112


    # data preparation
    train_label_path = './label_train.txt'
    train_data_path_1 = './front_skl_train'
    train_data_path_2 = './side_skl_train'

    test_label_path = './label_test.txt'
    test_data_path_1 = './front_skl_test' 
    test_data_path_2 = './side_skl_test' 

    weight_path = './weight/mvib_skl.{val_acc:.2f}-{epoch:02d}.h5'

    train_data_generator=generate_arrays_from_file(T=T, N=N, label_path=train_label_path, data_path_1=train_data_path_1, 
                                                   data_path_2=train_data_path_2,
                                                   batch_size=batch_size, num_classes=num_classes, shuffle='true')

    test_data_generator=generate_arrays_from_file(T=T, N=N, label_path=test_label_path, data_path_1=test_data_path_1, 
                                                   data_path_2=test_data_path_2,
                                                   batch_size=1, num_classes=num_classes, shuffle='false')
    # load model
    model = ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=num_classes)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(weight_path, monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    mode='max', period=5)
    callbacks_list = [checkpointer, LearningRateScheduler(lr_scheduler)]
    
    #train model
    history = model.fit_generator(train_data_generator,
                        steps_per_epoch=num_train_data/batch_size, 
                        epochs=epochs,
                        validation_data=test_data_generator,
                        validation_steps=num_test_data/batch_size,
                        callbacks=callbacks_list, verbose=1)


