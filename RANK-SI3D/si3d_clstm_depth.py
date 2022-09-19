from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D, ConvLSTM2D
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
import random
import keras
from PIL import Image
from keras.layers import GlobalAveragePooling3D
from keras.optimizers import SGD
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

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

def process_line(path, line, F, T, N):
    data = np.empty((F, T, N, 1
                     ))   
    tmp = [val for val in line.strip().split(' ')]
    label = int(tmp[1])
    folder_name = tmp[0]
    folder_path = path + '/' + folder_name

    for ii in range(1,16):
        img_path = folder_path + '/' + folder_name + '_' + str(ii) + '.jpg'
        image = Image.open(img_path)

        image = image.resize((224,224))
        image = np.array(image)

        image = image/255
        data[ii-1, :, :, 0] = image   
    return data, label

def generate_depth_files(F, T, N, label_path, data_path, batch_size, num_classes,shuffle='true'):
    train = np.empty((batch_size, F, T, N, 1), dtype = 'float32')
    while 1:
        f = open(label_path)
        lines = f.readlines()
        if shuffle=='true':
            random.shuffle(lines)
        cnt = 0
        Label = []
        for line in lines:
            data, label = process_line(data_path, line, F, T, N)
            train[cnt, :, :, :, :] = data
            Label.append(label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                one_hot_labels = keras.utils.to_categorical(Label, num_classes = num_classes)
                
                yield (train, one_hot_labels)
                train = np.empty((batch_size, F, T, N, 1), dtype = 'float32')
                Label = []
    f.close()

def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias = False,
              use_activation_fn = True,
              use_bn = True,
              name=None):
    
    bn_name = name + '_bn'
    conv_name = name + '_conv'
    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)
    
    bn_axis = 4
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x
    
def SI3D_CLSTM(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                classes=30):
    inputs = Input(shape=(15, 224, 224, 1))  
    channel_axis = 4
    
    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(inputs, 64, 7, 7, 7, strides=(1, 2, 2), padding='same', name='Conv3d_1a_7x7_c')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(1, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')
    
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', return_sequences=False)(x)
    
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    #x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs, x)

    return model

if __name__ == '__main__':
     
    #basic parameters
    num_classes = 30
    batch_size= 16
    epochs = 50
    num_train_data = 5528 
    num_test_data = 2886

    T, N = 224, 224
    
    #path
    train_label_path = './label_train.txt'
    train_data_path = './rankdepth_front_train'
    
    test_label_path = './label_test.txt'
    test_data_path = './rankdepth_front_test'
    
    weight_path = './weight/depthfront_si3dclstm.{val_acc:.2f}-{epoch:02d}.h5'

    #data generator
    train_data_generator = generate_depth_files(F=F, T=T, N=N, label_path=train_label_path, 
                                                data_path=train_data_path, batch_size=batch_size, 
                                                num_classes=num_classes,shuffle='true')
    test_data_generator = generate_depth_files(F=F, T=T, N=N, label_path=test_label_path, 
                                                data_path=test_data_path, batch_size=1, 
                                                num_classes=num_classes,shuffle='false')

    model = SI3D_CLSTM(classes=num_classes)
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
                        validation_steps=num_test_data/1,
                        callbacks=callbacks_list, verbose=1)

    