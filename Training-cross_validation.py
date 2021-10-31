# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 02:58:34 2020

@author: youya
"""


from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,Dropout,
Flatten,BatchNormalization,Conv2D,MaxPooling2D,
Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

IMAGE_RES = 48
Batch_size = 64
base_dir = os.path.dirname("fer2013\\cross\\")
classes = ['Angry', 'Happy', 'Neutral', 'Sad']
num_classes = len(classes)
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    zoom_range=.2,
                    shear_range=.2,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    )

image_gen_val = ImageDataGenerator(rescale=1./255)

epochs = 50

#Callbacks Defined
checkpoint = ModelCheckpoint('Emotion.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta = 0,
                          patience = 9,
                          verbose = 1,
                          restore_best_weights = True
                          )

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 6,
                              verbose = 1,
                              min_delta = 0.0001)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="\\tensorBoard")

callbacks = [earlystop,reduce_lr,checkpoint,tensorboard_callback]

# for file in os.listdir(base_dir):
#     total_train = 0
#     total_val = 0
#     train_dir = os.path.join(base_dir,f"{file}\\train")
#     validation_dir = os.path.join(base_dir,f"{file}\\validation")
#     for cl in classes:   
#         img_path = os.path.join(train_dir,cl)
#         img_path_val = os.path.join(validation_dir,cl)
#         images = len(os.listdir(img_path))
#         images_val = len(os.listdir(img_path_val))
#         total_train+= images
#         total_val+= images_val
#         print("File {} {}: {} Train Images".format(file,cl, images))
#         print("File {} {}: {} Validation Images".format(file,cl,images_val))
#     print("Total Train Images: "+str(total_train))
#     print("Total Validation Images: "+str(total_val))
#     train_data_gen = image_gen_train.flow_from_directory(
#                                                 batch_size=Batch_size,
#                                                 color_mode='grayscale',
#                                                 directory=train_dir,
#                                                 shuffle=True,
#                                                 target_size=(IMAGE_RES,IMAGE_RES),
#                                                 class_mode='categorical',
#                                                 )

#     val_data_gen = image_gen_val.flow_from_directory( directory=validation_dir,
#                                                   color_mode='grayscale',
#                                                   batch_size=Batch_size,
#                                                   target_size=(IMAGE_RES, IMAGE_RES),
#                                                   class_mode='categorical',
#                                                   shuffle=True,
#                                                   )
#     history = None
#     if(file==str(1)):
#         history = model.fit_generator(train_data_gen,
#                                   steps_per_epoch=total_train//Batch_size,
#                                   epochs = epochs,
#                                   callbacks = callbacks,
#                                   validation_data = val_data_gen,
#                                   validation_steps = total_val//Batch_size)
#     else:
#         reloaded = tf.keras.models.load_model("Emotion.h5")
#         history = reloaded.fit_generator(
#                         train_data_gen,
#                         steps_per_epoch=total_train//Batch_size,
#                         epochs=epochs,
#                         callbacks=callbacks,
#                         validation_data=val_data_gen,
#                         validation_steps=total_val//Batch_size)
#         reloaded.save("Emotion_final.h5")
#     # To show statistics about training accuracy
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
    
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs_range = range(epochs)
#     plt.figure(figsize=(8, 8))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Training and Validation Accuracy')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Training Loss')
#     plt.plot(epochs_range, val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.title('Training and Validation Loss')
#     plt.show()

train_dir = os.path.join(base_dir,"1\\train")
validation_dir = os.path.join(base_dir,"1\\validation")
total_train = 0
total_val = 0
for cl in classes:   
    img_path = os.path.join(train_dir,cl)
    img_path_val = os.path.join(validation_dir,cl)
    images = len(os.listdir(img_path))
    images_val = len(os.listdir(img_path_val))
    total_train+= images
    total_val+= images_val
    
train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=Batch_size,
                                                color_mode='grayscale',
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMAGE_RES,IMAGE_RES),
                                                class_mode='categorical',
                                                )

val_data_gen = image_gen_val.flow_from_directory( directory=validation_dir,
                                                  color_mode='grayscale',
                                                  batch_size=Batch_size,
                                                  target_size=(IMAGE_RES, IMAGE_RES),
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  )
reloaded = tf.keras.models.load_model("Emotion.h5")
history = reloaded.fit_generator(
                        train_data_gen,
                        steps_per_epoch=total_train//Batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_data_gen,
                        validation_steps=total_val//Batch_size)
reloaded.save("Emotion_final.h5")



