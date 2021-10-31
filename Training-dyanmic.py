# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:43:17 2020

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
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#Variables
LOG_DIR = f"tunnerLogs/{int(time.time())}"
IMAGE_RES = 48
Batch_size = 64
base_dir = os.path.dirname("fer2013\\")
image_file_paths = ["fer2013\\Angry","fer2013\\Happy","fer2013\\Neutral","fer2013\\Angry"]
classes = ['Angry', 'Happy', 'Neutral', 'Sad']
num_classes = len(classes)
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
total_train=0
total_val=0
for cl in classes:   
  img_path = os.path.join(train_dir,cl)
  img_path_val = os.path.join(validation_dir,cl)
  images = len(os.listdir(img_path))
  images_val = len(os.listdir(img_path_val))
  total_train+= images
  total_val+= images_val
  print("{}: {} Train Images".format(cl, images))
  print("{}: {} Validation Images".format(cl,images_val))

print("Total Train Images: "+str(total_train))
print("Total Validation Images: "+str(total_val))

#Building Model
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int("input_units_1",32,256,32),(3,3),
                     padding='same',kernel_initializer='he_normal',
                     input_shape=(IMAGE_RES,IMAGE_RES,1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(hp.Int("input_units_2",32,256,32),(3,3),
                     padding='same',kernel_initializer='he_normal',
                     input_shape=(IMAGE_RES,IMAGE_RES,1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    for i in range(hp.Int("n_layers",1,7)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units",32,512,32),(3,3),
                     padding='same',kernel_initializer='he_normal',
                     input_shape=(IMAGE_RES,IMAGE_RES,1)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(hp.Int(f"conv_{i}_units",32,512,32),(3,3),
                     padding='same',kernel_initializer='he_normal',
                     input_shape=(IMAGE_RES,IMAGE_RES,1)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(hp.Int("Dense_layer_1",256,1024,128),kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(hp.Int("Dense_layer_2",256,1024,128),kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes,
                kernel_initializer='he_normal',
                activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer = Adam(),
              metrics=['accuracy'])

    #model.summary()
    return model


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

callbacks = [earlystop,checkpoint,reduce_lr]
tunner_callbacks = [earlystop,reduce_lr]
#Tuning the model 
tuner = RandomSearch(
    build_model,
    objective = "val_accuracy",
    max_trials = 1,
    executions_per_trial = 1,
    directory = LOG_DIR,
    project_name="See_As_We_See"
    )
tuner.search_space_summary()
tuner.search(train_data_gen,
              steps_per_epoch=total_train//Batch_size,
              callbacks=tunner_callbacks,
              epochs = 40,
              validation_data = val_data_gen)

tuner.results_summary()
print("----------------------------------------------------------------------")
print(tuner.get_best_hyperparameters()[0].values)

#Executing the model
# epochs = 32
# model = build_model(1)
# history = model.fit_generator(
#                 train_data_gen,
#                 steps_per_epoch=total_train//Batch_size,
#                 epochs = epochs,
#                 callbacks = callbacks,
#                 validation_data = val_data_gen,
#                 validation_steps = total_val//Batch_size)


# reloaded = tf.keras.models.load_model("Emotion.h5")
# history = reloaded.fit_generator(
#                 train_data_gen,
#                 steps_per_epoch=total_train//Batch_size,
#                 epochs=1,
# #                 callbacks=callbacks,
#                 validation_data=val_data_gen,
#                 validation_steps=total_val//Batch_size)

# To show statistics about training accuracy
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(epochs)
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()













