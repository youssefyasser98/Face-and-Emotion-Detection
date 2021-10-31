# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:58:40 2020

@author: youya
"""
import tensorflow as tf
import pathlib

model = tf.keras.models.load_model(filepath="Emotion.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
# open("tf_lite_model.tflite", "wb").write(tflite_model)

tflite_model_file = pathlib.Path("Emotion_lite.tflite")
tflite_model_file.write_bytes(tflite_model)


# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.set_tensor(input_details[0]['index'],input_data)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
