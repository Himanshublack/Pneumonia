import re
import os 
import random 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
  print("Devices: ", tpu.master())
except:
  strategy = tf.distribute.get_strategy()
print("Number of replicas: ", strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 25*strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]
CLASS_NAME = ["NORMAL", "PNEUMONIA"]

train_images = tf.data.TFRecordDataset(
    "D:/pneumonia/archive/zip/chest_xray/images"
)
#train_paths = tf.data.TFRecordDataset(
#    "D:/pneumonia/archive/zip/chest_xray/paths"
#)

ds = tf.data.Dataset.zip((train_images))

def get_label(file_path):
  parts = tf.strings.split(file_path, "/")
  return parts[-2] == "PNEUMONIA"

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels = 3)
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(image, path):
  label = get_label(path)
  img = decode_img(image)
  return img, label

ds = ds.map(process_path , num_parallel_calls = AUTOTUNE)

ds = ds.shuffle(10000)
train_ds = ds.take(4200)
val_ds = ds.skip(4200)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  
train_images = tf.data.TFRecordDataset(
    "D:/pneumonia/archive.zip/chest_xray/test/images.tfrec"
)
train_paths = tf.data.TFRecordDataset(
    "D:/pneumonia/archive.zip/chest_xray/test/paths.tfrec"
)

test_ds = tf.data.Dataset.zip((train_images, train_paths))
test_ds = test_ds.map(process_path, num_parallel_calls = AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)


def prepare_for_training(ds, cache= True):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size = AUTOTUNE)
  return ds

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))

from tensorflow import keras
from tensorflow.keras import layers

def conv_block(filters, inputs):
  x = layers.SeparableConv2D(filters, 3, activation= "relu", padding= "same")(inputs)
  x = layers.SeparableConv2D(filters, 3, activation= "relu", padding ="same")(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.MaxPool2D()(x)
  return outputs

def dense_block(units, dropout_rate, inputs):
  x = layers.Dense(units, activation="relu")(inputs)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dropout(dropout_rate)(x)
  return outputs

def build_model():
  inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

  x = layers.Rescaling(1.0/225)(inputs)
  x = layers.Conv2D(16, 3, activation= "relu", padding="same")(x)
  x = layers.Conv2D(16, 3, activation= "relu", padding="same")(x)
  x = layers.MaxPool2D()(x)

  x = conv_block(32, x)
  x = conv_block(64, x)

  x = conv_block(128, x)
  x = layers.Dropout(0.2)(x)

  x = conv_block(256, x)
  x = layers.Dropout(0.2)(x)

  x = layers.Flatten()(x)
  x = dense_block(512, 0.7, x)
  x = dense_block(256, 0.5, x)
  x = dense_block(64, 0.3, x)

  outputs = layers.Dense(1, activation = "sigmoid")(x)

  model = keras.Model(inputs = inputs, outputs = outputs)
  return model

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only = True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience = 10, restore_best_weights=True
)

initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps = 100000, decay_rate= 0.96, staircase = True
)
with strategy.scope():
  model = build_model()
  METRICS = [
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.Precision(name = "precision"),
      tf.keras.metrics.Recall(name = "recall")
  ]
  model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule),
      loss = "binary_crossentropy",
      metrics = METRICS,
  )

history = model.fit(
    train_ds,
    epochs = 100,
    validation_data= val_ds,
    class_weight = class_weight,
    callbacks = [checkpoint_cb, early_stopping_cb],
)

model.evaluate(test_ds, return_dict = True)

for image, label in test_ds.take(2):
  plt.imshow(image[15]/225.0)
  #plt.title(CLASS_NAME[label[0].numpy()])

prediction = model.predict(test_ds.take(1))[1]
scores = [1 - prediction, prediction]

for score, name in zip(scores, CLASS_NAME):
  print("This image is %.2f percent %s" % ((100* score), name))
