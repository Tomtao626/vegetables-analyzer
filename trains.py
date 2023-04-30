# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: trains
    Github: https://github.com/Tomtao626/vegetables_analyzer
    Author: tp320670258@gmail.com
    Description: $
    CreateDate: 2023-04-30 22:48
    Project: vegetables_analyzer
    IDE: PyCharm
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = 'SimHei'
# mac/linux平台设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 提取数据集
PATH = Path("data_edgeai-app")

train_dir = PATH / "train"
validation_dir = PATH / "validation"

# 模型保存路径
SAVE_MODEL_PATH = "./vegetable_model_v2.h5"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

train_dataset = image_dataset_from_directory(
	train_dir,
	label_mode="categorical",
	shuffle=True,
	image_size=IMAGE_SIZE,
	batch_size=BATCH_SIZE,
)

validation_dataset = image_dataset_from_directory(
	validation_dir,
	label_mode="categorical",
	shuffle=True,
	image_size=IMAGE_SIZE,
	batch_size=BATCH_SIZE,
)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 6)
validation_dataset = validation_dataset.skip(val_batches // 6)

class_names = train_dataset.class_names

print(class_names)
#
# 提前加载数据，防止IO阻塞
AUTOTUNE = tf.data.AUTOTUNE
# 数据加载优化
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# 显示图像，看效果
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):  # labels [0,1,0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[np.argmax(labels[i])])
#         plt.axis("off")
# plt.show()
# plt.close("off")

# 图像水平变换，旋转
data_augmentation = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
	tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

# 图像像素缩放到[-1,1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

IMG_SHAPE = IMAGE_SIZE + (3,)

base_model = tf.keras.applications.MobileNetV2(
	input_shape=IMG_SHAPE,
	include_top=False,
	weights="imagenet")
# 冻结卷积基
base_model.trainable = False

# 分类层
prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")

# 测试base_model是否可用
# images,labels = next(iter(train_dataset))
# feature_batch = base_model(images)
# print(feature_batch)

# 显示图像，看翻转和旋转效果
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):  # labels [0,1,0]
#     first_image = images[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image,0))
#         plt.imshow(augmented_image[0] / 255)
#         # plt.title(class_names[np.argmax(labels[i])])
#         plt.axis("off")
# plt.show()
# plt.close("off")

# 2. 特征提取
inputs = tf.keras.Input(IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

# 3. 模型训练 得到模型
base_learning_rate = 0.0001
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.summary()
initial_epochs = 10
history = model.fit(
	train_dataset,
	epochs=initial_epochs,
	validation_data=validation_dataset
)
loss, accuracy = model.evaluate(test_dataset)
print("Test accurary:", accuracy)
model.save(SAVE_MODEL_PATH)
# 4. 评估模型的准确率
