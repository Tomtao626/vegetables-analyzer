# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: predict
    Github: https://github.com/Tomtao626/vegetables_analyzer
    Author: tp320670258@gmail.com
    Description: $
    CreateDate: 2023-04-30 23:34
    Project: vegetables_analyzer
    IDE: PyCharm
"""
import requests
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = 'SimHei'
# mac/linux平台设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAVE_MODEL_PATH = "./vegetable_model_v1.h5"
TMP_IMAGE_PATH = "./tmp.png"
IMAGE_SIZE = (160, 160)

CLASS_NAMES = ['玉米', '茄子', '萝卜']


# 从网络获取图像数据
def get_image(img_url):
	res = requests.get(img_url, stream=True)
	if res.status_code == 200:
		with open(TMP_IMAGE_PATH, "wb") as f:
			f.write(res.content)
		return True
	else:
		return False


# 预测
def predict():
	test_image_src = image.load_img(TMP_IMAGE_PATH, target_size=IMAGE_SIZE)
	test_image_arr = image.img_to_array(test_image_src)
	test_image = tf.expand_dims(test_image_arr, axis=0)
	model = tf.keras.models.load_model(SAVE_MODEL_PATH)
	predictions = model.predict_on_batch(test_image)  # [0.2,0.5,0,3]

	plt.imshow(test_image_arr.astype("uint8"))
	plt.title(CLASS_NAMES[np.argmax(predictions[0])])
	plt.axis("off")
	plt.show()
	plt.close("all")


if __name__ == "__main__":
	# img_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg1.doubanio.com%2Fview%2Fgroup_topic%2Fl%2Fpublic%2Fp483155559.jpg&refer=http%3A%2F%2Fimg1.doubanio.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1644858065&t=151c9c38047c4e270c3aa2df3c66196c"
	img_url = "https://img2.baidu.com/it/u=2813381008,399015191&fm=253&fmt=auto&app=138&f=JPEG?w=300&h=225"
	# img_url = "https://img0.baidu.com/it/u=3586130753,410457657&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=360"
	# img_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fphoto.tuchong.com%2F21064663%2Ff%2F1208230255.jpg&refer=http%3A%2F%2Fphoto.tuchong.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1644858200&t=53f444935eea574f2357deddc7397a8b"
	# img_url = "https://img1.baidu.com/it/u=2004952455,2524217174&fm=253&fmt=auto&app=120&f=JPEG?w=639&h=454"
	# img_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnimg.ws.126.net%2F%3Furl%3Dhttp%3A%2F%2Fdingyue.ws.126.net%2F2021%2F0826%2F74c9fdbcj00qyfuuw000pc000hs00buc.jpg%26thumbnail%3D650x2147483647%26quality%3D80%26type%3Djpg&refer=http%3A%2F%2Fnimg.ws.126.net&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1644858348&t=e8fb13caf7680299641e03b611bf36da"
	if get_image(img_url):
		predict()
