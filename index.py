# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: index
    Github: https://github.com/Tomtao626/vegetables_analyzer
    Author: tp320670258@gmail.com
    Description: $
    CreateDate: 2023-05-01 00:06
    Project: vegetables_analyzer
    IDE: PyCharm
"""

from flask import Flask, request
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
from pathlib import Path
import hashlib, re, base64
import paho.mqtt.client as mqtt
import threading

MQTT_SUCCESS_STATUS = 0
HOST = "127.0.0.1" if os.getenv("PROD") else "192.168.124.54"
PORT = 1883
# 监听字段更新
DEVICE_NAME = os.getenv("DEVICE_NAME")
TOPIC = f"$hw/events/device/{DEVICE_NAME}/twin/update"
# 上传业务数据
TOPIC_UPLOAD = os.getenv("TOPIC_UPLOAD") if os.getenv("TOPIC_UPLOAD") else "default/upload"

# 当前模型加载路径
SAVE_MODEL_PATH = "./vegetable_model_v1.h5"
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
# 模型分片的存储文件夹
MODEL_PART_DIR = Path("./models_part")
MODEL_PART_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ['yumi', 'qiezi', 'luobo']
CURRENT_MODEL_PATH = MODEL_DIR / 'current_model.json'
if CURRENT_MODEL_PATH.exists():
    with open(CURRENT_MODEL_PATH, 'r', encoding='utf-8')as f:
        content = f.read()
        if content:
            model_dict = json.loads(content)
            SAVE_MODEL_PATH = model_dict['model_path']
            CLASS_NAMES = str(model_dict['class_names']).split(",")

app = Flask(__name__)
model = tf.keras.models.load_model(SAVE_MODEL_PATH)

IMAGE_SIZE = (160, 160)

SUCCESS = 0
FAILURE = 999


# 核心模型升级逻辑
def inner_upgrade_model(version, file_md5, class_names):
    if not (version and file_md5 and class_names):
        return failure("参数缺失"), 400
    # 合并模型
    model_part_dir = MODEL_PART_DIR / version
    if not model_part_dir.exists():
        return failure("模型分片不存在！"), 400
    model_path = MODEL_DIR / f'{version}.h5'
    merge(model_path, model_part_dir)
    with open(model_path, 'rb')as f:
        file_content = f.read()
    my_hash = hashlib.md5()
    my_hash.update(file_content)
    file_md5_src = my_hash.hexdigest()
    if file_md5 != file_md5_src:
        return failure("模型错误！"), 500
    # 重新加载模型，分类信息
    global model
    model = tf.keras.models.load_model(model_path)
    global CLASS_NAMES
    CLASS_NAMES = class_names.split(",")
    # 保存当前的模型
    current_model = {
        'model_path': str(model_path),
        'class_names': class_names
    }
    with open(CURRENT_MODEL_PATH, 'w', encoding='utf-8')as f:
        f.write(json.dumps(current_model, ensure_ascii=False))
    return success(
        {
            'detail': "模型升级成功！"
        }
    ), 200


class MQTTServer(threading.Thread):

    def __init__(self):
        super(MQTTServer, self).__init__()

    # 连接回调
    def on_connect(self, client, userdata, flags, rc):
        if rc == MQTT_SUCCESS_STATUS:
            print("mqtt connected!")
            version = str(Path(SAVE_MODEL_PATH).name).split(".")[0]
            with open(SAVE_MODEL_PATH, 'rb')as f:
                content = f.read()
                my_hash = hashlib.md5()
                my_hash.update(content)
                file_md5 = my_hash.hexdigest()
            class_names = ",".join(CLASS_NAMES)
            self.push_msg(version, file_md5, class_names)

    def upload_data(self, data):
        self.__client.publish(TOPIC_UPLOAD, data)

    def push_msg(self, version, file_md5, class_names):
        updated = {
            "event_id": "",
            "timestamp": 0,
            "twin": {
                "version": {
                    "actual": {
                        "value": version
                    },
                    "metadata": {
                        "type": "Updated"
                    },
                },
                "file_md5": {
                    "actual": {
                        "value": file_md5
                    },
                    "metadata": {
                        "type": "Updated"
                    },
                },
                "class_names": {
                    "actual": {
                        "value": class_names
                    },
                    "metadata": {
                        "type": "Updated"
                    },
                },
            }
        }
        updated_str = json.dumps(updated)
        self.__client.publish(TOPIC, bytes(updated_str, encoding='utf-8'), qos=0)

    # 消息接收回调
    def on_message(self, client, userdata, message):
        print(message.payload)
        msg_dict = json.loads(message.payload)

        # 上一次的数据
        try:
            last_version = msg_dict['twin']['version']['last']['expected']['value']
            last_file_md5 = msg_dict['twin']['file_md5']['last']['expected']['value']
            last_class_names = msg_dict['twin']['class_names']['last']['expected']['value']
        except:
            last_version = None
            last_file_md5 = None
            last_class_names = None

        # 当前这次的数据
        try:
            current_version = msg_dict['twin']['version']['current']['expected']['value']
            current_file_md5 = msg_dict['twin']['file_md5']['current']['expected']['value']
            current_class_names = msg_dict['twin']['class_names']['current']['expected']['value']
        except:
            current_version = None
            current_file_md5 = None
            current_class_names = None

        # 两次不等，执行更新
        if [last_version, last_file_md5, last_class_names] != [current_version, current_file_md5, current_class_names]:
            print("执行更新操作")
            res, code = inner_upgrade_model(current_version, current_file_md5, current_class_names)
            res_dict = json.loads(res)
            if res_dict['code'] == SUCCESS:
                self.push_msg(current_version, current_file_md5, current_class_names)
            else:
                print(res)

    def run(self):
        client = mqtt.Client()
        client.connect(HOST, PORT)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        self.__client = client
        client.subscribe(TOPIC + "/document", qos=0)
        client.loop_forever()


mqtt_server = MQTTServer()
mqtt_server.start()


# 成功返回
def success(data):
    res = {
        'code': SUCCESS,
        'msg': '成功',
        'data': data
    }
    return json.dumps(res, ensure_ascii=False)


# 失败返回
def failure(msg=None):
    res = {
        'code': FAILURE,
        'msg': '失败' if not msg else msg,
    }
    return json.dumps(res, ensure_ascii=False)


@app.route("/analyzer", methods=["POST"])
def analyzer():
    file = request.files.get("file")
    if file:
        try:
            image_bytes = file.read()
            test_image_src = Image.open(file)
            test_image_src = test_image_src.resize(IMAGE_SIZE, Image.ANTIALIAS)
            test_image_arr = image.img_to_array(test_image_src)
            test_image = tf.expand_dims(test_image_arr, axis=0)
            predictions = model.predict_on_batch(test_image)  # [0.2,0.5,0,3]
            class_name = CLASS_NAMES[np.argmax(predictions[0])]

            # 数据上报
            image_base64 = str(base64.b64encode(image_bytes))[2:]
            print(image_base64)
            upload_dict = {
                'image_base64': image_base64,
                'class_name': class_name
            }
            upload_dict_str = json.dumps(upload_dict)
            mqtt_server.upload_data(upload_dict_str)

            return success({
                'class_name': class_name
            }), 200
        except Exception as e:
            print(e)
            return failure(), 500
    else:
        return failure('file not found'), 500


# 接收云端的模型
@app.route("/receive_model_and_upgrade", methods=['POST'])
def receive_model_and_upgrade():
    # 接收参数，参数校验
    file = request.files.get('file')
    file_md5 = request.form.get('file_md5')
    class_names = request.form.get('class_names')
    if not (file and file_md5 and class_names):
        return failure("参数缺失！"), 400
    # 模型校验
    # 模型是否已经存在
    filename = file.filename
    save_path = MODEL_DIR / filename
    if save_path.exists():
        return failure("模型已存在！"), 500
    # 模型完整性校验
    file_content = file.read()
    my_hash = hashlib.md5()
    my_hash.update(file_content)
    file_md5_src = my_hash.hexdigest()
    if file_md5 != file_md5_src:
        return failure("模型错误！"), 500
    with open(save_path, 'wb')as f:
        f.write(file_content)
    # 重新加载模型，分类信息
    global model
    model = tf.keras.models.load_model(save_path)
    global CLASS_NAMES
    CLASS_NAMES = class_names.split(",")
    # 保存当前的模型
    current_model = {
        'model_path': str(save_path),
        'class_names': class_names
    }
    with open(CURRENT_MODEL_PATH, 'w', encoding='utf-8')as f:
        f.write(json.dumps(current_model, ensure_ascii=False))
    return success(
        {
            'detail': "模型上传成功！"
        }
    ), 200


@app.route("/receive_model", methods=['POST'])
def receive_model():
    # 收到文件，参数校验
    file = request.files.get("file")
    if not file:
        return failure("参数缺失"), 400
    filename = file.filename
    # vegetable_model_v2$0.part
    pattern = '\w+\$[0-9]{1,3}.part$'
    result = re.match(pattern, filename)
    if not result:
        return failure("文件格式错误"), 400
    # 保存
    model_part_dir = MODEL_PART_DIR / str(filename.split("$")[0])
    model_part_dir.mkdir(exist_ok=True)
    file.save(model_part_dir / filename)
    return success(
        {
            'detail': "模型分片上传成功！"
        }
    ), 200


# 模型合并
def merge(model_path, model_part_dir):
    with open(model_path, 'wb')as f:
        model_part_list = list(model_part_dir.glob("*/"))
        for i in sorted(model_part_list):
            with open(i, 'rb')as fs:
                f.write(fs.read())


@app.route("/upgrade_model", methods=['POST'])
def upgrade_model():
    # 接收参数，参数校验
    version = request.form.get("version")
    file_md5 = request.form.get("file_md5")
    class_names = request.form.get("class_names")
    return inner_upgrade_model(version, file_md5, class_names)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
