# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: trains
    Github: https://github.com/Tomtao626/vegetables_analyzer
    Author: tp320670258@gmail.com
    Description: 模型切分
    CreateDate: 2023-04-30 23:20
    Project: vegetables_analyzer
    IDE: PyCharm
"""

from pathlib import Path
import requests


def cut():
	src = Path('../vegetable_model_v2.h5')
	tmp_dir = Path('./tmp_dir')
	tmp_dir.mkdir(exist_ok=True)

	# 切分分片大小
	buf_size = 1 * 1024 * 1024
	index = 0
	with open(src, 'rb') as f1:
		while True:
			buf = f1.read(buf_size)
			if buf:
				save_path = tmp_dir / f'vegetable_model_v2${index}.part'
				index += 1
				with open(save_path, 'wb') as f2:
					f2.write(buf)
				res = requests.post(
					url='http://192.168.124.53:9443/imooc-edge02/default/receive_model',
					files={
						'file': open(save_path, 'rb')
					}
				)
				print(res.text)

			else:
				break


cut()
