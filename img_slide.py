# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/31 0031下午 7:45
图片切片操作
"""
import os
import cv2
import logging
import numpy as np


def yield_sub_img(img_path, start_x, start_y, sub_width, sub_height, padding=True):
    """
    直接切割图片
    :param img_path: 图片路径或者直接传递cv2读取的图片
    :param start_x: x开始坐标
    :param start_y: y开始坐标
    :param sub_width: 切片宽度
    :param sub_height: 切片高度
    :param padding: 是否padding
    :param yield_flag: yield方式还是普通列表方式输出
    :return: box格式  [1400, 1400, 2448, 2050]  切片之后实际的x1，y1, x3, y3
              切片之后的img，如果是边缘，可能打过padding
    """
    # 传递路径
    if isinstance(img_path, str):
        if not os.path.isfile(img_path):
            raise Exception("img not exists: {}".format(img_path))
        else:
            slide = cv2.imread(img_path)
    # 直接传递的img
    else:
        slide = img_path

    # 获取图片的长宽
    sp = slide.shape
    WIDTH = sp[1]
    HEIGHT = sp[0]

    # 循环遍历，默认padding
    for x1 in range(start_x, WIDTH, sub_width):
        for y1 in range(start_y, HEIGHT, sub_height):
            try:
                box = [x1, y1, x1 + sub_width, y1 + sub_height]
                # 超出边框
                if box[2] > WIDTH or box[3] > HEIGHT:
                    # 不打padding
                    if not padding:
                        continue
                    # 填充0，打padding
                    else:
                        # 实际的x3和y3
                        box[2], box[3] = min(x1 + sub_width, WIDTH), min(y1 + sub_height, HEIGHT)
                        # 构造全0图片
                        img = np.zeros((sub_height, sub_width, 3), np.uint8)
                        # 将有效图片填充到这个空白图片中
                        img[0: box[3]-y1, 0: box[2]-x1] = slide[y1:box[3], x1:box[2]]
                else:
                    img = slide[y1:y1 + sub_height, x1:x1 + sub_width]

                # 返回框和图片
                yield box, img
            except Exception as e:
                logging.error("图片{}切片异常：{}".format(img_path, e), exc_info=1)