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


def yield_sub_img(img_path, start_x, start_y, sub_width, sub_height, WIDTH=0, HEIGHT=0, padding=True):
    """
    直接切割图片，正常调用，如下，其他参数传递方式均为测试用
    yield_sub_img(img_path=img, start_x=0, start_y=0, sub_width=800, sub_height=800)
    :param img_path: 图片路径或者直接传递cv2读取的图片， 传递空或者None时，只处理参数
    :param start_x: x开始坐标
    :param start_y: y开始坐标
    :param sub_width: 切片宽度， 切片高或者宽为0，则整图返回
    :param sub_height: 切片高度， 切片高或者宽为0，则整图返回
    :param padding: 是否padding
    :param WIDTH: 图片宽度，切图之前会计算宽度，因此切图时直接使用外部的宽度即可
    :param HEIGHT: 图片高度
    :return: box格式  [1400, 1400, 2448, 2050]  切片之后实际的x1，y1, x3, y3
              切片之后的img，如果是边缘，可能打过padding
    """
    # 不传递图片时，仅生成切片参数，分支跟else冗余，方便阅读
    if img_path is None:
        slide = None
    # 传递路径
    elif isinstance(img_path, str):
        if not os.path.isfile(img_path):
            raise Exception("img not exists: {}".format(img_path))
        else:
            slide = cv2.imread(img_path)
    # 直接传递的img
    else:
        slide = img_path

    # 如果没有携带宽和高，必须从图片中获取
    if not WIDTH or not HEIGHT:
        WIDTH = slide.shape[1]
        HEIGHT = slide.shape[0]

    # 切片高或者宽为0，则整图返回
    if not sub_height or not sub_width:
        yield [0, 0, WIDTH, HEIGHT], slide
        return

    x_list = []
    # 如果start_x不等于0，且不超过宽度，左边切片一列
    if 0 < start_x <= sub_width:
        x_list.append(0)

    # 构造切片左上角x起始坐标集合
    x_list.extend(list(range(start_x, WIDTH, sub_width)))
    # 循环遍历，默认padding
    for x1 in x_list:
        for y1 in range(start_y, HEIGHT, sub_height):
            try:
                # 初始化为None，可以不返回图片，直接返回切片参数
                img = None
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
                        if slide is None:
                            pass
                        else:
                            # 构造全0图片
                            img = np.zeros((sub_height, sub_width, 3), np.uint8)
                            # 将有效图片填充到这个空白图片中
                            img[0: box[3]-y1, 0: box[2]-x1] = slide[y1:box[3], x1:box[2]]
                else:
                    if slide is None:
                        pass
                    else:
                        img = slide[y1:y1 + sub_height, x1:x1 + sub_width]

                # 返回框和图片
                yield box, img
            except Exception as e:
                logging.error("图片{}切片异常：{}".format(img_path, e), exc_info=1)

if __name__ == '__main__':
    for box, img in yield_sub_img(None, 0, 0, 1400, 1400, 6576, 4384):
        print(box[0], box[1], box[2]-box[0], box[3]-box[1])