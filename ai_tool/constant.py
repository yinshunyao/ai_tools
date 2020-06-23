#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/25 15:48
# @Author  : ysy
# @Site    : 
# @File    : constant.py
# @Software: PyCharm


# H5模型配置文件相关section命名
MODEL_CFG_FILE_NAME = "config.ini"

# 模型文件相关配置
# section
MODEL_CFG = "model"
# 模型文件
MODEL_FILE = "model"
# 模型name
MODEL_NAME = "name"
# 模型数据
MODEL_DATA = "data"
#
MODEL_IMAGE = "OnlyImage"
# 模型检测图片并发数
MODEL_CONCURRENCY = "concurrency"
# 模型参数配置
MODEL_PARAMS_CFG = "params"
# 检测类型 merge 和 delete两种类型
MODEL_DETECT_TYPE = "detect"
UNET_SAVE_PATH = "save_path"

# bbox配置section名称
SECTION_BBOX="bbox"

SECTION_CENTERNET = "centernet"


class DetectType:
    """检测类型，主要针对子模型返回结果处理"""
    # 杆号检测
    ganhao = 0
    # 分类检测
    classify = 1
    # 未检测
    not_detect = -1