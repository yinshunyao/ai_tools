#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 13:12
# @Author  : ysy
# @Site    : 
# @File    : model_config.py
# @Software: PyCharm
from configparser import ConfigParser
import logging
import os
from GTUtility.GTTools.constant import *


class ModelConfigLoad(object):
    def __init__(self, model_path):
        self.model_path = model_path
        # 构造ini配置文件绝对路径
        ini_cfg_path = os.path.join(model_path, MODEL_CFG_FILE_NAME)  # /data2/model/uring/config.ini
        logging.warning("load config from file:{}".format(ini_cfg_path))
        self.ini_config = ConfigParser()
        self.ini_config.read(ini_cfg_path, encoding="utf-8")
        # 模型类型 unet或者其他
        self.model_type = self.ini_config.get(MODEL_CFG, "Type")
        # 传递给模型的是否是图片，或者文件路径
        self.image_only = self.ini_config.get(MODEL_CFG, "OnlyImage", fallback="false").strip() == "true"
        # model文件
        self.model_file = os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, MODEL_FILE, fallback=""))  # /data2/model/uring/mask_rcnn_4c_defect_0275.h5
        self.name_class = os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, MODEL_NAME, fallback=""))
        # # 并发数 与 thread_max重复
        # self.concurrency = int(self.ini_config.get(MODEL_CFG, MODEL_CONCURRENCY, fallback=1))
        # 切片的长宽，默认值为0,表示不切片
        self.patch_width = int(self.ini_config.get(MODEL_CFG, "width", fallback=0))
        self.patch_height = int(self.ini_config.get(MODEL_CFG, "height", fallback=0))
        self.padding = (self.ini_config.get(MODEL_CFG, "padding", fallback="true")).lower() == "true"
        logging.warning("切片配置：宽-{}，高-{}，padding-{}".format(self.patch_width, self.patch_height, self.padding))
        # 调试模式
        self.debug = bool((self.ini_config.get(MODEL_CFG, "debug", fallback="false")).lower() == "true")
        # 线程数
        self.thread_max = int(self.ini_config.get(MODEL_CFG, "thread_max", fallback=1))

        # 黑名单，主要针对4C，部分图片在特定的模型不需要检测
        # 例如015946004_K292703_24_5_01.jpg 表示杆号，不需要检测管帽缺陷
        self.black_list = [item.strip() for item in self.ini_config.get(MODEL_CFG, "black_list", fallback="").split(",") if not not item.strip()]
        logging.warning("模型黑名单配置：{}".format(self.black_list))

        # 配置是否去掉边缘不完整器件出的框
        self.edge = bool((self.ini_config.get(MODEL_CFG, "edge", fallback="false")).lower() == "true")
        logging.warning("模型去除边缘不完整器件框配置：{}".format(self.edge))

        # 用户配置门限全局使用
        self.thresh_user = float(self.ini_config.get(MODEL_CFG, "thresh_user", fallback=0.8))
        self.thresh_model = float(self.ini_config.get(MODEL_CFG, "thresh_model", fallback=0)) or self.thresh_user
        self.thresh_ai = float(self.ini_config.get(MODEL_CFG, "thresh_ai", fallback=0)) or self.thresh_user

        # self.relative_path = self.ini_config.get(MODEL_CFG, "relative_path", fallback="")
        self.train_type = self.ini_config.get(MODEL_CFG, "train_type", fallback="")
        self.model_path_str = self.ini_config.get(MODEL_CFG, "model_path_str", fallback="")
        # 子模型配置参数， 按照列表方式初始化
        self.sub_model_cfg = []
        self._load_sub_model_cfg()

        # 进入子模型之前扩展像素
        self.padding_pixel = int(self.ini_config.get(MODEL_CFG, "padding_pixel", fallback=0))

        # 几何参数特征构造
        self.bbox_geo_feature = self._get_bbox_geo_params()
        logging.warning("几何参数配置信息{}".format(self.bbox_geo_feature))
        logging.warning("配置文件读取完成")

    def _load_sub_model_cfg(self, section_flag="sub_model_"):
        """
        加载子模型的配置，目前仅支持res50模型
        :param section_flag:
        :return:
        """
        for i in range(1, 15):
            section = "{}{:0>2d}".format(section_flag, i)
            if section not in self.ini_config.sections():
                logging.warning("子模型配置获取结束")
                break

            logging.warning("发现子模型 {}".format(section))
            # 配置参数转换成字典
            config_params = dict(self.ini_config.items(section))
            if not config_params.get(MODEL_CFG, "") or not config_params.get(MODEL_NAME, ""):
                logging.error("模型{}没有配置子模型{}的路径或者名称文件，将不会加载".format(self.model_type, section))
                continue

            # 更新为绝对路径
            config_params.update({
                MODEL_CFG: os.path.join(self.model_path, config_params[MODEL_CFG]),
                MODEL_NAME: os.path.join(self.model_path, config_params[MODEL_NAME])
            })
            self.sub_model_cfg.append(config_params)

    def _get_bbox_geo_params(self):
        """
        bbox几何参数配置，构造参数字典，供bbox中judge_by_geo使用
        w_th_min = 40 # 宽度最小阈值
        w_th_max = 560  # 宽度最大阈值
        h_th_min = 50 # 高度最小阈值
        h_th_max = 620 # 高度最大阈值
        hw_ratio_th_min = 0.7 # 长宽比阈值
        hw_ratio_th_max = 1.95 # 长宽比阈值
        :return:
        """
        geo_params = {
            'w_range': None,
            'h_range': None,
            'w_to_h_range': None,
            's_range': None
        }
        try:
            if SECTION_BBOX not in self.ini_config.sections():
                return geo_params

            params = dict(self.ini_config.items(SECTION_BBOX))
            for k, v in params.items():
                if isinstance(v, str):
                    params[k] = v.strip()
            # 宽度范围
            if 'w_th_min' in params.keys() and 'w_th_max' in params.keys():
                geo_params['w_range'] = [float(params['w_th_min']), float(params['w_th_max'])]
            # 高度范围
            if 'h_th_min' in params.keys() and 'h_th_max' in params.keys():
                geo_params['h_range'] = [float(params['h_th_min']), float(params['h_th_max'])]
            # 宽高比范围
            if 'hw_ratio_th_min' in params.keys() and 'hw_ratio_th_max' in params.keys():
                geo_params['w_to_h_range'] = [float(params['hw_ratio_th_min']), float(params['hw_ratio_th_max'])]
        except Exception as e:
            logging.warning("模型{}解析几何参数配置异常:{}".format(self.model_file, e), exc_info=1)
            raise Exception("配置加载失败")

        return geo_params

    @property
    def cfg_file(self):
        """
        配置文件，部分模型可能不存在
        :return:
        """
        return os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, "cfg"))

    @property
    def data_file(self):
        """
        数据文件，部分模型可能不存在
        :return:
        """
        return os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, "data"))

    @property
    def name_file(self):
        return os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, MODEL_NAME))

    @property
    def update_config_items(self):
        """
        更新参数
        :return:
        """
        return self.ini_config.items(MODEL_PARAMS_CFG)

    @property
    def save_path(self):
        save_path = self.ini_config.get(MODEL_CFG, "save_path", fallback="")
        # 如果没有配置，取模型当前路径
        if not save_path:
            logging.warning("save path for {} not config, will save to {}".format(self.model_type, self.model_path))
            return self.model_path
        else:
            return save_path

    @property
    def detect_type(self):
        """
        检测类型，新模型使用 merge 和 delete两种类型
        :return:
        """
        return self.ini_config.get(MODEL_CFG, MODEL_DETECT_TYPE, fallback="merge")

    @property
    def sub_model(self):
        return self.ini_config.get(MODEL_CFG, "sub_model", fallback="")

    @property
    def get_memory(self):
        memory = self.ini_config.get(MODEL_CFG, "memory", fallback=1)
        if type(memory):
            memory = float(memory)
        return memory


if __name__ == '__main__':
    from GTAI.resnet50.config import Config
    model_config = ModelConfigLoad("/data2/model/bm")
    print("black_list", model_config.black_list)
    print("子模型配置", model_config.sub_model_cfg)

    print("params", model_config.update_config_items)
    cfg = Config(**model_config.sub_model_cfg[0])
    cfg = Config(model="test", name="tttt", image_width=255)
    print(cfg)