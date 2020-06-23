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


class ModelInfo:
    """模型相关信息"""

    def __init__(self, path, name, version="", md5="", *args, **kwargs):
        self.path = path
        self.name = name
        self.version = version
        self.md5 = md5

    def dict(self):
        return {"version": self.version, "md5": self.md5, "name": self.name}


class ModelConfigLoad(object):
    def __init__(self, model_path):
        self.model_path = model_path
        # 构造ini配置文件绝对路径
        self.ini_cfg_path = os.path.join(model_path, MODEL_CFG_FILE_NAME)  # /data2/model/uring/config.ini
        logging.warning("load config from file:{}".format(self.ini_cfg_path))
        self.ini_config = ConfigParser()
        self.ini_config.read(self.ini_cfg_path, encoding="utf-8")
        # 模型类型 unet或者其他
        self.model_type = self.ini_config.get(MODEL_CFG, "Type")
        # 传递给模型的是否是图片，或者文件路径，默认传递给模型图片
        self.image_only = self.ini_config.get(MODEL_CFG, "OnlyImage", fallback="true").strip() == "true"
        # model文件
        self.model_file = os.path.join(self.model_path, self.ini_config.get(MODEL_CFG, MODEL_FILE,
                                                                            fallback=""))  # /data2/model/uring/mask_rcnn_4c_defect_0275.h5
        # names 配置初始化
        self.names = {}
        if not self.ini_config.get(MODEL_CFG, MODEL_NAME, fallback=""):
            pass
        else:
            self._load_names()

        # 模型相关信息
        self.model_info = ModelInfo(
            self.model_file,
            self.ini_config.get(MODEL_CFG, MODEL_FILE, fallback=""),
            version=self.ini_config.get(MODEL_CFG, "version", fallback=""),
            md5=self.ini_config.get(MODEL_CFG, "md5", fallback=""),
        )
        logging.warning("模型信息如下：{}".format(self.model_info))
        # # 并发数 与 thread_max重复
        # self.concurrency = int(self.ini_config.get(MODEL_CFG, MODEL_CONCURRENCY, fallback=1))
        # 切片的长宽，默认值为0,表示不切片
        self.patch_width = int(self.ini_config.get(MODEL_CFG, "width", fallback=0))
        self.patch_height = int(self.ini_config.get(MODEL_CFG, "height", fallback=0))
        self.padding = (self.ini_config.get(MODEL_CFG, "padding", fallback="false")).lower() == "true"
        logging.warning("切片配置：宽-{}，高-{}，padding-{}".format(self.patch_width, self.patch_height, self.padding))
        # 调试模式
        self.debug = bool((self.ini_config.get(MODEL_CFG, "debug", fallback="false")).lower() == "true")
        # 线程数
        self.thread_max = int(self.ini_config.get(MODEL_CFG, "thread_max", fallback=1))

        # 黑名单，主要针对4C，部分图片在特定的模型不需要检测
        # 例如015946004_K292703_24_5_01.jpg 表示杆号，不需要检测管帽缺陷
        self.black_list = [item.strip() for item in self.ini_config.get(MODEL_CFG, "black_list", fallback="").split(",")
                           if not not item.strip()]
        logging.warning("模型黑名单配置：{}".format(self.black_list))

        self.white_list = [item.strip() for item in self.ini_config.get(MODEL_CFG, "white_list", fallback="").split(",")
                           if not not item.strip()]
        logging.warning("模型白名单配置：{}".format(self.white_list))

        # 配置是否去掉边缘不完整器件出的框
        self.edge = bool((self.ini_config.get(MODEL_CFG, "edge", fallback="false")).lower() == "true")
        logging.warning("模型去除边缘不完整器件框配置：{}".format(self.edge))

        # 用户配置门限全局使用
        self.thresh_user = float(self.ini_config.get(MODEL_CFG, "thresh_user", fallback=0.8))
        self.thresh_model = float(self.ini_config.get(MODEL_CFG, "thresh_model", fallback=0)) or self.thresh_user
        self.thresh_ai = float(self.ini_config.get(MODEL_CFG, "thresh_ai", fallback=0)) or self.thresh_user
        self.thresh_ai = float(self.ini_config.get(MODEL_CFG, "thresh_ai", fallback=0)) or self.thresh_user

        # self.relative_path = self.ini_config.get(MODEL_CFG, "relative_path", fallback="")
        self.train_type = self.ini_config.get(MODEL_CFG, "train_type", fallback="")
        self.model_path_str = self.ini_config.get(MODEL_CFG, "model_path_str", fallback="")
        # 子模型配置参数， 按照列表方式初始化
        self.sub_model_cfg = []
        self._load_sub_model_cfg()
        self.expand_type = int(self.ini_config.get(MODEL_CFG, "expand_type", fallback=0))
        # 进入子模型之前扩展像素，可能配置一个值，也可能配置四个值，兼容，按照上下左右的顺序配置
        self.padding_pixel = [int(item) for item in
                              self.ini_config.get(MODEL_CFG, "padding_pixel", fallback="0").split(",")]
        if len(self.padding_pixel) < 4:
            self.padding_pixel = self.padding_pixel * 4
        # 进入子模型之前扩展的像素
        self.padding_rate = float(self.ini_config.get(MODEL_CFG, "padding_rate", fallback="0"))
        # 模糊门限
        self.vague_min = float(self.ini_config.get(MODEL_CFG, "vague_min", fallback="0"))

        # 几何参数特征构造
        self.bbox_geo_feature = self._get_bbox_geo_params()
        logging.warning("几何参数配置信息{}".format(self.bbox_geo_feature))
        self.bbox_geo_feature_for_name = {}
        self._load_bbox_geo_params_dict()
        logging.warning("不同类型的几何参数配置信息{}".format(self.bbox_geo_feature_for_name))

        # centernet 网络专用参数
        self.centernet_params = self._load_centernet_params()

        # top框参数配置，例如杆号 需要返回 方差最大的 top5 号牌框，默认-1表示不筛选
        self.bbox_top = int(self.ini_config.get(MODEL_CFG, "bbox_top", fallback=-1))

        # 单图内预测merge开关，默认false
        self.merge_one_img = str(self.ini_config.get(MODEL_CFG, "merge_one_img", fallback="true")).lower() == "true"

        # 中间结果是否输出，当有分类子模型的时候，可能输出中间结果，默认不输出
        self.out = str(self.ini_config.get(MODEL_CFG, "out", fallback="false")).lower() == "true"

        # y1_thresh  杆号公里标场景下，判断y1和高比例的门限参数
        self.y1_thresh = float(self.ini_config.get(MODEL_CFG, "y1_thresh", fallback=0.67))

        # 子模型字典，key 是 主模型检测的结果ai_name，value是子模型
        self.sub_model_dict = {}

        logging.warning("配置文件读取完成")

    def _load_centernet_params(self):
        """加载centernet模型参数"""
        centernet_params = {
            'default_resolution': [512, 512],
            'mean': [.0, .0, .0],
            'std': [.0, .0, .0],
            'dataset': '',
            'num_classes': 0,
            'arch': ''        # 网络结构
        }
        if SECTION_CENTERNET not in self.ini_config.sections():
            return centernet_params

        params = dict(self.ini_config.items(SECTION_CENTERNET))
        if 'default_resolution' in params.keys():
            centernet_params['default_resolution'] = [int(item) for item in params['default_resolution'].split(",")]
        if 'mean' in params.keys():
            centernet_params['mean'] = [float(item) for item in params['mean'].split(",")]
        if 'std' in params.keys():
            centernet_params['std'] = [float(item) for item in params['std'].split(",")]
        if 'dataset' in params.keys():
            centernet_params['dataset'] = str(params['dataset'])
        if 'num_classes' in params.keys():
            centernet_params['num_classes'] = int(params['num_classes'])
        if 'arch' in params.keys():
            centernet_params['arch'] = str(params['arch'])

        return centernet_params

    def _load_names(self):
        # 加载命名文件
        with open(self.name_file, "r") as name_file:
            index = 1
            for line in name_file.readlines():
                self.names[index] = line.strip()
                index += 1
            else:
                pass

        if not self.names:
            raise Exception("mask-rcnn's name is null")

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
            # if not config_params.get(MODEL_CFG, "") or not config_params.get(MODEL_NAME, ""):
            #     logging.error("模型{}没有配置子模型{}的路径或者名称文件，将不会加载".format(self.model_type, section))
            #     continue

            # 更新为绝对路径
            config_params.update({
                MODEL_CFG: os.path.join(self.model_path, config_params.get(MODEL_CFG, "")),
                MODEL_NAME: os.path.join(self.model_path, config_params.get(MODEL_NAME, "")),
                # 如果是直接指定子模型
                "ai_type": config_params.get("ai_type", ""),
                "detect_name": config_params.get("detect_name", "")
            })
            self.sub_model_cfg.append(config_params)

    def _load_bbox_geo_params_dict(self, section_flag="bbox_"):
        """
        加载 bbo
        :param section_flag:
        :return:
        """
        for section_name in self.ini_config.sections():
            """符合条件的几何参数"""
            if section_name.startswith(section_flag):
                geo_params = self._get_bbox_geo_params(section_name)
                logging.warning("解析{}的几何参数".format(section_name))
                # 按照主模型的框的名称来筛选参数
                detect_name = geo_params.pop('detect_name', "")
                if not detect_name:
                    logging.error("{}文件的{}配置必须携带detect_name配置".format(self.ini_cfg_path, section_name))
                    continue

                # 支持列表配置
                detect_name_list = detect_name.split(",")
                for name in detect_name_list:
                    logging.warning("加载支持{}的几何参数".format(name))
                    self.bbox_geo_feature_for_name[name] = geo_params

    def _get_bbox_geo_params(self, section_name=SECTION_BBOX):
        """
        bbox几何参数配置，构造参数字典，供bbox中judge_by_geo使用
        w_th_min = 40 # 宽度最小阈值
        w_th_max = 560  # 宽度最大阈值
        h_th_min = 50 # 高度最小阈值
        h_th_max = 620 # 高度最大阈值
        hw_ratio_th_min = 0.7 # 长宽比阈值
        hw_ratio_th_max = 1.95 # 长宽比阈值
        edge_distance_th  # 到边缘的最短距离
        :return:
        """
        geo_params = {
            # 检测出框名字，例如杆号2C模型出的框name分为 ganhao 和 ganhao_v， 判断
            'detect_name': None,
            'w_range': None,
            'h_range': None,
            'w_ratio_range': None,
            'h_ratio_range': None,
            'h_to_w_range': None,
            's_range': None,
            'edge_distance_th': None,

        }
        try:
            if section_name not in self.ini_config.sections():
                return None

            params = dict(self.ini_config.items(section_name))
            for k, v in params.items():
                if isinstance(v, str):
                    params[k] = v.strip()

            # detect_name 检测名称
            if 'detect_name' in params.keys():
                geo_params['detect_name'] = params['detect_name']

            # 面积范围
            if 's_th_min' in params.keys() or 's_th_max' in params.keys():
                geo_params['s_range'] = [float(params.get('s_th_min', "-inf")), float(params.get('s_th_max', 'inf'))]

            # 宽度范围，可能只有一半区间
            if 'w_th_min' in params.keys() or 'w_th_max' in params.keys():
                geo_params['w_range'] = [float(params.get('w_th_min', "-inf")), float(params.get('w_th_max', 'inf'))]

            # 高度范围
            if 'h_th_min' in params.keys() or 'h_th_max' in params.keys():
                geo_params['h_range'] = [float(params.get('h_th_min', "-inf")), float(params.get('h_th_max', "inf"))]

            # 宽度和高宽占整体比例范围
            if 'w_ratio_th_min' in params.keys() or 'w_ratio_th_max' in params.keys():
                geo_params['w_ratio_range'] = [float(params.get('w_ratio_th_min', "-inf")), float(params.get('w_ratio_th_max', "inf"))]

            if 'h_ratio_th_min' in params.keys() or 'h_ratio_th_max' in params.keys():
                geo_params['h_ratio_range'] = [float(params.get('w_ratio_th_min', "-inf")), float(params.get('w_ratio_th_max', "inf"))]

            # 宽高比范围
            if 'hw_ratio_th_min' in params.keys() or 'hw_ratio_th_max' in params.keys():
                geo_params['h_to_w_range'] = [float(params.get('hw_ratio_th_min', "-inf")),
                                              float(params.get('hw_ratio_th_max', "inf"))]

            # 到边框最小距离判断，支持四个数字配置一样，  edge_distance_th=5
            # 或者四个数字不一样，按照上下左右的顺序配置   edge_distance_th=5,3,2,0
            if 'edge_distance_th' in params.keys():
                edge_th_list = params['edge_distance_th'].split(",")
                if len(edge_th_list) >= 4:
                    geo_params['edge_distance_th'] = tuple([int(item.strip()) for item in edge_th_list][:4])
                else:
                    # 全部转换成数组
                    geo_params['edge_distance_th'] = tuple([int(params['edge_distance_th']), ] * 4)

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
    def params(self):
        """
        参数配置
        :return:
        """
        return dict(self.update_config_items)

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

    model_config = ModelConfigLoad("/data2/model/ganhao_2c")
    print("black_list", model_config.black_list)
    print("子模型配置", model_config.sub_model_cfg)

    print("centernet配置", model_config.centernet_params)

    print("params", model_config.update_config_items)
    cfg = Config(**model_config.sub_model_cfg[0])
    cfg = Config(model="test", name="tttt", image_width=255)
    print(cfg)
