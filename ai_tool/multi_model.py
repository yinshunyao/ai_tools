#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 9:59
# @Author  : ysy
# @Site    : 
# @File    : multi_model.py
# @Software: PyCharm
from GTUtility.GTTools.predict_pipe import predict_pipe
import logging
from GTUtility.GTTools.model_config import ModelConfigLoad


class MultiModel:
    def __init__(self, model_cfg, model=None, sub_model_list=None, between_main_sub_func=None):
        """

        :param model_cfg:
        :param model: 主模型可以直接传递进来
        :param sub_model_list: 子模型可以直接传递进来
        :param between_main_sub_func:  在主模型和子模型之间需要处理的函数
        """
        if not isinstance(model_cfg, ModelConfigLoad):
            raise Exception("model must cfg  in type ModelConfigLoad, but type {}".format(type(model_cfg)))

        if not model:
            raise Exception("model must init first")
        self.model_cfg = model_cfg
        # 切片的置信度门限thresh_model
        self.thresh_model = model_cfg.thresh_model
        # ai存数据库的置信度门限
        self.thresh_ai = model_cfg.thresh_ai
        # 初始化模型
        predict_pipe.load(model=model, model_cfg=self.model_cfg, sub_model_list=sub_model_list, between_main_sub_func=between_main_sub_func)
        # 检测类型 merge 和 delete两种类型
        self.detect_type = model_cfg.detect_type
        # 切片宽度和高度
        self.patch_width = model_cfg.patch_width
        self.patch_height = model_cfg.patch_height
        self.padding = model_cfg.padding
        # 保存路径获取
        self.save_path = model_cfg.save_path
        # 线程数
        self.thread_max = model_cfg.thread_max
        # 去除边缘不完整器件出框的配置
        self.edge = model_cfg.edge
        logging.warning("load the model {} success".format(model_cfg.model_file))

    def IDM_detection(self, file_path, detect_type=None, patch_width=None, patch_height=None, **kwargs):
        """
        unet模型检测
        :param file_path:
        :param detect_type:
        :param patch_width:
        :param patch_height:
        :param kwargs:
        :return:
        """
        # 可以函数传参，或者使用模型默认值
        detect_type = detect_type or self.detect_type
        patch_width = patch_width or self.patch_width
        patch_height = patch_height or self.patch_height
        # logging.warning("start detect the file {}".format(file_path))
        if not patch_height or not patch_height:
            patch_params = []
        else:
            patch_params = [
                (0, 0, patch_width, patch_height, self.padding),
                (patch_width // 2, patch_height // 2, patch_width, patch_height, self.padding),
            ]

        # 监测对象每次实例化，model不会重新装载
        detect_obj = predict_pipe(thread_max=self.thread_max, edge=self.edge, thresh_model=self.thresh_model, thresh_ai=self.thresh_ai)
        if detect_type == "merge":
            vertices = detect_obj.process_list_merge(tif_path=file_path, save_path=self.save_path, patch_params=patch_params)
        elif detect_type == "delete":
            vertices = detect_obj.process_list_delete(tif_path=file_path, save_path=self.save_path, patch_params=patch_params)
        else:
            raise Exception("error detect type {} in unet".format(detect_type))

        return vertices