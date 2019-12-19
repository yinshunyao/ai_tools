#coding:utf-8
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import cv2
import numpy as np
from GTUtility.GTTools.bbox import BBoxes, BBox
from GTUtility.GTTools.img_slide import yield_sub_img
from GTUtility.GTTools.model_config import ModelConfigLoad


class predict_pipe(object):
    # model提取为类公共变量
    model = None
    model_cfg = None
    names = []
    debug = False
    # 子模型列表
    sub_model_list = []

    @classmethod
    def load(cls, model, model_cfg: ModelConfigLoad,
             sub_model_dict=None, sub_model_list=None,
             verify_after_main_model=None,
             need_sub_model_detect=None,
             handle_for_result=None,
             **kwargs):
        """
        初始化模型的检测实体，一个进程里面最好只load一次，不要修改
        :param verify_after_main_model:  在主模型和子模型之间需要处理的函数
        :param model:  模型本身
        :param names: 对象列表
        :param sub_model_dict: 子模型字典，key为主模型输出的name，对应需要检测
        :param sub_model_list: 对象列表，如果没有配置key，则尝试所有子模型
        :param model_cfg: 模型配置传递进来
        :param verify_after_main_model: 主模型校验 verify_after_main_model(obj: predict_pipe, img, bboxes)
        :param need_sub_model_detect: 传入的方法，判断是否需要子模型检测
        :param handle_for_result: 结果的后处理方法 handle_for_result(obj: predict_pipe, img, bboxes)
        :return:
        """
        cls.model = model
        cls.names = model_cfg.names
        cls.sub_model_dict = sub_model_dict or {}
        cls.sub_model_list = sub_model_list or []
        cls.model_cfg = model_cfg
        cls.debug = model_cfg.debug
        cls.edge = model_cfg.edge
        cls.thread_max = model_cfg.thread_max
        # 钩子，主模型检测之后，判断一下，主模型结果是否合法
        cls.verify_after_main_model = verify_after_main_model
        # 钩子，子模型检测判断，不检测以主模型结果为准
        cls.need_sub_model_detect = need_sub_model_detect
        # 钩子，后处理
        cls.post_handle_for_result = handle_for_result

    @classmethod
    def _judge_by_bbox_geo_feature(cls, vertice: BBox, WIDTH, HEIGHT):
        """
        bbox判断几何参数是否合理
        :param vertice: bbox框
        :param WIDTH:
        :param HEIGHT:
        :return: True 表示合法  False表示非法
        """
        # 几何参数判断
        # name判断
        if vertice.class_name in cls.model_cfg.bbox_geo_feature_for_name.keys():
            geo_params = cls.model_cfg.bbox_geo_feature_for_name[vertice.class_name]
        else:
            geo_params = cls.model_cfg.bbox_geo_feature

        if geo_params and (not vertice.judge_by_geo(**geo_params)
                              or not vertice.judge_by_edge(WIDTH, HEIGHT, **geo_params)):
            return False
        else:
            return True

    @classmethod
    def _detect_sub_img(cls, box, img, thresh, WIDTH, HEIGHT, edge=False):
        """
        监测子图片img缺陷，并转换成大图坐标，构造成类方法，所有对象共用model和监测方法
        :param box:  切图的起始坐标和宽度
        :param img: 图片本身
        :param thresh: 置信度阈值门限，子图的置信度
        :param WIDTH: 整图宽度
        :param HEIGHT: 整图高度
        :param edge: 图片边缘出的框是否去除
        :return: 标注bbox，参照box传入的起始x和y坐标，转换bbox成大图坐标
        """
        # resize大小 20190722于涛确认不需要resize
        # sub_img = cv2.resize(img, (self.model.config.IMAGE_MIN_DIM, self.model.config.IMAGE_MAX_DIM))
        # 子图片检查到的缺陷
        start = time.time()
        results = cls.model.detect(img)
        vertices = BBoxes()
        # 缺陷数为0
        if len(results) == 0:
            return vertices

        for defect in results:
            try:
                # 如果置信度不到门限，不处理
                if defect['confidence'] <= thresh:
                    continue

                class_name = defect.get("name")
                class_id = defect.get("class_id")
                if not class_name:
                    class_name = cls.names[class_id]

                # 如果需要去除边缘框
                if edge and (defect['x1'] <= 0 or defect['y1'] <= 0 or defect['x2'] >= WIDTH or defect['y2'] >=HEIGHT):
                    logging.warning("缺陷{}的坐标{}靠近子图片边缘{}".format(class_name, defect, box))
                    continue

                vertice = BBox([defect['x1'] + box[0], defect['y1'] + box[1], defect['x2'] + box[0], defect['y2'] + box[1], class_name, defect['confidence']])
                # 几何参数判断
                if not cls._judge_by_bbox_geo_feature(vertice=vertice, WIDTH=WIDTH, HEIGHT=HEIGHT):
                    # 不满足几何参数配置，返回
                    logging.warning(
                        "[{}]缺陷{}的几何参数w:{}, h:{} , hTow:{}不满足配置门限".format(cls.model_cfg.model_type, vertice, vertice.w,
                                                                  vertice.h, vertice.hTow))
                    continue

                # append能够自动转换成自定义的BBox列表
                vertices.append(vertice)
                # 合并
                # boxes = BBoxes([[x0, y0, x1, y1, class_name, confidence], ])
                # vertices = vertices | boxes
            except Exception as e:
                # 发生异常，这一个标注框不处理，不影响其他标注框
                logging.error("解析标注框发生异常：{}".format(e), exc_info=1)

        if cls.debug:
            logging.warning("分片{}的检测耗时{:.3f}结果：{}".format(box, time.time() - start, vertices))
            # 将图片保存到文件
            #  [1400, 1400, 2448, 2050]
            cv2.imwrite("_".join([str(item) for item in box]) + ".jpg", img)
        return vertices

    """
    生产环境监测图片，一个进程初始化一个实例
    检测主要是执行 process_list_merge 进行图片分析，具体参数含义参见函数定义
    """
    def __init__(self, *args, **kwargs):
        """
        初始化模型的检测实体
        :param args: 预留
        :param thread_max: 分片监测的最大线程数，默认不支持并发，单线程启动
        :param kwargs: 预留  thresh_model, 分片出框的置信度
        """
        # 并发处理，结果回填需要加锁
        self.vertices = BBoxes()
        self._lock = RLock()

    def _detect_img(self, slide, start_x=0, start_y=0, patch_width=0, patch_height=0, padding=True):
        """
        按照一组分片参数检测图片，分片，每片图片检测
        :param slide: 图片 或者 图片路径
        :param start_x: 起始x坐标
        :param start_y: 起始y坐标
        :param patch_width: 分片宽度 0时表示不切片
        :param patch_height: 分片高度
        :param padding:
        :return:
        """
        # 清空缓存
        self.vertices = BBoxes()

        sp = slide.shape
        WIDTH = sp[1]
        HEIGHT = sp[0]

        if self.debug:
            logging.warning("图片宽{}，高{}".format(WIDTH, HEIGHT))

        # 单线程
        excutor = None
        pool = []
        if self.thread_max > 1:
            excutor = ThreadPoolExecutor(max_workers=self.thread_max)
            if self.debug:
                logging.warning("分片采用线程池{}的方式监测".format(self.thread_max))

        for box, img in yield_sub_img(
                img_path=slide, start_x=start_x, start_y=start_y,
                sub_width=patch_width, sub_height=patch_height,
                WIDTH=WIDTH, HEIGHT=HEIGHT,
                padding=padding):
            try:
                # 单线程
                if not excutor:
                    # 子图片检查到的缺陷
                    bboxes = self._detect_sub_img(
                        box=box, img=img, thresh=self.model_cfg.thresh_model or self.model_cfg.thresh_ai,
                        WIDTH=WIDTH, HEIGHT=HEIGHT, edge=self.edge
                    )
                    # # 合并到大图
                    # self.vertices.extend(bboxes)
                    # merge方式合并
                    self.vertices = self.vertices | bboxes
                # 多线程采用线程池的方式
                else:
                    t = excutor.submit(
                        self._detect_sub_img, box=box, img=img, thresh=self.model_cfg.thresh_model,
                        WIDTH=WIDTH, HEIGHT=HEIGHT, edge=self.edge)
                    pool.append(t)
            except Exception as e:
                logging.error("子图{}检查发生异常：{}".format(box, e), exc_info=1)
        else:
            pass

        for task in as_completed(pool):
            vertices = task.result()
            # merge方式合并
            self.vertices = self.vertices | vertices
            # self.vertices.extend(vertices)
            # 合并结果
            # self._collection_vertices(vertices)

        return self.vertices

    def _classify_sub_img(self, slide, vertice, WIDTH, HEIGHT):
        """
        虚警分类消除，杆号检测子模型
        :param slide:
        :param vertice:  x1, y1, x2, y2, class_name, confidence
        :return:
        """
        # 如果子图需要扩充，可能配置为0
        # if self.model_cfg.padding_pixel:
        vertice[0] = max(vertice[0] - self.model_cfg.padding_pixel[0], 0)
        vertice[1] = max(vertice[1] - self.model_cfg.padding_pixel[1], 0)
        vertice[2] = min(vertice[2] + self.model_cfg.padding_pixel[2], WIDTH)
        vertice[3] = min(vertice[3] + self.model_cfg.padding_pixel[3], HEIGHT)


        x0, y0, x1, y1 = vertice[0:4]
        sub_slide = slide[y0: y1, x0:x1]

        # 如果有指定类型检测的模型
        if vertice[4] and vertice[4] in self.sub_model_dict.keys():
            type_name, class_confidence = self.sub_model_dict[vertice[4]].detect(sub_slide)
            return type_name, class_confidence

        # 没有指定子模型，则尝试处理子模型处理
        for sub_model in self.sub_model_list:
            # class_confidence 约定大于100的场景下，表示杆号识别
            type_name, class_confidence = sub_model.detect(sub_slide)
            # 传递None表明未检测，不需要处理
            if type_name is None:
                continue
            return type_name, class_confidence

        # 没有子模型
        return None, None

    def process_list_merge(self, tif_path, patch_params, **kwargs):  # , patch_width, patch_height, padding=True):
        """
        merge方式检测图片
        :param tif_path: 图片路径
        :param save_path: 保存路径，不使用
        :param patch_params:  参数组，如果传递空，则不分片，一组切片参数[ (start_x, start_y, patch_width, patch_height, padding)]
        :return:
        """
        # TIPS:modified for mrcnn
        if self.debug:
            logging.warning("开始检测图片{}".format(tif_path))
        slide = cv2.imread(tif_path)

        sp = slide.shape
        WIDTH = sp[1]  # 图片宽度
        HEIGHT = sp[0]  # 图片高度

        vertices = BBoxes()
        if not patch_params:
            patch_params = [None,]

        for patch_param in patch_params:
            # 不需要切片
            if not patch_param:
                vertices_temp = self._detect_img(slide)
            # 需要切片
            else:
                vertices_temp = self._detect_img(slide, *patch_param)
            # todo 超过2组参数场景后续需要确认merge算法
            vertices = vertices | vertices_temp
        else:
            pass

        results = []

        # 如果自定义了过滤方法  例如杆号 保留最大方差的框，根据方差计算筛选杆号的框，待后续考虑
        if callable(self.verify_after_main_model):
            vertices = self.verify_after_main_model(slide, vertices)

        for vertice in vertices:
            # 类型判断
            if not isinstance(vertice, BBox):
                logging.error("bbox {} 必须为BBox类型".format(vertice))
                continue

            # 子模型置信度有效且小于模型置信度，才需要筛选  3个门限处理在配置文件中完成
            # 小于门限的不处理
            if vertice[5] < self.model_cfg.thresh_ai:
                continue

            # x2, y2归一化
            vertice[2] = min(vertice[2], WIDTH)
            vertice[3] = min(vertice[3], HEIGHT)

            # x1, y1, x2, y2, class_name, confidence 转换成字典
            defect = {
                "x1": vertice[0],
                "y1": vertice[1],
                "x2": vertice[2],
                "y2": vertice[3],
                "name": vertice[4],
                "confidence": vertice[5],
            }

            # 不需要运行子模型，检测结果直接返回
            if not self.sub_model_dict and not self.sub_model_list or \
                            callable(self.need_sub_model_detect) and not self.need_sub_model_detect(vertice):
                results.append(defect)
                continue

            # 消除虚警，子类型检测
            try:
                class_name, class_confidence = self._classify_sub_img(slide, vertice=vertice, WIDTH=WIDTH, HEIGHT=HEIGHT)
                # class_confidence 约定大于99的场景下，表示杆号识别
                if class_confidence and int(class_confidence) > 99:
                    defect['number'] = class_name
                # 可能没有子模型
                elif not class_name:
                    # todo 暂时存储ai名称到number字段中
                    defect['number'] = vertice[4]
                    # logging.error("图片{}的{}区域子模型预测结果为空".format(tif_path, vertice))
                else:
                    # 更新子模型分类，添加分类模型置信度
                    defect['name'] = class_name
                    # todo 暂时存储ai名称到number字段中
                    defect['number'] = vertice[4]
                    defect['class_confidence'] = class_confidence
                    # vertice[4] = class_name
                    # vertice.append(class_confidence)
            except Exception as e:
                logging.error("图片{}的{}区域子模型预测发生异常：{}".format(tif_path, vertice, e), exc_info=1)

            results.append(defect)

        # 如果有后处理
        if callable(self.post_handle_for_result):
            results = self.post_handle_for_result(results)

        return results

    # todo delete方式暂时不支持
    def process_list_delete(self, tif_path, patch_params, **kwargs):
        """
        delete方式检测图片
        :param tif_path: 图片路径
        :param save_path: 保存路径，不使用
        :param patch_params:  参数组，一组切片参数[ (start_x, start_y, patch_width, patch_height, padding)]
        :return:
        """
        self.vertices = []
        if self.debug:
            logging.warning("开始检测图片{}".format(tif_path))
        slide = cv2.imread(tif_path)
        vertices_first = BBoxes()
        vertices = BBoxes()
        for patch_param in patch_params:
            vertices_temp = self._detect_img(slide, *patch_param)
            # todo 超过2组参数场景后续需要确认delete算法
            if not vertices_first:
                vertices_first = vertices_temp
            vertices = vertices_first - vertices_temp
        else:
            return vertices


if __name__ == '__main__':
    # 使用demo
    # 初始化模型
    predict_pipe.load(model="模型对象", names="name列表", sub_model_list="子模型列表", debug=False)
    # 检测对象实例化，模型不会重新加载
    detect_obj = predict_pipe(thread_max=1, edge=True, thresh_model=0.90,thresh_ai=0.95)
    # 分片参数
    patch_params = [
        (0, 0, 1400, 1400, True),
        (700, 700, 1400, 1400, True),
    ]
    # merge方式预测
    vertices = detect_obj.process_list_merge(tif_path="图片路径", save_path="", patch_params=patch_params)
