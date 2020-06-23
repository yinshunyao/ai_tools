# coding:utf-8
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import cv2
import numpy as np
from GTUtility.GTTools.bbox import BBoxes, BBox
from GTUtility.GTTools.img_slide import yield_sub_img
from GTUtility.GTTools.model_config import ModelConfigLoad
from GTUtility.GTTools.constant import DetectType
from copy import deepcopy
import uuid
import os


class Model:
    def __init__(self, model, model_cfg: ModelConfigLoad,
                 verify_after_main_model=None, need_sub_model_detect=None,
                 post_handle_for_result=None, **kwargs
                 ):
        self.model = model
        self.names = model_cfg.names
        self.model_cfg = model_cfg
        self.debug = model_cfg.debug
        self.edge = model_cfg.edge
        self.thread_max = model_cfg.thread_max
        # 钩子，主模型检测之后，判断一下，主模型结果是否合法
        self.verify_after_main_model = verify_after_main_model
        # 钩子，子模型检测判断，不检测以主模型结果为准
        self.need_sub_model_detect = need_sub_model_detect
        # 钩子，后处理
        self.post_handle_for_result = post_handle_for_result

    def judge_by_bbox_geo_feature(self, vertice: BBox, WIDTH, HEIGHT):
        """
        bbox判断几何参数是否合理，可以外部直接引用，不经过模型预测
        :param vertice: bbox框
        :param WIDTH:
        :param HEIGHT:
        :return: True 表示合法  False表示非法
        """
        # 几何参数判断
        # name判断
        if vertice.class_name in self.model_cfg.bbox_geo_feature_for_name.keys():
            geo_params = self.model_cfg.bbox_geo_feature_for_name[vertice.class_name]
        else:
            geo_params = self.model_cfg.bbox_geo_feature

        if geo_params and (not vertice.judge_by_geo(w=WIDTH, h=HEIGHT, **geo_params)
                           or not vertice.judge_by_edge(WIDTH, HEIGHT, **geo_params)):
            return False
        else:
            return True

    def detect_sub_img(self, box, img, thresh, WIDTH, HEIGHT, edge=False, merge_flag=False):
        """
        监测子图片img缺陷，并转换成大图坐标，构造成类方法，所有对象共用model和监测方法
        :param box:  切图的起始坐标和宽度
        :param img: 图片本身
        :param thresh: 置信度阈值门限，子图的置信度
        :param WIDTH: 整图宽度
        :param HEIGHT: 整图高度
        :param edge: 图片边缘出的框是否去除
        :param merge_flag: 子图（可能没有分片就是全图）的框是否合并
        :return: 标注bbox，参照box传入的起始x和y坐标，转换bbox成大图坐标
        """
        # resize大小 20190722于涛确认不需要resize
        # sub_img = cv2.resize(img, (self.model.config.IMAGE_MIN_DIM, self.model.config.IMAGE_MAX_DIM))
        # 子图片检查到的缺陷
        start = time.time()
        results = self.model.detect(img)
        vertices = BBoxes()
        # 缺陷数为0
        if len(results) == 0:
            return vertices

        for defect in results:
            try:
                # 如果置信度不到门限，不处理
                if defect['confidence'] < thresh:
                    continue

                class_name = defect.get("name")
                class_id = defect.get("class_id")
                if not class_name:
                    class_name = self.names[class_id]

                # 如果需要去除边缘框
                if edge and (defect['x1'] <= 0 or defect['y1'] <= 0 or defect['x2'] >= WIDTH or defect['y2'] >= HEIGHT):
                    logging.warning("缺陷{}的坐标{}靠近子图片边缘{}".format(class_name, defect, box))
                    continue

                vertice = BBox(
                    [defect['x1'] + box[0], defect['y1'] + box[1], defect['x2'] + box[0], defect['y2'] + box[1],
                     class_name, defect['confidence']])
                # 几何参数判断
                if not self.judge_by_bbox_geo_feature(vertice=vertice, WIDTH=WIDTH, HEIGHT=HEIGHT):
                    # 不满足几何参数配置，返回
                    logging.warning(
                        "[{}]缺陷{}的几何参数S:{}, w:{}, h:{}, hTow:{}不满足配置门限".format(self.model_cfg.model_type, vertice,
                                                                               vertice.S, vertice.w,
                                                                               vertice.h, vertice.hTow))
                    continue

                # append能够自动转换成自定义的BBox列表
                vertices.append(vertice)
            except Exception as e:
                # 发生异常，这一个标注框不处理，不影响其他标注框
                logging.error("解析标注框发生异常：{}".format(e), exc_info=1)

        # 一次检测结果merge
        if merge_flag:
            vertices = vertices.merge_each_other()

        # if cls.debug:
        #     logging.warning("分片{}的检测耗时{:.3f}结果：{}".format(box, time.time() - start, vertices))
        #     # 将图片保存到文件
        #     #  [1400, 1400, 2448, 2050]
        #     cv2.imwrite("_".join([str(item) for item in box]) + ".jpg", img)
        return vertices

    def detect_sub_model(self, sub_slide, **kwargs):
        """
        监测子图片img缺陷，并转换成大图坐标，构造成类方法，所有对象共用model和监测方法
        :param sub_slide: 图片本身
        :param vertice: 上一个模型检测结果
        :param WIDTH: 整图宽度
        :param HEIGHT: 整图高度
        :return: detect_type, type_name, class_confidence
        """
        # 子图片检查到的缺陷
        # detect_type, type_name, class_confidence
        return self.model.detect(sub_slide)


class predict_pipe(object):
    # model提取为类公共变量
    # model = None
    # model_cfg = None
    # names = []
    # debug = False
    # 模型字典
    model_dict = {}

    @classmethod
    def load(cls, model, model_cfg: ModelConfigLoad,
             verify_after_main_model=None,
             need_sub_model_detect=None,
             handle_for_result=None,
             **kwargs):
        """
        初始化模型的检测实体，一个进程里面最好只load一次，不要修改
        :param verify_after_main_model:  在主模型和子模型之间需要处理的函数
        :param model:  模型本身
        :param names: 对象列表
        :param model_cfg: 模型配置传递进来
        :param verify_after_main_model: 主模型校验 verify_after_main_model(obj: predict_pipe, img, bboxes)
        :param need_sub_model_detect: 传入的方法，判断是否需要子模型检测
        :param handle_for_result: 结果的后处理方法 handle_for_result(obj: predict_pipe, img, bboxes)
        :return:
        """
        uid = str(uuid.uuid1())
        cls.model_dict[uid] = Model(
            model, model_cfg,
            verify_after_main_model=verify_after_main_model,
            need_sub_model_detect=need_sub_model_detect,
            post_handle_for_result=handle_for_result,
        )
        return uid

    """
    生产环境监测图片，一个进程初始化一个实例
    检测主要是执行 process_list_merge 进行图片分析，具体参数含义参见函数定义
    """

    def __init__(self, uid, *args, **kwargs):
        """
        初始化模型的检测实体
        :param args: 预留
        :param thread_max: 分片监测的最大线程数，默认不支持并发，单线程启动
        :param kwargs: 预留  thresh_model, 分片出框的置信度
        """
        self.uuid = uid
        self.model = self.model_dict.get(self.uuid)
        if not self.model or not isinstance(self.model, Model):
            raise Exception("uuid {} 对应的模型已经不存在".format(self.uuid))

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

        # 运行参数构造
        run_params = dict(
            thresh=self.model.model_cfg.thresh_model or self.model.model_cfg.thresh_ai,
            WIDTH=WIDTH, HEIGHT=HEIGHT, edge=self.model.edge,
            merge_flag=self.model.model_cfg.merge_one_img)

        # 单线程
        excutor = None
        pool = []
        if self.model.thread_max > 1:
            excutor = ThreadPoolExecutor(max_workers=self.model.thread_max)
            # if self.debug:
            #     logging.warning("分片采用线程池{}的方式监测".format(self.thread_max))

        for box, img in yield_sub_img(
                img_path=slide, start_x=start_x, start_y=start_y,
                sub_width=patch_width, sub_height=patch_height,
                WIDTH=WIDTH, HEIGHT=HEIGHT,
                padding=padding):
            try:
                # 单线程
                if not excutor:
                    # 子图片检查到的缺陷
                    bboxes = self.model.detect_sub_img(box=box, img=img, **run_params)
                    # # 合并到大图
                    # merge方式合并
                    self.vertices = self.vertices | bboxes
                # 多线程采用线程池的方式
                else:
                    t = excutor.submit(self.model.detect_sub_img, box=box, img=img, **run_params)
                    pool.append(t)
            except Exception as e:
                logging.error("[{}]子图{}检查发生异常：{}".format(self.model.model_cfg.model_type, box, e), exc_info=1)
        else:
            pass

        for task in as_completed(pool):
            vertices = task.result()
            # merge方式合并
            self.vertices = self.vertices | vertices

        return self.vertices

    def square_padding(self, slide, width, height):
        """
        子图正方形扩充
        :param slide: 子图
        :param width:子图的宽
        :param height:子图的高
        :return:
        """
        max_length = max(width, height)
        img = np.zeros((max_length, max_length, 3), np.uint8)
        # 在左右或上下两边补零
        if height > width:
            img[0:height, round((height - width) / 2): round((height - width) / 2) + width] = slide
        else:
            img[round((width - height) / 2):height + round((width - height) / 2), 0:width] = slide
        return img

    def _classify_sub_img(self, slide, vertice: BBox, WIDTH, HEIGHT):
        """
        虚警分类消除，杆号检测子模型，仅仅主模型中调用
        :param slide:
        :param vertice:  x1, y1, x2, y2, class_name, confidence
        :return:
        """
        if self.model.model_cfg.expand_type > 0:
            sub_slide = self.square_padding(slide[vertice.y1: vertice.y2, vertice.x1:vertice.x2], vertice.w, vertice.h)
        # 如果子图需要扩充，可能配置为0
        elif self.model.model_cfg.expand_type == 0 and not self.model.model_cfg.padding_rate:
            # 扩充并处理  参数列表按照上下左右顺序
            sub_bbox = vertice.expand_by_padding(*self.model.model_cfg.padding_pixel, w=WIDTH, h=HEIGHT)
            sub_slide = slide[sub_bbox.y1: sub_bbox.y2, sub_bbox.x1:sub_bbox.x2]
        else:
            # 倍数扩充只有1倍
            sub_bbox = vertice.expand_by_rate(self.model.model_cfg.padding_rate, w=WIDTH, h=HEIGHT)
            sub_slide = slide[sub_bbox.y1: sub_bbox.y2, sub_bbox.x1:sub_bbox.x2]

        # 优先选择递归子模型检测
        if vertice.class_name and vertice.class_name in self.model.model_cfg.sub_model_dict.keys():
            # 在子模型里面merge父子模型结果
            vertice_result = self.model.model_cfg.sub_model_dict[vertice[4]].detect(image=sub_slide, vertice=vertice,
                                                                                    WIDTH=WIDTH, HEIGHT=HEIGHT)
            self.return_bboxes.extend(vertice_result)
            return

        logging.error("{}没有对应的子模型".format(vertice.class_name))
        # 没有子模型
        return

    @staticmethod
    def _merge_bbox_with_classify(vertice: BBox, detect_type, class_name, class_confidence):
        """合并框和子分类"""
        # 无效合并
        if detect_type is None or class_name is None or class_confidence is None:
            return None

        # 杆号识别
        if detect_type == DetectType.ganhao:
            # 杆号为空，则消除该框，不考虑该框
            if not class_name:
                return ""
            #
            vertice.class_confidence = class_confidence
            vertice.number = class_name
        # 分类子模型
        elif detect_type == DetectType.classify:
            # 分类模型更新
            if not not class_name:
                vertice.class_name = class_name
                vertice.class_confidence = class_confidence
        # 其他场景暂时不处理
        else:
            pass

        return vertice

    def _judge_image_vague(self, image, vague_min=12.5):
        """
        判断图像是否模糊
        :param image:
        :param vague_min:
        :return:
        """
        # 未配置，默认不配置
        if not self.model.model_cfg.vague_min:
            return True

        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image_gray=image
            image_vague = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        except Exception as e:
            logging.error("判断图像是否模糊异常：{}".format(e))
            return True

        # logging.warning("图像模糊计算值为：{}".format(image_vague))
        if image_vague < self.model.model_cfg.vague_min:
            return False
        else:
            return True

    def detect_for_image(self, sub_slide, vertice: BBox, width, height,
                         **kwargs):  # , patch_width, patch_height, padding=True):
        """
        子图检测
        :param sub_image: 图片路径
        :param vertice:  参数组，如果传递空，则不分片，一组切片参数[ (start_x, start_y, patch_width, patch_height, padding)]
        :return: detect_type, type_name, class_confidence
        """
        # TIPS:modified for mrcnn
        # self.return_bboxes = BBoxes()
        vertice_result = BBoxes()
        # 【0】如果子模型有过滤
        if not self.model.judge_by_bbox_geo_feature(vertice, width, height):
            return vertice_result

        # 【1】优先选择子模型 todo  暂时仅支持分类子模型，传入图片
        detect_type, type_name, class_confidence = self.model.detect_sub_model(sub_slide)

        # deepcopy 防止引用被修改
        vertice_temp = predict_pipe._merge_bbox_with_classify(deepcopy(vertice), detect_type, type_name,
                                                              class_confidence)
        # 父子模型结果merge失败
        if not vertice_temp:
            return vertice_result

        # 子模型类型，防止合并
        vertice_temp.class_type = "sub_model.{}".format(self.model.model_cfg.model_file)

        # 【2】子模型的中间结果是否需要输出 ;  没有子模型，返回结果
        if self.model.model_cfg.out or type_name not in self.model.model_cfg.sub_model_dict:
            # 子模型结果和vertice合并失败时，vertice_temp为空
            vertice_result.append(vertice_temp)

        # 还有子模型，递归调用
        if type_name in self.model.model_cfg.sub_model_dict:
            vertice_result_temp = self.model.model_cfg.sub_model_dict[type_name].detect(image=sub_slide,
                                                                                        vertice=deepcopy(vertice),
                                                                                        WIDTH=width, HEIGHT=height)
            # 拼接递归结果
            vertice_result.extend(vertice_result_temp)

        # 如果有后处理
        if callable(self.model.post_handle_for_result):
            vertice_result = self.model.post_handle_for_result(
                vertice_result
            )

        return vertice_result

    def process_list_merge(self, slide, patch_params, **kwargs):  # , patch_width, patch_height, padding=True):
        """
        merge方式检测图片
        :param slide: 图片
        :param patch_params:  参数组，如果传递空，则不分片，一组切片参数[ (start_x, start_y, patch_width, patch_height, padding)]
        :return:
        """
        # TIPS:modified for mrcnn
        # if self.debug:
        #     logging.warning("开始检测图片{}".format(tif_path))
        sp = slide.shape
        # 先判断图像清晰度，清晰度不够，不需要检测
        if not self._judge_image_vague(slide):
            return []
        WIDTH = sp[1]  # 图片宽度
        HEIGHT = sp[0]  # 图片高度

        vertices = BBoxes()
        if not patch_params:
            patch_params = [None, ]

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

        self.return_bboxes = BBoxes()

        # 如果自定义了过滤方法  例如杆号 保留最大方差的框，根据方差计算筛选杆号的框，待后续考虑
        if callable(self.model.verify_after_main_model):
            vertices = self.model.verify_after_main_model(self.model.model_cfg, slide, vertices)

        for vertice in vertices:
            # 类型判断
            if not isinstance(vertice, BBox):
                logging.error("bbox {} 必须为BBox类型".format(vertice))
                continue

            # 子模型置信度有效且小于模型置信度，才需要筛选  3个门限处理在配置文件中完成
            # 小于门限的不处理
            if vertice.confidence < self.model.model_cfg.thresh_ai:
                continue

            # x2, y2归一化
            vertice.x2 = min(vertice.x2, WIDTH)
            vertice.y2 = min(vertice.y2, HEIGHT)

            # ai_name 初值
            vertice.ai_name = vertice.class_name

            # 不需要运行子模型，检测结果直接返回 todo  待整合
            if not self.model.model_cfg.sub_model_dict:
                self.return_bboxes.append(vertice)
                continue

            # 判断是否需要子模型检测
            if callable(self.model.need_sub_model_detect):
                is_need, vertice = self.model.need_sub_model_detect(vertice)
                if not is_need:
                    self.return_bboxes.append(vertice)
                    continue

            # 如果有子模型，同时需要输出中间结果
            if self.model.model_cfg.out:
                self.return_bboxes.append(vertice)

            # 消除虚警，子类型检测
            try:
                self._classify_sub_img(slide, vertice=vertice, WIDTH=WIDTH, HEIGHT=HEIGHT)
            except Exception as e:
                logging.error("图片{}的{}区域子模型预测发生异常：{}".format(sp, vertice, e), exc_info=True)

        # 如果有后处理
        if callable(self.model.post_handle_for_result):
            self.return_bboxes = self.model.post_handle_for_result(
                self.model.model_cfg,
                self.return_bboxes)

        # 格式转化你返回
        results = []
        for vertice in self.return_bboxes:
            if not isinstance(vertice, BBox):
                logging.error("[{}]{}错误的bbox类型{}".format(self.model.model_cfg.model_type, vertice, type(vertice)))
                continue

            results.append(vertice.dict())

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
        pass
        self.vertices = []
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
    u = predict_pipe.load(model="t", model_cfg=ModelConfigLoad("/data4/ai_model/nest"))
    # 检测对象实例化，模型不会重新加载
    detect_obj = predict_pipe(u)
    t_dir = r"/data4/test_jpg/vague_test"
    file_list = os.listdir(t_dir)
    file_list = sorted(file_list)
    while True:
        for file_name in file_list:
            path = os.path.join(t_dir, file_name)
            # image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            start = time.time()
            image = cv2.imread(path)
            start_judge = time.time()
            print(detect_obj._judge_image_vague(image))
            end = time.time()
            print("耗时{}, 读图:{}, 检测：{}，测试图片{}".format(end - start, start_judge - start, end - start_judge, path))
        time.sleep(1)
