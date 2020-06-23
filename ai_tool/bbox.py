# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/19 0019上午 9:35
自定义bbox，定义两个bbox相等 和 merge等功能
"""
from copy import deepcopy
import math

# 框合并策略
# 1、先按照iou消除冗余框
# 2、同一器件同种类别完全被包含的框消除，同一器件不同类别被包含的框保留置信度大的
# 3、不同器件被包含的框保留

DEFALT_IOU_THRESH = 0.5
DEFAULT_INTER_THRESH = 0.5

class BBox(list):
    """
    BBox处理，功能介绍
    【初始化】跟普通的list初始化一样，序列顺序代表的含义 x1, y1, x3, y3, name, confidence, bbox_type
    【S】面积属性，bbox框的面积，只需要四元组参数 x1, y1, x3, y3
    【&】 a & b，两个bbox求交集，返回BBox类型的交集矩形框，只需要四元组参数 x1, y1, x3, y3
    【+或者|】 a + b， a | b两个bbox合并，需要六元组参数，x1, y1, x3, y3, class_name, confidence
    【==】 a == b 根据iou>0.5 判断两个bbox 是否相同，只需要四元组参数 x1, y1, x3, y3
    【/】 a / b 求两个bbox的iou，只需要四元组参数 x1, y1, x3, y3
    """
    
    def __init__(self, *args, iou_thresh=DEFALT_IOU_THRESH, intersection_thresh=DEFAULT_INTER_THRESH, s_thresh=1.7, **kwargs):
        """
        全量如下，class_type为分类大的类型， class_confidence为分类置信度， number为杆号号码，后面可以无限追加，含义在调用处自己约束
        x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
        定义案例如下
        bbox2 = BBox((200, 200, 300, 300, "test1", 0.8,  "cap"))
        bbox3 = BBox((200, 200, 300, 300, "test1", 0.8))
        bbox4 = BBox((210, 210, 320, 320))
        :param args: 初始化元组参数
        :param iou_thresh: iou门限
        :param intersection_thresh: 交集面积除以最小面积的比例
        :param s_thresh: iou门限
        :param kwargs:保留参数，暂时不使用
        """
        # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
        if len(args) > 0:
            value = list(args[0])
        else:
            value = [0, 0, 0, 0, "", "", "", "", "", "", "", ""]

        if len(value) < 6:
            value = [0, 0, 0, 0, "", ""]

        # 补齐 class_type
        if len(value) < 7:
            value.append("")

        # class_confidence
        if len(value) < 8:
            value.append("0")

        # 补齐 number
        if len(value) < 9:
            value.append("")

        # 补齐 ai_name
        if len(value) < 10:
            value.append("")

        # 补齐 detect_info
        if len(value) < 11:
            value.append("")

        # 补齐 model_info
        if len(value) < 12:
            value.append("")

        args = [value, ]
        super(BBox, self).__init__(*args)

        # iou门限
        self.iou_thresh = iou_thresh
        self.intersection_thresh = intersection_thresh
        self.s_thresh = s_thresh

    def dict(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            # 冗余数据，外面按照name来处理
            "name": self.class_name,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "class_confidence": self.class_confidence,
            "number": self.number,
            "ai_name": self.ai_name,
            # todo 检测信息和模型信息待补充
            "detect_info": self.detect_info,
            "model_info": self.model_info
        }

    def from_dict(self, **kwargs):
        """
        从字典更新值进来
        :param kwargs:
        :return:
        """
        for key in (
                "x1", "y1", "x2", "y2", "class_name",
                "confidence", "class_type", "class_confidence",
                "number", "ai_name"):
            if key in kwargs.keys():
                if key in ("x1", "y1", "x2", "y2", "confidence","class_confidence"):
                    self.__setattr__(key, float(kwargs.get(key) or 0))
                else:
                    self.__setattr__(key, kwargs.get(key))

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def x1(self):
        return self[0]

    @x1.setter
    def x1(self, value):
        self[0] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def y1(self):
        return self[1]

    @y1.setter
    def y1(self, value):
        self[1] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def x2(self):
        return self[2]

    @x2.setter
    def x2(self, value):
        self[2] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def y2(self):
        return self[3]

    @y2.setter
    def y2(self, value):
        self[3] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def class_name(self):
        return self[4]

    @class_name.setter
    def class_name(self, value):
        self[4] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def confidence(self):
        return self[5]

    @confidence.setter
    def confidence(self, value):
        self[5] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def class_type(self):
        return self[6]

    @class_type.setter
    def class_type(self, value):
        self[6] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def class_confidence(self):
        return self[7]

    @class_confidence.setter
    def class_confidence(self, value):
        self[7] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def number(self):
        return self[8]

    @number.setter
    def number(self, value):
        self[8] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def ai_name(self):
        return self[9]

    @ai_name.setter
    def ai_name(self, value):
        self[9] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def detect_info(self):
        return self[10]

    @detect_info.setter
    def detect_info(self, value):
        self[10] = value

    # x1, y1, x3, y3, class_name, confidence，class_type, class_confidence, number, ai_name, detect_info, model_info
    @property
    def model_info(self):
        return self[11]

    @model_info.setter
    def model_info(self, value):
        self[11] = value

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def hTow(self):
        """
        宽和高的比例
        :return:
        """
        return self.h/self.w if self.w > 0 else -1

    @property
    def S(self):
        """
        求矩形框的面积S，只需要4元组即可
        :return:
        """
        # 面积0场景
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return 0
        return self.w * self.h

    def expand_to_square(self, expand_type=0, w=0, h=0, **kwargs):
        """
        扩展为正方形
        :param expand_type: 扩展类型  1-固定左上，扩充右边和下边  2-固定中心  0-不处理
        """
        # 不处理
        if not expand_type:
            return self

        # 正方形边长
        a = max(self.w, self.h)

        # 固定左上
        if expand_type == 1:
            result = [self.x1, self.y1, self.x1 + a, self.y1 + a]
        # 固定中心点
        elif expand_type == 2:
            # 中心点
            center = ((self.x1+self.x2)/2, (self.y1+self.y2)/2)
            result = [max(0, int(center[0]-a/2)),
                      max(0, int(center[1]-a/2)),
                      min(w or math.inf, int(center[0]+a/2)),
                      min(h or math.inf, int(center[1]+a/2))]
        else:
            raise Exception("错误的配置参数，无法扩展为正方形：{}".format(expand_type))

        # 其他属性不变
        for i in range(4, self.__len__()):
            result.append(self[i])

        return BBox(result, iou_thresh=self.iou_thresh)

    def expand_by_padding(self, up, down, left, right, w=0, h=0, *args, **kwargs):
        """
        扩展倍率
        :param up: 上面扩展
        :param down: 下面扩展
        :param left: 左面扩展
        :param right: 右面扩展
        :param w: 宽极限
        :param h: 高极限
        :return:
        """
        # 新的x1, y1 ,x3, y3
        result = [
            max(0, int(self.x1 - left)), max(0, int(self.y1 - up)),
            min(w, int(self.x2 + right)), min(h, int(self.y2 + down))
        ]
        # 其他属性不变
        for i in range(4, self.__len__()):
            result.append(self[i])

        return BBox(result, iou_thresh=self.iou_thresh)

    def expand_by_rate(self, rate, w, h):
        """
        扩展倍率
        :param rate: 扩展倍率
        :param w: 宽极限
        :param h: 高极限
        :return:
        """
        # 中心点x坐标
        x_center = (self.x2 + self.x1)/2
        # 中心点y坐标
        y_center = (self.y2 + self.y1)/2
        # 左右变换后的宽度
        _w = self.w * rate / 2
        # 上下变换后的高度
        _h = self.h * rate / 2
        # 新的x1, y1 ,x3, y3
        result = [
            max(0, int(x_center-_w)), max(0, int(y_center-_h)),
            min(w, int(x_center+_w)), min(h, int(y_center+_h))
        ]
        # 其他属性不变
        for i in range(4, self.__len__()):
            result.append(self[i])

        return BBox(result, iou_thresh=self.iou_thresh)

    def judge_by_edge(self, w, h, edge_distance_th=None, x_start=0, y_start=0, **kwargs):
        """
        到边框的判断
        :param w:
        :param h:
        :param edge_distance_th: 为None的时候表明不需要判断
        :param x_start:
        :param y_start:
        :return:
        """
        # 不需要判断
        if edge_distance_th is None or not isinstance(edge_distance_th, tuple):
            return True

        # 需要判断， 按照上下左右的顺序判断
        return all([self.y1 - y_start >= edge_distance_th[0],
                    h - self.y2 >= edge_distance_th[1],
                    self.x1 - x_start >= edge_distance_th[2],
                    w - self.x2 >= edge_distance_th[3]
                    ])

    def judge_by_geo(self, w, h, w_range=None, h_range=None, h_to_w_range=None, s_range=None, **kwargs):
        """
        图形几何条件判断
        :param w:  整体宽带
        :param h:  整图高度
        :param h_range:  高度范围
        :param w_range:
        :param s_range: 面积区间，闭区间
        :param h_to_w_range: 纵横比，高除以宽
        :param kwargs:
        :return:
        """
        # 面积判断
        if isinstance(s_range, (tuple, list)) and len(s_range) >= 2 and not (s_range[0] <= self.S <= s_range[1]):
            return False

        # 宽度范围判断
        if isinstance(w_range, (tuple, list)) and len(w_range) >= 2 and not (w_range[0] <= self.w <= w_range[1]):
            return False

        # 高度范围判断
        if isinstance(h_range, (tuple, list)) and len(h_range) >= 2 and not (h_range[0] <= self.h <= h_range[1]):
            return False

        # 纵横比判断
        if isinstance(h_to_w_range, (tuple, list)) and len(h_to_w_range) >= 2 and not (h_to_w_range[0] <= self.hTow<= h_to_w_range[1]):
            return False

        # 宽度比例
        w_ratio_th_min = kwargs.get("w_ratio_th_min", None)
        if isinstance(w_ratio_th_min, (tuple, list)) and len(w_ratio_th_min) >= 2 and not (w_ratio_th_min[0] <= self.w/w <= w_ratio_th_min[1]):
            return False

        h_ratio_th_min = kwargs.get("h_ratio_th_min", None)
        if isinstance(h_ratio_th_min, (tuple, list)) and len(h_ratio_th_min) >= 2 and not (h_ratio_th_min[0] <= self.h/h <= h_ratio_th_min[1]):
            return False

        return True

    def __and__(self, other):
        """
        求self 和 other矩形框交集
        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        result = []
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        # 没有交集
        if x1 >= x2 or y1 >= y2:
            # 返回空
            return BBox(result)
        # 后面其他属性取self一样的
        return BBox([x1, y1, x2, y2, self.class_name, self.confidence], iou_thresh=self.iou_thresh)

    def __or__(self, other):
        """
        求并集
        :param other:
        :return:
        """
        return self.__add__(other)

    def __truediv__(self, other):
        """
        浮点除法，计算iou
        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        # 任意一个面积为0，返回0
        if not self.S or not other.S:
            return 0

        # 交集框
        inter = self & other
        # 没有交集
        if not inter:
            return 0
        # 交叠区域面积除以面积总和
        return inter.S / (self.S + other.S - inter.S)

    def __floordiv__(self, other):
        """
        同器件同类别框，计算面积小的框与交集面积的比例
        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        # 任意一个面积为0，返回0
        if not self.S or not other.S:
            return 0

        # 交集框
        inter = self & other
        # 没有交集
        if not inter:
            return 0
        # 交叠区域面积除以较小的面积
        # print(inter.S / min(self.S, other.S))
        return inter.S / min(self.S, other.S)

    def __le__(self, other):
        """
        判断方框包含在另外一个框里面
        :param other:
        :return:
        """
        if self.intersection_thresh is not None:
            return self.__floordiv__(other) >= self.intersection_thresh and max(self.S, other.S) / min(self.S, other.S) >= self.s_thresh

        return self.x1 >= other.x1 and self.y1 >= other.y1 and self.x2 <= other.x2 and self.y2 <= other.y2

    """bbox自定义列表"""
    def __eq__(self, other):
        """
        相等重载，iou大于门限的时候表明两个框相等
        1、同种器件，先按照iou消除冗余框
        2、同一器件同种类别完全被包含的框消除，同一器件不同类别被包含的框保留置信度大的
        3、不同器件被包含的框保留

        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        # 空不比较
        if not self or not other:
            return False

        # 器件种类不同
        if self.class_type != other.class_type:
            return False

        # 1、同种器件，不考虑类别，先按照iou消除冗余框
        if self.__truediv__(other) >= self.iou_thresh:
            return True
        # 2、同一器件同种类别完全被包含的框消除，同一器件不同类别被包含的框保留置信度大的
        if self <= other or other <= self:
            return True

        return False

    def _merge_range_max(self, other):
        """
        合并为区域最大
        :param other:
        :return:
        """
        result = []
        # merge两个框
        result.append(min(self.x1, other.x1))
        result.append(min(self.y1, other.y1))
        result.append(max(self.x2, other.x2))
        result.append(max(self.y2, other.y2))

        # 按照置信度高度整合类别
        if self[5] >= other[5]:
            for i in range(4, self.__len__()):
                result.append(self[i])
        else:
            for i in range(4, other.__len__()):
                result.append(other[i])

        return BBox(result, iou_thresh=self.iou_thresh)

    def _merge_thresh_max(self, other):
        """
        合并为置信度最大
        :param other:
        :return:
        """
        # merge两个框
        # 按照置信度高度整合类别
        if self[5] >= other[5]:
            return deepcopy(self)
        else:
            return deepcopy(other)

    def __add__(self, other):
        """
        和 other 框 merge  x0, y0, x1, y1, class_name, confidence, class_type
        :param other: 另外一个框，可以是None
        :return:
        """
        if not other:
            return self

        if not self.__eq__(other):
            raise Exception("not equal for bbox and other, cannot add")

        # 先判断框是不是包含内部，再判断IOU
        # 1 框包含到另外一个框里面
        if self <= other or other <= self:
            return self._merge_thresh_max(other)
            # # 2 （1）同一器件同种类别完全被包含的框消除
            # if self.class_name == other.class_name:
            #     return self._merge_range_max(other)
            # # 2 （2）同一器件不同类别被包含的框保留置信度大的
            # else:
            #     return self._merge_thresh_max(other)

        # 2、同种器件，不考虑类别，先按照iou消除冗余框
        if self.__truediv__(other) >= self.iou_thresh:
            # merge两个框
            return self._merge_range_max(other)

        raise Exception("没有定义运算规则{}-{}".format(self, other))


class BBoxes(list):
    """
    定义一组BBOX
    元素为(1000, 210, 1100, 320, "test2", 0.9, ...)
    【-】 a - b 列表a中不在b中的元素
    【|】 a | b 列表a和列表b求并集，iou小于0.5的合并
    """
    def __init__(self, *args, iou_thresh=DEFALT_IOU_THRESH, intersection_thresh=DEFAULT_INTER_THRESH, s_thresh=1.7, **kwargs):
        # 校验类型
        args = list(*args)
        for i in range(len(args)):
            if not isinstance(args[i], BBox):
                # logging.warning("{}不是BBox类型，尝试强制转换，可能出错".format(args[i]))
                args[i] = BBox(args[i], iou_thresh=iou_thresh, intersection_thresh=intersection_thresh, s_thresh=s_thresh, **kwargs)
        super(BBoxes, self).__init__(args)
        self.iou_thresh = iou_thresh
        self.intersection_thresh = intersection_thresh
        self.s_thresh = s_thresh
        # 已经merge过的列表
        self._has_merged_bboxes = []

    def sorted(self, key=None, reverse=True):
        """
        排序，默认按照置信度排序,从高往低
        :param key:
        :param reverse:
        :return:
        """
        key = key or (lambda item: item.confidence)
        return BBoxes(sorted(self, key=key, reverse=reverse))

    def merge_each_other(self):
        # 先按照置信度排序
        sorted_bboxes = self.sorted(reverse=True)
        bboxes = BBoxes()
        has_merged_bboxes = []
        for index_main, bbox_main in enumerate(sorted_bboxes):
            # 已经被merge过的框不再添加
            if bbox_main in has_merged_bboxes:
                continue

            # 遍历之后的bbox，相同的则merge
            for index, bbox in enumerate(sorted_bboxes[index_main+1:]):
                if bbox_main == bbox:
                    bbox_main = bbox_main + bbox
                    has_merged_bboxes.append(bbox)

            # 合并完之后添加
            bboxes.append(bbox_main)

        return bboxes

    def append_dict(self, **kwargs):
        """
        从字典导入bbox
        :param kwargs:
        :return:
        """
        bbox = BBox()
        bbox.from_dict(**kwargs)
        self.append(bbox)

    def __add__(self, other):
        return self.extend(other)

    def extend(self, other):
        """扩展"""
        # 如果other不符合预期
        if not other or not isinstance(other, BBoxes) or len(other) == 0:
            return self

        for bbox in other:
            self.append(bbox)

        return self
        
    def append(self, *args, **kwargs):
        """
        bbox的列表中添加一个BBox对象，也可以直接传六元组转换
        # 直接添加6元组
        list1.append((1000, 210, 1100, 320, "test2", 0.9))

        添加BBOx
        bbox5 = BBox((1000, 210, 1100, 320, "test2", 0.9))
        list1.append(bbox5)
        :param args:
        :param kwargs:
        :return:
        """
        # 支持第一个参数解析
        arg = args[0]
        # 不需要处理
        if not arg:
            return

        # 如果类型不匹配，强制转换
        if not isinstance(arg, BBox):
            # logging.warning("{}不是BBox类型，尝试强制转换，可能出错".format(arg))
            arg = BBox(arg, iou_thresh=self.iou_thresh, intersection_thresh=self.intersection_thresh, s_thresh=self.s_thresh)

        super(BBoxes, self).append(arg)

    def copy(self, *args, clear=False, **kwargs):
        """
        复制结构参数
        :param args:
        :param clear:  是否复制一个空的结构，一般复制结构
        :param kwargs:
        :return:
        """
        if clear:
            return self.__class__(iou_thresh=self.iou_thresh, **kwargs)
        else:
            return deepcopy(self)

    def __sub__(self, other):
        """
        当前self中不存在other中的元素，集合一样求差集
        :param other:
        :return:
        """
        # 类型转换
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        result = []
        # 如果 other为空
        if not other:
            return self.copy()

        # 如果self为空，返回对象
        # if not self:
        #     return []

        for item in self:
            if item not in other:
                result.append(item)

        return result

    def __and__(self, other):
        """
        当前self中也存在other中的元素，集合一样求交集
        :param other:
        :return:
        """
        # 类型转换
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        result = self.copy(clear=True)
        # 如果 other为空
        if not other:
            return result

        for item in self:
            if item in other:
                result.append(item)
        return result

    def __or__(self, other):
        """
        像集合一样求并集
        :param other:
        :return:
        """
        # 类型转换
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        result = self.copy(clear=True)
        for item in self:
            for other_item in other:
                # 如果是相同的框，合并
                if item == other_item:
                    result.append(item+other_item)
                    # 移除相同项
                    other.remove(other_item)
                    break
                else:
                    continue
            else:
                # 在other中没有找到相同的框，添加到结果集中
                result.append(item)

        # 合并 other中遗留的元素
        result.extend(other)
        return result


if __name__ == '__main__':
    # BBox逻辑判断
    # assert BBox((1, 1, 100, 100, "1", 0.5, "type1")) == BBox((20, 20, 100, 100, "1", 0.5, "type1")), "类型相同，IOU满足条件相等"
    # # 同一器件同种类别完全被包含的框消除
    # t = BBox((1, 1, 100, 100, "1", 0.5, "type1")) + BBox((20, 20, 100, 100, "1", 0.5, "type1"))
    # assert t.x1 == 1 and t.y1 == 1, "类型相同，IOU满足条件相等"
    #
    # assert BBox((1, 1, 100, 100, "1", 0.6, "type1")) == BBox((20, 20, 101, 101, "2", 0.5, "type1")), "类型相同，名称不同，IOU满足条件"
    # # 同一器件同种类别完全被包含的框消除
    # t = BBox((1, 2, 100, 100, "1", 0.6, "type1")) + BBox((20, 20, 101, 102, "2", 0.5, "type1"))
    # assert t.x1 == 1 and t.y1 == 2 and t.x2 == 101 and t.y2 == 102, "类型相同，名称不同，IOU满足条件{}".format(t)
    #
    # assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((90, 90, 100, 100, "1", 0.5, "type2")), "类型不相同，包含"
    # assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((20, 20, 101, 101, "1", 0.5, "type2")), "类型不相同，IoU满足，不包含"
    # assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((90, 90, 120, 120, "1", 0.5, "type1")), "类型相同，IoU不满足"
    #
    # # 同一器件同种类别完全被包含的框消除
    # assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) == BBox((90, 90, 100, 100, "1", 0.5, "type1")), "同一器件同种类别完全被包含的框消除"
    # t = BBox((0, 0, 100, 100, "1", 0.5, "type1")) + BBox((90, 90, 100, 100, "1", 0.6, "type1"))
    # print("同一器件同种类别完全被包含的框消除", t)
    # assert t.x1 == 0 and t.y1 == 0 and t.confidence == 0.6, "同一器件同种类别完全被包含的框消除"
    #
    # t = BBox((0, 0, 100, 100, "1", 0.5, "type1")) + BBox((90, 90, 100, 100, "2", 0.9, "type1"))
    # assert t.x1 == 90 and t.y1 == 90 and t.confidence == 0.9, "同一器件不同类别被包含的框保留置信度大的"
    #
    # t = BBox((0, 0, 100, 100, "1", 0.9, "type1")) + BBox((90, 90, 100, 100, "2", 0.5, "type1"))
    # assert t.x1 == 0 and t.y1 == 0 and t.confidence == 0.9, "同一器件不同类别被包含的框保留置信度大的"
    #
    # # 定义bbox
    # bbox1 = BBox((0, 0, 100, 100, "test1", 0.95))
    # print("bbox1", bbox1)
    # bbox2 = BBox((0, 0, 100, 120, "test2", 0.9))
    # print("bbox2", bbox2)
    #
    # bbox3 = BBox((200, 200, 300, 300, "test1", 0.8))
    # bbox4 = BBox((210, 210, 320, 320, "test2", 0.9))
    # bbox5 = BBox((1000, 210, 1100, 320, "test2", 0.9))
    #
    # #  不同的框，不考虑类型和名称
    # assert bbox1 != bbox3, "bbox1 != bbox3"
    #
    # # 比较名字不相等
    # bbox1.cmp_with_bbox_name = True
    # assert  bbox1 != bbox2, "bbox1 == bbox2 with name"
    # # 忽略名字相等
    # bbox1.cmp_with_bbox_name = False
    #
    # # iou
    # print("iou for bbox1 and bbox2", bbox1 / bbox2)
    # print("iou for bbox2 and bbox5", bbox2 / bbox5)
    # print("iou for bbox4 and bbox5", bbox4 / bbox5)
    #
    # # 定义bbox列表
    # list1 = BBoxes([bbox1, bbox3])
    # list2 = BBoxes([bbox2, bbox4])
    # print("list1", list1)
    # print("list2", list2)
    # # delete算法
    # print("delete算法：list1 - list2", list1 - list2)
    # print("copy for list1", list1.copy())
    #
    # # delete算法
    # print("delete算法：list1 - list1", list1 - list1)
    #
    # list_empty = BBoxes()
    # # delete算法
    # print("delete算法：list1 - list_empty", list1 - list_empty)
    #
    # list_empty = BBoxes()
    # # delete算法
    # print("delete算法：list_empty - list1", list_empty - list1)
    #
    # # 添加 bbox5到list1中
    # # list1.append(bbox5)
    # list1.append((1000, 210, 1100, 320, "test2", 0.9))
    # print("list1", list1)
    # print("list2", list2)
    # print("delete算法：list1 - list2", list1 - list2)
    # # merge算法
    # print("merge算法：list1 | list2", list1 | list2)
    #
    # print("bbox1 | bbox2", bbox1 | bbox2)
    #
    # # 缺陷匹配
    # b1 = BBox([100, 0, 220, 220, 'test1', 0.95, 'abnormal'],)
    # b2 = BBox([100, 0, 200, 200, 'test2', 0.95, 'abnormal'],)
    # assert b1 == b2, "正确缺陷匹配"
    #
    # # 缺陷匹配
    # b1 = BBox([4906, 3846, 5132, 4064, 'test1', 0.95, 'abnormal'], iou_thresh=0.8)
    # b2 = BBox([4942, 3820, 5273, 4051, 'test2', 0.95, 'abnormal'], iou_thresh=0.8)
    # print("b1 == b2", b1 == b2)
    # inter = b1&b2
    # print("b1&b2", inter, inter.S)
    # print("b1.S", b1.S, "b2.S", b2.S)
    # print(b1.S+b2.S-inter.S)
    # print("iou for b1 and b2", b1 / b2)
    #
    # # {"x1": "754.00", "x2": "946.00", "y1": "2415.00", "y2": "2680.00", "confidence": "1.00"}
    # # {"x1": "786.00", "x2": "952.00", "y1": "2408.00", "y2": "2677.00", "confidence": "1.00"}
    # b1 = BBox([754,2415,946,2680,1.0, "01010301"])
    # b2 = BBox([786,2408,952,2677,1.0, "01010301"])
    # print(b1==b2)
    # print(b1+b2)
    #
    # # x1, y1, x3, y3, name, confidence, bbox_type
    # b1 = BBoxes()
    # bbox1 = BBox([1026,554, 1163, 1247, '01020102', 1.0, 0])
    # b1.append(bbox1)
    #
    #
    # b2 = BBoxes()
    # bbox2 = BBox([1025,708, 1158, 1227, '01020102', 1.0, 0])
    # b2.append(bbox2)
    #
    # print('merge', b1 | b2)
    # print(bbox1 + bbox2)
    # print(bbox1 == bbox2)


    vertices = BBoxes()
    b1 = BBoxes([[3233, 3480, 3474, 3812, 'pemissing1', 0.59], [3233, 3480, 3474, 3812, 'penormal1', 0.47]])
    # b1.append([875, 692, 1316, 1118, 'capmissing', 0.66])
    vertices = vertices | b1
    # b2 = BBoxes([[3214, 3478, 3467, 3488, 'penormal1', 0.31]])
    # vertices = vertices | b2
    b3 = BBoxes([[3214, 3478, 3467, 3488, 'penormal1', 0.31], [3236, 3500, 3471, 3811, 'pemissing1', 0.45], [3236, 3500, 3471, 3811, 'penormal1', 0.61]])
    vertices = vertices | b3
    print("merge", vertices)

    print(b3 | b1)

    # b = BBox([2158, 2825, 2233, 2892, 'capnormal', 0.9660947])
    # print("bbox", b)
    # print("几何参数judge_by_geo比对", b.judge_by_geo(w_range=(50, 280), h_range=(50, 280), h_to_w_range=(0.6, 1.35)))
    # print("几何参数judge_by_edge比对", b.judge_by_edge(w=4920, h=3280, edge_distance_th=15))


    b1 = BBoxes( [[3233, 3480, 3474, 3812, 'pemissing1', 0.59], [3233, 3480, 3474, 3812, 'penormal1', 0.47]])
    b2 = BBoxes([[3214, 3478, 3467, 3488, 'penormal1', 0.31], [3236, 3500, 3471, 3811, 'pemissing1', 0.45], [3236, 3500, 3471, 3811, 'penormal1', 0.61]])

    print("b1 merge", b1.merge_each_other())
    print("b2 merge", b2.merge_each_other())

    print(b1 | b2)
    print(b2 | b1)

    print('bbox merge', BBox([3236, 3500, 3471, 3811, 'penormal1', 0.61]) | BBox([3233, 3480, 3474, 3812, 'pemissing1', 0.59]))

    print('sus merge',
          BBox([156, 2690, 689, 3321, '01010601', 0.78]) | BBox([155.00, 2621.00, 694.00, 3316.00, '0307', 0.68]))

    # print('sus floordiv',
    #       BBox([1352, 1072, 1407, 1621, 'susnormal', 0.39]) // BBox([1352, 1399, 1389, 1623, 'susnormal', 0.15]))

    # 放大缩小
    a=BBox([100, 200, 300, 400, 'penormal1', 0.61])
    b = a.expand_by_rate(1.2, 4000, 4000)
    print("{}变成了{}".format(a.dict(), b.dict()))

    a = BBox([2361, 3923, 2440, 3979, 'penormal1', 0.61])
    b = a.expand_by_rate(1.2, 4000, 4000)
    print("{}变成了{}".format(a.dict(), b.dict()))





