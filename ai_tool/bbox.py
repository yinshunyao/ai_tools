# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/19 0019上午 9:35
自定义bbox，定义两个bbox相等 和 merge等功能
"""
from copy import deepcopy

# 框合并策略
# 1、先按照iou消除冗余框
# 2、同一器件同种类别完全被包含的框消除，同一器件不同类别被包含的框保留置信度大的
# 3、不同器件被包含的框保留


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
    
    def __init__(self, *args, iou_thresh=0.5, **kwargs):
        """
        初始化BBOX，全量参数为x1, y1, x3, y3, class_name, confidence，至少要保证有4元组
        定义案例如下
        bbox2 = BBox((200, 200, 300, 300, "test1", 0.8,  "cap"))
        bbox3 = BBox((200, 200, 300, 300, "test1", 0.8))
        bbox4 = BBox((210, 210, 320, 320))
        :param args: 初始化元组参数
        :param iou_thresh: iou门限
        :param kwargs:保留参数，暂时不使用
        """
        # x1, y1, x3, y3, class_name, confidence
        # if len(*args) < 4:
        #     raise Exception("must init for x1, y1, x3, y3, but given {}".format(*args))
        super(BBox, self).__init__(*args)
        value = []
        if len(args) > 0:
            value = args[0]
        if len(value) < 6:
            self.x1 = 0
            self.x2 = 0
            self.y1 = 0
            self.y2 = 0
            self.class_name = ""
            self.confidence = 0
        else:
            self.x1 = value[0]
            self.y1 = value[1]
            self.x2 = value[2]
            self.y2 = value[3]
            # 名称
            self.class_name = value[4]
            # 置信度
            self.confidence = value[5]

        # 子模型分类置信度暂时不需要参考
        # 类型
        self.class_type = ""
        if len(value) > 6:
            self.class_type = value[6]

        # iou门限
        self.iou_thresh = iou_thresh

    @property
    def S(self):
        """
        求矩形框的面积S，只需要4元组即可
        :return:
        """
        # 面积0场景
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return 0
        return (self.x2 - self.x1) * (self.y2 - self.y1)

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
        if edge_distance_th is None or not isinstance(edge_distance_th, (int, float)):
            return True

        # 需要判断
        return self.x1 - x_start <= edge_distance_th  or w - self.x2 <= edge_distance_th \
               or self.y1 - y_start or h - self.y2 < edge_distance_th

    def judge_by_geo(self, w_range=None, h_range=None, w_to_h_range=None, s_range=None, **kwargs):
        """
        图形几何条件判断
        :param h_range:  高度范围
        :param w_range:
        :param s_range: 面积区间，闭区间
        :param w_to_h_range: 纵横比，长边与短边比例关系
        :param kwargs:
        :return:
        """
        # 面积判断
        if isinstance(s_range, (tuple, list)) and len(s_range) >= 2 and not (s_range[0] <= self.S <= s_range[1]):
            return False

        # 长和宽判断
        w = self.x2 - self.x1
        h = self.y2 - self.y1

        # 宽度范围判断
        if isinstance(w_range, (tuple, list)) and len(w_range) >= 2 and not (w_range[0] <= w <= w_range[1]):
            return False

        # 高度范围判断
        if isinstance(h_range, (tuple, list)) and len(h_range) >= 2 and not (h_range[0] <= h <= h_range[1]):
            return False

        # 纵横比判断
        if isinstance(w_to_h_range, (tuple, list)) and len(w_to_h_range) >= 2 and not (w_to_h_range[0] <= w/h <= w_to_h_range[1]):
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

    def __le__(self, other):
        """
        判断方框包含在另外一个框里面
        :param other:
        :return:
        """
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

        # 1、同种器件，不考虑类别，先按照iou消除冗余框
        if self.__truediv__(other) >= self.iou_thresh:
            # merge两个框
            return self._merge_range_max(other)

        # 2 框包含到另外一个框里面
        if self <= other or other <= self:
            # 2 （1）同一器件同种类别完全被包含的框消除
            if self.class_name == other.class_name:
                return self._merge_range_max(other)
            # 2 （2）同一器件不同类别被包含的框保留置信度大的
            else:
                return self._merge_thresh_max(other)

        raise Exception("没有定义运算规则{}-{}".format(self, other))


class BBoxes(list):
    """
    定义一组BBOX
    元素为(1000, 210, 1100, 320, "test2", 0.9, ...)
    【-】 a - b 列表a中不在b中的元素
    【|】 a | b 列表a和列表b求并集，iou小于0.5的合并
    """
    def __init__(self, *args, iou_thresh=0.5, **kwargs):
        # 校验类型
        args = list(*args)
        for i in range(len(args)):
            if not isinstance(args[i], BBox):
                # logging.warning("{}不是BBox类型，尝试强制转换，可能出错".format(args[i]))
                args[i] = BBox(args[i], iou_thresh=iou_thresh, **kwargs)
        super(BBoxes, self).__init__(args)
        self.iou_thresh = iou_thresh
        # 几何判断参数
        self.s_range = kwargs.get("s_range", None)
        self.w_to_h = kwargs.get("w_to_h", None)
        
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
        if not arg:
            raise Exception("不能添加空的BBox")

        # 如果类型不匹配，强制转换
        if not isinstance(arg, BBox):
            # logging.warning("{}不是BBox类型，尝试强制转换，可能出错".format(arg))
            arg = BBox(arg, iou_thresh=self.iou_thresh)

        # # 几何参数过滤 todo 待算法确定细节
        # if not arg.judge_by_geo(s_range=self.s_range, w_to_h=self.w_to_h):
        #     return

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
    assert BBox((1, 1, 100, 100, "1", 0.5, "type1")) == BBox((20, 20, 100, 100, "1", 0.5, "type1")), "类型相同，IOU满足条件相等"
    # 同一器件同种类别完全被包含的框消除
    t = BBox((1, 1, 100, 100, "1", 0.5, "type1")) + BBox((20, 20, 100, 100, "1", 0.5, "type1"))
    assert t.x1 == 1 and t.y1 == 1, "类型相同，IOU满足条件相等"

    assert BBox((1, 1, 100, 100, "1", 0.6, "type1")) == BBox((20, 20, 101, 101, "2", 0.5, "type1")), "类型相同，名称不同，IOU满足条件"
    # 同一器件同种类别完全被包含的框消除
    t = BBox((1, 2, 100, 100, "1", 0.6, "type1")) + BBox((20, 20, 101, 102, "2", 0.5, "type1"))
    assert t.x1 == 1 and t.y1 == 2 and t.x2 == 101 and t.y2 == 102, "类型相同，名称不同，IOU满足条件{}".format(t)

    assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((90, 90, 100, 100, "1", 0.5, "type2")), "类型不相同，包含"
    assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((20, 20, 101, 101, "1", 0.5, "type2")), "类型不相同，IoU满足，不包含"
    assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) != BBox((90, 90, 120, 120, "1", 0.5, "type1")), "类型相同，IoU不满足"

    # 同一器件同种类别完全被包含的框消除
    assert BBox((0, 0, 100, 100, "1", 0.5, "type1")) == BBox((90, 90, 100, 100, "1", 0.5, "type1")), "同一器件同种类别完全被包含的框消除"
    t = BBox((0, 0, 100, 100, "1", 0.5, "type1")) + BBox((90, 90, 100, 100, "1", 0.6, "type1"))
    print("同一器件同种类别完全被包含的框消除", t)
    assert t.x1 == 0 and t.y1 == 0 and t.confidence == 0.6, "同一器件同种类别完全被包含的框消除"

    t = BBox((0, 0, 100, 100, "1", 0.5, "type1")) + BBox((90, 90, 100, 100, "2", 0.9, "type1"))
    assert t.x1 == 90 and t.y1 == 90 and t.confidence == 0.9, "同一器件不同类别被包含的框保留置信度大的"

    t = BBox((0, 0, 100, 100, "1", 0.9, "type1")) + BBox((90, 90, 100, 100, "2", 0.5, "type1"))
    assert t.x1 == 0 and t.y1 == 0 and t.confidence == 0.9, "同一器件不同类别被包含的框保留置信度大的"

    # 定义bbox
    bbox1 = BBox((0, 0, 100, 100, "test1", 0.95))
    print("bbox1", bbox1)
    bbox2 = BBox((0, 0, 100, 120, "test2", 0.9))
    print("bbox2", bbox2)

    bbox3 = BBox((200, 200, 300, 300, "test1", 0.8))
    bbox4 = BBox((210, 210, 320, 320, "test2", 0.9))
    bbox5 = BBox((1000, 210, 1100, 320, "test2", 0.9))

    #  不同的框，不考虑类型和名称
    assert bbox1 != bbox3, "bbox1 != bbox3"

    # 比较名字不相等
    bbox1.cmp_with_bbox_name = True
    assert  bbox1 != bbox2, "bbox1 == bbox2 with name"
    # 忽略名字相等
    bbox1.cmp_with_bbox_name = False

    # iou
    print("iou for bbox1 and bbox2", bbox1 / bbox2)
    print("iou for bbox2 and bbox5", bbox2 / bbox5)
    print("iou for bbox4 and bbox5", bbox4 / bbox5)

    # 定义bbox列表
    list1 = BBoxes([bbox1, bbox3])
    list2 = BBoxes([bbox2, bbox4])
    print("list1", list1)
    print("list2", list2)
    # delete算法
    print("delete算法：list1 - list2", list1 - list2)
    print("copy for list1", list1.copy())

    # delete算法
    print("delete算法：list1 - list1", list1 - list1)

    list_empty = BBoxes()
    # delete算法
    print("delete算法：list1 - list_empty", list1 - list_empty)

    list_empty = BBoxes()
    # delete算法
    print("delete算法：list_empty - list1", list_empty - list1)

    # 添加 bbox5到list1中
    # list1.append(bbox5)
    list1.append((1000, 210, 1100, 320, "test2", 0.9))
    print("list1", list1)
    print("list2", list2)
    print("delete算法：list1 - list2", list1 - list2)
    # merge算法
    print("merge算法：list1 | list2", list1 | list2)

    print("bbox1 | bbox2", bbox1 | bbox2)

    # 缺陷匹配
    b1 = BBox([100, 0, 220, 220, 'test1', 0.95, 'abnormal'],)
    b2 = BBox([100, 0, 200, 200, 'test2', 0.95, 'abnormal'],)
    assert b1 == b2, "正确缺陷匹配"

    # 缺陷匹配
    b1 = BBox([4906, 3846, 5132, 4064, 'test1', 0.95, 'abnormal'], iou_thresh=0.8)
    b2 = BBox([4942, 3820, 5273, 4051, 'test2', 0.95, 'abnormal'], iou_thresh=0.8)
    print("b1 == b2", b1 == b2)
    inter = b1&b2
    print("b1&b2", inter, inter.S)
    print("b1.S", b1.S, "b2.S", b2.S)
    print(b1.S+b2.S-inter.S)
    print("iou for b1 and b2", b1 / b2)

    # {"x1": "754.00", "x2": "946.00", "y1": "2415.00", "y2": "2680.00", "confidence": "1.00"}
    # {"x1": "786.00", "x2": "952.00", "y1": "2408.00", "y2": "2677.00", "confidence": "1.00"}
    b1 = BBox([754,2415,946,2680,1.0, "01010301"])
    b2 = BBox([786,2408,952,2677,1.0, "01010301"])
    print(b1==b2)
    print(b1+b2)

    # x1, y1, x3, y3, name, confidence, bbox_type
    b1 = BBoxes()
    bbox1 = BBox([1026,554, 1163, 1247, '01020102', 1.0, 0])
    b1.append(bbox1)


    b2 = BBoxes()
    bbox2 = BBox([1025,708, 1158, 1227, '01020102', 1.0, 0])
    b2.append(bbox2)

    print('merge', b1 | b2)
    print(bbox1 + bbox2)
    print(bbox1 == bbox2)






