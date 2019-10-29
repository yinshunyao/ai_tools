# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/19 0019上午 9:35
自定义bbox，定义两个bbox相等 和 merge等功能
"""
import logging
from copy import deepcopy

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
    
    def __init__(self, *args, iou_thresh=0.5, cmp_with_bbox_name=False, **kwargs):
        """
        初始化BBOX，全量参数为x1, y1, x3, y3, class_name, confidence，至少要保证有4元组
        定义案例如下
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
        self.iou_thresh = iou_thresh
        self.cmp_with_bbox_name = cmp_with_bbox_name

    def copy(self, *args, clear=False, **kwargs):
        """
        复制结构参数
        :param args:
        :param clear: 是否复制一个空的结构
        :param kwargs:
        :return:
        """
        if clear:
            return self.__class__(iou_thresh=self.iou_thresh, cmp_with_bbox_name=self.cmp_with_bbox_name)
        else:
            return deepcopy(self)

    @property
    def S(self):
        """
        求矩形框的面积S，只需要4元组即可
        :return:
        """
        # 面积0场景
        if self[2] <= self[0] or self[3] <= self[1]:
            return 0
        return (self[2] - self[0]) * (self[3] - self[1])

    def __and__(self, other):
        """
        求self 和 other矩形框交集
        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        result = self.copy(clear=False)
        x1 = max(self[0], other[0])
        y1 = max(self[1], other[1])
        x3 = min(self[2], other[2])
        y3 = min(self[3], other[3])
        # 没有交集
        if x1 >= x3 or y1 >= y3:
            # 返回空
            return self.copy(clear=True)

        # 修改矩形框
        result[0] = x1
        result[1] = y1
        result[2] = x3
        result[3] = y3
        # 后面其他属性取self一样的
        return result

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

    """bbox自定义列表"""
    def __eq__(self, other):
        """
        相等重载，iou大于门限的时候表明两个框相等
        :param other:
        :return:
        """
        if not isinstance(other, BBox):
            raise Exception("other必须是BBox类型")

        # 空不比较
        if not self or not other:
            return False

        # 需要判断名称是否相同，长度要够
        if len(other) > 4 and self.__len__() > 4 and self.cmp_with_bbox_name and other[4] != self[4]:
            return False

        # 比较iou
        if self.__truediv__(other) > self.iou_thresh:
            return True
        else:
            return False

    def __add__(self, other):
        """
        和 other 框 merge  x0, y0, x1, y1, class_name, confidence
        :param other: 另外一个框，可以是None
        :return:
        """
        if not other:
            return self

        if self.__len__() < 6 or len(other) < 6:
            raise Exception("BBox合并，左值和右值的元素个数必须大于等于6")

        if not self.__eq__(other):
            raise Exception("not equal for bbox and other, cannot add")

        result = self.copy()

        # merge两个框
        result[0] = min(self[0], other[0])
        result[1] = min(self[1], other[1])
        result[2] = max(self[2], other[2])
        result[3] = max(self[3], other[3])

        # 按照置信度高度整合类别
        if self[5] >= other[5]:
            for i in range(4, self.__len__()):
                result[i] = self[i]
        else:
            for i in range(4, other.__len__()):
                result[i] = other[i]

        return result


class BBoxes(list):
    """
    定义一组BBOX
    元素为(1000, 210, 1100, 320, "test2", 0.9, ...)
    【-】 a - b 列表a中不在b中的元素
    【|】 a | b 列表a和列表b求并集，iou小于0.5的合并
    """
    def __init__(self, *args, iou_thresh=0.5, cmp_with_bbox_name=True,  **kwargs):
        # 校验类型
        args = list(*args)
        for i in range(len(args)):
            if not isinstance(args[i], BBox):
                # logging.warning("{}不是BBox类型，尝试强制转换，可能出错".format(args[i]))
                args[i] = BBox(args[i], iou_thresh=iou_thresh, cmp_with_bbox_name=cmp_with_bbox_name)
        super(BBoxes, self).__init__(args)
        self.iou_thresh = iou_thresh
        self.cmp_with_bbox_name = cmp_with_bbox_name
        
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
            arg = BBox(arg, iou_thresh=self.iou_thresh, cmp_with_bbox_name=self.cmp_with_bbox_name)

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
            return self.__class__(iou_thresh=self.iou_thresh, cmp_with_bbox_name=self.cmp_with_bbox_name)
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
    assert  bbox1 == bbox2, "bbox1 == bbox2 without name"

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
    b1 = BBox([100, 0, 220, 220, 'test1', 0.95, 'abnormal'], cmp_with_bbox_type=True)
    b2 = BBox([100, 0, 200, 200, 'test2', 0.95, 'abnormal'], cmp_with_bbox_type=True)
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






