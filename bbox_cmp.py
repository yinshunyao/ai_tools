# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/26 0026下午 9:11
一系列标注框比较，评价图片分析结果
"""
from bbox import BBoxes
from constant import ImgType, BBoxType, BBoxAIDetectType, ImgAIDetectType


class ImgBBoxes(BBoxes):

    def __init__(self, *args, img_type=None, iou_thresh=0.5, cmp_with_bbox_name=False, cmp_with_bbox_type=True, **kwargs):
        """
        初始化某一个图片的bbox列表
        :param args: 支持可以 索引的 列表类似的实体
        每个arg格式  x1, y1, x3, y3, name, confidence, bbox_type
        :param img_type: 图片类型 可能是 正常没有标注  正常有标注  和 异常 三种情况
        :param iou_thresh: iou阈值
        :param cmp_with_bbox_name: 比较的时候是否考虑 bbox的name匹配
        :param cmp_with_bbox_type: 比较的时候是否考虑bbox_type匹配， 分两类，正常框和异常框
        :param kwargs:
        """
        super(ImgBBoxes, self).__init__(*args, iou_thresh=iou_thresh,
                                        cmp_with_bbox_name=cmp_with_bbox_name,
                                        cmp_with_bbox_type=cmp_with_bbox_type, **kwargs)
        self._img_type = img_type

    @property
    def img_type(self):
        """
        根据缺陷名称确定本图片属于那种类型
        可能是 正常没有标注  正常有标注  和 异常 三种情况
        :return: ImgType的属性值
        """
        # 如果图片类型已经赋值传进来，正常使用
        if self._img_type is not ImgType.unknown:
            return self._img_type

        # 没有标注框
        if self.__len__() == 0:
            self._img_type = ImgType.normal_null
        # 有任意一个缺陷框
        elif any([item[6] == BBoxType.abnormal for item in self]):
            self._img_type = ImgType.abnormal
        # 所有都是正常的标注框
        else:
            self._img_type = ImgType.normal

        return self._img_type

    def check_status(self, other):
        """
        便于如下统计需求，参照other给self 定性
        三种类型  正确检测、漏检 和 误检
        3、正确检测图像数量（仅考虑检测框位置）
             (1) 正确检测图像数量       = (2) + (3)
             (2) 正确检测正常图像数量 = 针对含有器件的正常图像，只要算法在其中任何一个器件处出检测框  +  没有器件的正常图像，不出检测框
             (3) 正确检测缺陷图像数量 = 针对缺陷图像，只要算法在其中任何一个缺陷处出检测框
        4、漏检测图像数量：针对缺陷图像，算法未在任何一处缺陷出检测框
        5、误检测图像数量：针对正常图像，算法检测出缺陷框
        :param other:  x1, y1, x3, y3, class_name
        :return:
        """
        # 类型转换
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        other_img_type = other.img_type
        # 4、漏检测图像数量：针对缺陷图像，算法未在任何一处缺陷出检测框
        if other_img_type == ImgType.abnormal:
            # 使用交集生成新的ImgBBoxes
            inter = self & other
            # 如果 交集 不是缺陷类型，则漏检
            if inter.img_type != ImgType.abnormal:
                return ImgAIDetectType.missing
            # 否则正确
            else:
                return ImgAIDetectType.correct
        # 5、误检测图像数量：针对正常图像，算法检测出缺陷框
        #  other_img_type in (ImgType.normal, ImgType.normal_null):
        else:
            # 虚警
            if self.img_type == ImgType.abnormal:
                return ImgAIDetectType.false_alarm
            else:
                pass

        # 监测正确
        return ImgAIDetectType.correct

    def compare(self, other):
        """
        与另外一个列表比较，获取比较结果，生成如下四个类型的列表
         # AI监测框 结果定义
        normal_correct = "正常正确"
        abnormal_correct = "缺陷正确"
        normal_alarm = "正常虚警"
        abnormal_alarm = "缺陷虚警"
        abnormal_missing = "缺陷漏检"
        normal_missing = "正常漏检"
        :param other:
        :return: 返回字典结果
        normal_correct: ImgBBoxes[x1, x2, x3]， abnormal_correct: ...，
        abnormal_alarm: ...， abnormal_missing: ...,
        normal_missing: ..., normal_alarm: ...
        """
        # 类型转换
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        normal_correct = self.copy()
        abnormal_correct = self.copy()
        normal_alarm = self.copy()
        abnormal_alarm = self.copy()
        abnormal_missing = self.copy()
        normal_missing = self.copy()
        # 交集中包含：算法准确检测的缺陷框, 算法准确检测的正常框
        # x1, y1, x3, y3, name, confidence, bbox_type
        for item in self & other:
            # 正常标注框
            if item[6] == BBoxType.normal:
                normal_correct.append(item)
            # 缺陷标注框
            else:
                abnormal_correct.append(item)

        # 误检测缺陷框数量
        for item in self - other:
            # 正常标注框
            if item[6] == BBoxType.normal:
                normal_alarm.append(item)
            # 缺陷标注框
            else:
                abnormal_alarm.append(item)

        # 漏检测缺陷框数量
        for item in other - self:
            # 缺陷漏检
            if item[6] == BBoxType.abnormal:
                abnormal_missing.append(item)
            # 正常漏检
            else:
                normal_missing.append(item)

        # return normal_correct, abnormal_correct, abnormal_alarm, abnormal_missing

        return {
            # 正确
            BBoxAIDetectType.normal_correct: normal_correct,
            BBoxAIDetectType.abnormal_correct: abnormal_correct,
            # 虚警
            BBoxAIDetectType.normal_alarm: normal_alarm,
            BBoxAIDetectType.abnormal_alarm: abnormal_alarm,
            # 漏检
            BBoxAIDetectType.abnormal_missing: abnormal_missing,
            BBoxAIDetectType.normal_missing: normal_missing,
        }


if __name__ == '__main__':
    img_bbox = ImgBBoxes()
    print("copy from ImgBBoxes, type is", type(img_bbox.copy()))

    # 正常正确
    img_bbox.append([0, 0, 100, 100, "test1", 0.95, BBoxType.normal])
    # 缺陷正确
    img_bbox.append([100, 0, 220, 120, "test1", 0.95, BBoxType.abnormal])
    # 缺陷虚警
    img_bbox.append([200, 0, 320, 120, "test1", 0.95, BBoxType.abnormal])
    # 正常虚警
    img_bbox.append([300, 0, 420, 120, "test1", 0.95, BBoxType.normal])

    img_bbox2 = img_bbox.copy()
    img_bbox2.append([0, 0, 100, 100, "test2", 0.95, BBoxType.normal])
    img_bbox2.append([100, 0, 200, 100, "test2", 0.95, BBoxType.abnormal])
    # 正常漏检
    img_bbox2.append([0, 100, 100, 200, "test2", 0.95, BBoxType.normal])
    # 缺陷漏检
    img_bbox2.append([0, 200, 100, 300, "test2", 0.95, BBoxType.abnormal])
    print("img_bbox", img_bbox)
    print("img_bbox2", img_bbox2)

    assert img_bbox.img_type == BBoxType.abnormal, "缺陷图片"


    # 比对测试
    print("img_bbox.compare(img_bbox2)")
    for k, v in img_bbox.compare(img_bbox2).items():
        print(k, v)