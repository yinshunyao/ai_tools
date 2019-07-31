# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/31 0031下午 9:33
test for bbox
"""
from ai_tool.bbox import BBox, BBoxes
import unittest


class BBoxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bbox1 = BBox([0, 0, 100, 100])
        cls.bbox2 = BBox([0, 0, 120, 120])

    def test_iou(self):
        from ai_tool.bbox import BBox
        bbox1 = BBox([1, 2, 101, 102])
        bbox2 = BBox([11, 12, 121, 122])
        iou = bbox1 / bbox2
        print("iou", iou)
        assert iou > 0.5

        print('box1 S is', bbox1.S)
        print('box1 & box2', bbox1 & bbox2)
        print('box1 == box2', bbox1 == bbox2)
        print('merge box1 + box2', bbox1 + bbox2)
        print('merge box1 | box2', bbox1 | bbox2)


class BBoxesTest(unittest.TestCase):
    def test_bboxes(self):
        from ai_tool.bbox import BBoxes, BBox
        bb1 = BBoxes(iou_thresh=0.6)
        bb2 = BBoxes()

        bb1.append([1,2, 101, 102])
        bb1.append([1000, 2, 1101, 102])

        bb2.append([11, 12, 111, 112])
        bb2.append([1, 1002, 101, 1102])

        # judge the bbox in bb1
        print("[5, 5, 100, 100] in bb1", BBox([5, 5, 100, 100]) in bb1)
        print("[100, 5, 200, 100] in bb1", BBox([100, 5, 200, 100]) in bb1)

        # bb1 & bb2
        print("bb1 & bb2", bb1 & bb2)
        print("bb1 - bb2", bb1 - bb2)
        print("bb2 - bb1", bb2 - bb1)