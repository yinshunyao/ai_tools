# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/31 7:50
test
"""
import cv2
import unittest


class ImgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jpg = "test.jpg"

    def test_slide_img(self):
        from ai_tool.img_slide import yield_sub_img
        # 使用切图功能
        for bbox, sub_img in yield_sub_img(self.jpg, 0, 0, 180, 60):
            clip = "-".join([str(x) for x in bbox])
            print("slide:{}".format(clip))
            cv2.imshow(clip, sub_img)
            cv2.waitKey(0)

    def test_slide_para(self):
        from ai_tool.img_slide import yield_sub_img
        for box, img in yield_sub_img(None, 0, 0, 1400, 1400, 6576, 4384):
            print(box[0], box[1], box[2] - box[0], box[3] - box[1])