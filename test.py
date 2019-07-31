# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/31 0031下午 7:50
测试
"""
import cv2
import unittest


class ImgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jpg = "test.jpg"

    def test_slide_img(self):
        from img_slide import yield_sub_img
        # 使用切图功能
        for bbox, sub_img in yield_sub_img(self.jpg, 0, 0, 180, 60):
            clip = "-".join([str(x) for x in bbox])
            print("分片:{}".format(clip))
            cv2.imshow(clip, sub_img)
            cv2.waitKey(0)