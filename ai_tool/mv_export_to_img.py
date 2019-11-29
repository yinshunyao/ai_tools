#!/home/web/python/ai/bin/python2
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 11:01
# @File    : mv_export_to_img.py
# @Software: PyCharm
# @__title__ = ''
# @__author__ = ZhaoHui
import logging
# from PIL import Image
# import io
import os
import datetime
import time
from ctypes import *
from GTUtility.GTBase.thread_pool import ThreadPool



headerLength = 100


class MVIndex_header(Structure):
    _pack_ = 1
    _fields_ = [
        ('width', c_uint32),
        ('height', c_uint32),
        ('channel', c_uint32),
        ('pad', c_ubyte * 88),
    ]

    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        memmove(addressof(self), data, sizeof(self))
        return len(data)


class MVIndex_body(Structure):
    _pack_ = 1
    _fields_ = [
        ('size', c_uint32),
        ('offset', c_ulonglong),
        ('year', c_ushort),
        ('month', c_ushort),
        ('dayOfWeek', c_ushort),
        ('day', c_ushort),
        ('hour', c_ushort),
        ('minute', c_ushort),
        ('second', c_ushort),
        ('milliseconds', c_ushort),
        ('time', c_ulonglong),
        ('pad', c_ubyte * 72),
    ]

    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        memmove(addressof(self), data, sizeof(self))
        return len(data)


def _write(i, system_time, img, save_dir):
    """
    写入文件
    :param i: 索引
    :param system_time: 时间戳
    :param img: 图片数据
    :param save_dir: 保存的路径
    :return:
    """
    file_name = os.path.join(save_dir, "[{:0>6d}][{}].jpg".format(i, system_time))
    logging.warning("write to {}".format(file_name))
    with open(file_name, "wb") as f:
        f.write(img)
    return


class resolvemv_ope:
    def __init__(self, mvFile, mvindexFile):
        self.mvFile = mvFile
        self.mvindexFile = mvindexFile
        self.list = []
        self.mvheaderObject = MVIndex_header()
        self.mvbodyObject = MVIndex_body()
        self.resolveMVIndex()

    def resolveMVIndex(self):
        if os.path.exists(self.mvindexFile):
            with open(self.mvindexFile, "rb") as f:
                self.content = f.read()
                if self.content is not None and len(self.content) > 0:
                    self.framenum = int((len(self.content) - 1) / 100)
                    self.mvheaderObject.decode(self.content[0:100])
                    self.width = self.mvheaderObject.width
                    self.hight = self.mvheaderObject.height
        else:
            self.framenum = 0

    def getFrameNum(self):
        return self.framenum

    def extractFrameInfo(self, number):
        returnval = []
        if self.framenum < number or number < 0:
            return returnval

        self.mvbodyObject.decode(self.content[number * headerLength:number * headerLength + 100])
        returnval.append(self.mvbodyObject.size)
        returnval.append(self.mvbodyObject.offset)

        year = self.mvbodyObject.year
        month = self.mvbodyObject.month
        dayOfWeek = self.mvbodyObject.dayOfWeek
        day = self.mvbodyObject.day
        hour = self.mvbodyObject.hour
        minute = self.mvbodyObject.minute
        second = self.mvbodyObject.second
        milliseconds = self.mvbodyObject.milliseconds
        dt = ('%d-%d-%d %d:%d:%d') % (year, month, day, hour, minute, second)
        d = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        # print(dt,d)
        secondsFrom1970 = int(time.mktime(d.timetuple()))
        milliseconds = secondsFrom1970 * 1000 + milliseconds
        returnval.append(milliseconds)
        return returnval

    def readPicture(self, number):
        returnval = self.extractFrameInfo(number)
        if len(returnval):
            systime = returnval.pop()
            offset = returnval.pop()
            size = returnval.pop()
            return systime, offset, size
        return None, None, None

    def save(self, offset, size, file_name):
        """
        二进制保存
        :param offset:
        :param size:
        :param file_name:
        :return:
        """
        byte_image = self.getPicture(offset, size)
        with open(file_name, "wb") as f:
            f.write(byte_image)

    def getPicture(self, offset, size):
        """
        获取图片数据
        :param offset:
        :param size:
        :return:
        """
        with open(self.mvFile, "rb") as f:
            f.seek(offset, 0)
            byte_image = f.read(size)

        return byte_image

    def yield_img(self, start, end):
        with open(self.mvFile, "rb") as f:
            for i in range(start, end):
                system_time, offset, size = self.readPicture(i)
                f.seek(offset, 0)
                yield i, system_time, f.read(size)

    def save_img_list(self, start, end, save_dir, thread_max=1):
        thread_pool = ThreadPool(max_workers=thread_max, queue_max_size=thread_max * 2)
        for i, system_time, img in self.yield_img(start, end):
            thread_pool.submit(_write, i, system_time, img, save_dir)

        thread_pool.shutdown()


def export(mvFile, mvindexFile, base_dir, start=1, end=None, max_workers=5):
    """
    根据mv和mvindex导出图片
    :param mvFile: mv文件的绝对路径
    :param mvindexFile: mvindex的绝对路径
    :param base_dir: 要导出到哪个文件夹里面
    :return:导出的图片总数
    """
    re = resolvemv_ope(mvFile, mvindexFile)  # 实例化对象
    Num = re.getFrameNum()  # 获取总数
    end = min(end or Num, Num) + 1  # 导出多少张,不传end意味着导出全部
    logging.warning("总共有:{}张图片".format(end - start))
    if not os.path.exists(base_dir):  # 没有目录则创建
        os.mkdir(base_dir)
    thread_pool = ThreadPool(max_workers=max_workers)  # 线程池的方式
    # 分拆
    width = (end - start) // max_workers or 1  # 每个分支多少张图片
    logging.warning("总计{}，每组{}个".format(end - start, width))
    for i in range(start, end, width):
        logging.warning("拆分提图：[{},{})".format(i, min(end, i + width)))
        thread_pool.submit(re.save_img_list, i, i + width, base_dir, 3)
    thread_pool.shutdown()
    return end - start


if __name__ == '__main__':
    t1 = time.time()
    base = r"E:\2019.11.22_唐跃明_侯马数据\2C\2019.11.11大西高铁2C\20191111_125807684_大西高铁_太原南_永济北_1"
    mv_file = os.path.join(base, "C2_A_0_2_20191111_125807_684.MV")
    mvindexFile = os.path.join(base, "C2_A_0_2_20191111_125807_684.MV.IDX")
    base_dir = os.path.join(base, "C2_A_0_2_20191111_125807_684")
    Num = export(mvFile=mv_file,
                 mvindexFile=mvindexFile,
                 base_dir=base_dir,
                 max_workers=50)
    t2 = time.time()
    t = t2 - t1
    print("总共导出{}张图片,耗时:{}秒,平均每张图片:{}毫秒".format(Num, t, (t / Num) * 1000))
