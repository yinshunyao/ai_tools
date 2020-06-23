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


class NumEndException(Exception):
    pass


def export(mv_file, idx_file, dst_path, start=0, end=0, max_workers=5):
    """

    :param mv_file: mv所在目录
    :param idx_file: idx所在目录
    :param dst_path: 结果存放目录
    :param start:
    :param end:
    :param max_workers:
    :return:
    """
    mv_op = resolvemv_ope(mv_file, idx_file)  # 实例化对象
    nums = mv_op.getFrameNum()  # 获取总数
    end = end or nums  # 导出多少张,不传end意味着导出全部
    logging.warning("总共有:{}张图片".format(end - start))
    if not os.path.exists(dst_path):  # 没有目录则创建
        os.makedirs(dst_path)
    thread_pool = ThreadPool(max_workers=max_workers)  # 线程池的方式
    # 分拆
    width = (end - start) // max_workers or 1  # 每个分支多少张图片
    logging.warning("总计{}，每组{}个".format(end - start, width))
    for i in range(start, end, width):
        logging.warning("拆分提图：[{},{})".format(i, min(end, i + width)))
        thread_pool.submit(mv_op.save_img_list, i, i + width, dst_path, 3)
    thread_pool.shutdown()
    return end - start


if __name__ == '__main__':
    t1 = time.time()
    # src_path = r"E:\2018年线路数据拷贝\京沪高铁+合蚌高铁\5月份\北京-上海 上下行"
    src_path = r"R:\data\京沪高铁+合蚌高铁-5月份\CR400BF-5025_2018052609002145"
    dst_path = r"F:\京沪高铁+合蚌高铁-5月份"
    total, num = 0, 0
    a = 0
    try:
        for root, dirs, files in os.walk(src_path):
            print("export image for :", root)
            for f in files:
                if f.endswith(".MV"):
                    mv_file = os.path.join(root, f)
                    b=os.path.basename(mv_file).split("_")[3]
                    print("b:{},type:{}".format(b,type(b)))
                    # if os.path.basename(mv_file).split("_")[3] != 2:
                    #     continue
                    idx_file = mv_file + ".IDX"
                    print(mv_file, idx_file)
                    dst_dir = os.path.join(dst_path, root[len(src_path) + 1:])
                    # num = export(mv_file, idx_file, dst_dir, max_workers=50)
                    total += num
                    a += 1
    except NumEndException as e:
        num = e.message
        total += int(num)
    except Exception as e:
        print(e)

    t2 = time.time()
    t = t2 - t1
    print("a:{}".format(a))
    print("总共导出{}张图片,耗时:{}秒,平均每张图片:{}毫秒".format(total, t, (t / total) * 1000))
