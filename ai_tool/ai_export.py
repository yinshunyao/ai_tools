# -*-coding:utf-8-*-
import logging
import os
import codecs
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import pymysql
import json
import xml.etree.ElementTree as ET

import sys
from lxml import etree
from GTUtility.GTConfig.global_dict import GlobalDict
import easyargs
from threading import RLock

# no_xml = ["01010302", "01010306", "01010601", "01010603", "01010501", "01010501"]  # 赛选出不需要的检测类型
# no_xml = ["01010302", "01010306", "01010601", "01010603"]  # 赛选出不需要的检测类型
no_xml = []  # 赛选出不需要的检测类型
#  默认写文件方式
DEFAULT_METHOD = 'w'
# 默认的编码
DEFAULT_ENCODING = 'utf-8'
# xml后缀
XML_EXT = '.xml'
# 颜色配置
frame_color = {1: (0, 255, 0),
               0: (0, 0, 255)}
# 线条的粗细程度,越大则越粗
LINE_SIZE = 3
# 置信度(阈值)配置,小于此值的不导出
thresh = 0.3
# 并发数
thread_nun = 40
# 线程锁,导出xml时 创建任务的时候需要加锁
_lock_mkdir = RLock()
ganhao_code = ["0301", "0302"]
dir_ext = "2019_12_16_管帽_U型环_等电位线"


class ImgFile(object):
    def __init__(self, img_name, shapes, new_img_name=None):
        """
        初始化图片信息
        :param img_name:图片名
        :param shapes: [
        {"points": ("302", "65", "571", "286"), "label": "capmissing", "difficult": "0", "is_ai": 1},
        {"points": ("604", "130", "1142", "572"), "label": "capmissing", "difficult": "0", "is_ai": 1},
        {"points": ("906", "195", "1713", "858"), "label": "capmissing", "difficult": "0", "is_ai": 1},
    ]
        :param new_img_name: 导出的图片名,可不设置
        """
        # 判断此文件是否存在,不存在则抛出异常
        if not img_name or not os.path.exists(img_name):
            raise Exception("{}不存在".format(img_name))

        self.img_name = img_name
        self.shapes = shapes
        self.new_img_name = new_img_name

    def show_img(self, size=LINE_SIZE):
        """
        展示当前图片带缺陷框
        :param size:线条的粗细,数字越大则越粗
        :return:
        """
        image = cv2.imread(self.img_name)
        for one in self.shapes:
            cv2.rectangle(image, (int(one["points"][0]), int(one["points"][1])), (int(one["points"][2]), int(one["points"][3])), frame_color[one["is_ai"]], size)
        cv2.imshow(os.path.split(self.img_name)[1], image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_img(self, size=LINE_SIZE):
        """
        导出带缺陷信息的图片
        :param size: 线条的粗细,数字越大则越粗
        :return:
        """
        image = cv2.imread(self.img_name)  # 读取图片获得一个图片对象
        for one in self.shapes:
            #  分别将缺陷坐标等信息加进去
            cv2.rectangle(image, (int(one["points"][0]), int(one["points"][1])), (int(one["points"][2]), int(one["points"][3])), frame_color[one["is_ai"]], size)
            # 判断有没有传新的图片文件名
        if self.new_img_name is None:
            # 没有传则单独构造一个
            img_name = os.path.split(self.img_name)[1]
            new_name = os.path.join(self.get_img_dir(), img_name)
        else:
            new_name = self.new_img_name
        # 根据图片对象保存图片
        cv2.imwrite(new_name, image)
        self.save_xml()

    def get_img_dir(self):
        """
        如果没有指定新的图片地址,则要使用此目录
        :return:定义的一个存新的img图片的文件夹
        """
        img_dir = os.path.join(os.path.splitext(os.path.split(self.img_name)[0])[0], "img")  # 构造存放输出图片的文件夹
        if not os.path.exists(img_dir):  # 如果不存在则创建一个
            os.mkdir(img_dir)
        return img_dir

    def save_xml(self):
        """
        直接实例化xml类,然后根据缺陷和图片名保存xml文件
        :return:
        """
        xml_file = XmlFile()
        xml_file.save_xml(self.shapes, img_name=self.img_name)


class XmlFile(object):
    def __init__(self, attr_flag=None, bnd_flag=None):
        self.verified = False
        self.attr_flag = attr_flag
        self.bnd_flag = bnd_flag

    def save_xml(self, shapes, img_name=None, xml_name=None):
        """
        根据xml名和图片名已经对应的标注框保存到对应的xml
        :param xml_name:         xml的保存路径,一般不用传
        :param shapes:           标注框的列表
        :param img_name:        图片的绝对的路径
        :return:
        """
        if img_name is None:
            raise Exception("img_name:文件{}不存在".format(img_name))

        if not os.path.isfile(img_name):
            raise Exception("img_name:{}不是文件".format(img_name))
        imageShape = get_h_w_d(img_name)
        imgFolderPath = os.path.dirname(img_name)  # 返回图片的目录
        imgFolderName = os.path.split(imgFolderPath)[-1]  # 返回 图片的目录的最底层
        imgFileName = os.path.basename(img_name)  # 返回图片名字
        #  实例化xml的writer
        writer = XmlWriter(imgfolddir=imgFolderPath,
                           foldername=imgFolderName,
                           filename=imgFileName,
                           imgSize=imageShape,
                           localImgPath=img_name,
                           bnd_flag=self.bnd_flag,
                           attr_flag=self.attr_flag,
                           img_name=img_name)
        writer.verified = self.verified

        for shape in shapes:
            bndbox = shape['points']  # 缺陷框
            label = shape['label']  # 缺陷类型
            difficult = int(shape['difficult'])

            # 特殊的属性标注
            if label in writer.attr_map.keys():
                writer.set_attr_flag(bndbox[0], bndbox[1], bndbox[2], bndbox[3], bndbox[4], label)
            # 正常的标注框
            else:
                writer.add_box(bndbox[0], bndbox[1], bndbox[2], bndbox[3], bndbox[4], label, difficult)

        writer.save(default_xml_name=xml_name)
        return


class XmlWriter(object):

    def __init__(self, imgfolddir, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None, bnd_flag=None, attr_flag=None, img_name=None):
        self.imgfolddir = imgfolddir
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False
        # 定制换bnd，可能涉及展示name到xml标签转换
        self.bnd_map = bnd_flag or {}
        # 属性标签传递进来
        self.attr_map = attr_flag or {}
        self.attr_flag = {}
        for k, attr in self.attr_map.items():
            self.attr_flag[attr] = ""
        self.img_name = img_name
        print(img_name)

    def set_attr_flag(self, xmin, ymin, xmax, ymax, number, label):
        """
        设置特殊的属性标签
        :param xmin:
        :param ymin:
        :param xmax:
        :param ymax:
        :param label:缺陷的label capmissing
        :return:
        """
        attr = self.attr_map[label]
        self.attr_flag[attr] = dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, number=number)

    def _prettify(self, elem):
        """
        :param elem:一般是根节点
        :return:返回一个xml字符串
        """
        rough_string = ET.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace("  ".encode(), "\t".encode())

    def _get_root(self):
        """
            初始化xml对象,返回xml文件的root
            Return XML root
        """
        # Check conditions检查图片名.图片目录,和图片大小是否存在,不存在则返回空
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None
        # 创建一个为annotation的根节点
        top = ET.Element('annotation')
        if self.verified:  # 是否是已经证实的
            top.set('verified', 'yes')

        # 其他自由属性设置,默认是没有的
        for k, v in self.attr_flag.items():
            k_node = ET.SubElement(top, k)
            flag = ET.SubElement(k_node, "flag")
            if not v:
                flag.text = "false"
                continue

            flag.text = "true"
            bndbox = ET.SubElement(k_node, 'bndbox')  # 创建一个为bndbox的叶子节点
            xmin = ET.SubElement(bndbox, 'xmin')  # 创建一个为xmin的叶子节点
            xmin.text = str(v['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')  # 创建一个为ymin的叶子节点
            ymin.text = str(v['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')  # 创建一个为xmax的叶子节点
            xmax.text = str(v['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')  # 创建一个为ymax的叶子节点
            ymax.text = str(v['ymax'])
        # folder的创建
        folder = ET.SubElement(top, 'folder')  # 创建一个为folder的叶子节点
        folder.text = self.foldername
        # filename的创建
        filename = ET.SubElement(top, 'filename')  # 创建一个为filename的叶子节点
        filename.text = self.filename
        # path的创建,文件的绝对路径
        if self.localImgPath is not None:
            localImgPath = ET.SubElement(top, 'path')  # 创建一个为path的叶子节点
            localImgPath.text = self.localImgPath
        # source的创建,内部还有一个database
        source = ET.SubElement(top, 'source')  # 创建一个为source的叶子节点
        database = ET.SubElement(source, 'database')  # 创建一个为database的叶子节点
        database.text = self.databaseSrc
        # size,width,height,depth的创建
        size_part = ET.SubElement(top, 'size')  # 创建一个为size的叶子节点
        width = ET.SubElement(size_part, 'width')  # 创建一个为width的叶子节点
        width.text = str(self.imgSize[1])
        height = ET.SubElement(size_part, 'height')  # 创建一个为height的叶子节点
        height.text = str(self.imgSize[0])
        depth = ET.SubElement(size_part, 'depth')  # 创建一个为depth的叶子节点
        depth.text = str(self.imgSize[2])
        # width.text = str(self.imgSize[1])
        # height.text = str(self.imgSize[0])
        # if len(self.imgSize) == 3:
        #     depth.text = str(self.imgSize[2])
        # else:
        #     depth.text = '1'

        segmented = ET.SubElement(top, 'segmented')  # 创建一个为segmented的叶子节点
        segmented.text = '0'
        return top

    def add_box(self, xmin, ymin, xmax, ymax, number, name, difficult):
        """
        往self.boxlist里面添加1个或者多个的缺陷坐标框
        :param xmin: x1
        :param ymin: y1
        :param xmax: x3
        :param ymax: y3
        :param name:缺陷名称,类似capmissing
        :param difficult:"0"或者"1"
        :return:
        """
        # logging.warning("xmin之前:{}".format(xmin))
        xmin = str(min(float(xmin), float(self.imgSize[1])))
        # logging.warning("width:{}, xmin之后:{}".format(self.imgSize[0], xmin))

        # logging.warning("ymin之前:{}".format(ymin))
        ymin = str(min(float(ymin), float(self.imgSize[0])))
        # logging.warning("hight:{}, ymin之后:{}".format(self.imgSize[1], ymin))

        # logging.warning("xmax之前:{}".format(xmax))
        xmax = str(min(float(xmax), float(self.imgSize[1])))
        # logging.warning("width:{}, xmax之后:{}".format(self.imgSize[0], xmax))

        # logging.warning("ymax之前:{}".format(ymax))
        ymax = str(min(float(ymax), float(self.imgSize[0])))
        # logging.warning("hight:{}, ymax之后:{}".format(self.imgSize[1], ymax))
        # time.sleep(1)

        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        label = self.bnd_map.get(name) or name
        bndbox['name'] = label
        bndbox['difficult'] = difficult
        bndbox['number'] = number
        self.boxlist.append(bndbox)

    def add_object(self, top):
        """
        遍历缺陷框框,分别加入到object
        :param top: 根节点
        :return:
        """
        for each_object in self.boxlist:
            object_item = ET.SubElement(top, 'object')  # 创建一个为object的叶子节点
            name = ET.SubElement(object_item, 'name')  # 创建一个为name的叶子节点
            name.text = each_object['name']
            # name_src = ET.SubElement(object_item, 'name_src')
            # name_src.text = each_object['name']
            pose = ET.SubElement(object_item, 'pose')  # 创建一个为pose的叶子节点
            pose.text = "Unspecified"
            truncated = ET.SubElement(object_item, 'truncated')  # 创建一个为truncated的叶子节点
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin'])) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(float(each_object['xmax'])) == int(float(self.imgSize[1]))) or (int(float(each_object['xmin'])) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = ET.SubElement(object_item, 'difficult')  # 创建一个为truncated的叶子节点
            difficult.text = str(bool(each_object['difficult']) & 1)
            value = ET.SubElement(object_item, "value")  # 创建一个节点为value,存放的是杆号的值,为字符串形式
            value.text = str(each_object['number'])
            bndbox = ET.SubElement(object_item, 'bndbox')  # 创建一个为truncated的叶子节点
            xmin = ET.SubElement(bndbox, 'xmin')  # 创建一个为xmin的叶子节点
            xmin.text = str(each_object['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')  # 创建一个为ymin的叶子节点
            ymin.text = str(each_object['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')  # 创建一个为xmax的叶子节点
            xmax.text = str(each_object['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')  # 创建一个为ymax的叶子节点
            ymax.text = str(each_object['ymax'])

    def get_xml_dir(self):
        # xml_dir = os.path.join(os.path.split(self.imgfolddir)[0], "xml")  # xml目录拼接
        global _lock_mkdir
        # name = exclude_name()
        # xml_dir = os.path.join(self.imgfolddir, "xml" + str(thresh) + name)  # xml目录拼接
        xml_dir = os.path.join(self.imgfolddir, "xml" + str(thresh) + dir_ext)  # xml目录拼接
        # 创建文件夹加锁
        with _lock_mkdir:
            if not os.path.exists(xml_dir):  # 判断目录是否存在,如果不存在则创建一个
                logging.warning("{}不存在,准备创建".format(xml_dir))
                os.mkdir(xml_dir)
                logging.warning("创建目录{}成功".format(xml_dir))
        return xml_dir

    def save(self, default_xml_name=None):
        """
        保存到xml文件
        :param default_xml_name:xml文件名,可以不传,则单独构造一个
        :return:
        """
        root = self._get_root()  # 获取根节点
        self.add_object(root)  # 将子节点加入到根节点

        # 有设置xml名就使用,如果没有则另外构造
        if default_xml_name is None:
            xml_dir = self.get_xml_dir()  # 获取xml目录
            xml_name = os.path.join(xml_dir, os.path.splitext(self.filename)[0] + XML_EXT)  # 构造xml名字
            logging.warning("没有提前设置xml名,自定义为:{}".format(xml_name))
        else:
            xml_name = default_xml_name

        out_file = codecs.open(xml_name, DEFAULT_METHOD, encoding=DEFAULT_ENCODING)  # 创建一个文件对象

        #  构造为字符串,以utf-8编码写入
        prettify_result = self._prettify(root)
        out_file.write(prettify_result.decode(DEFAULT_ENCODING))  # 编码并写入
        out_file.close()  # 关闭连接


def exclude_name():
    l1 = []
    for code in no_xml:
        l1.append(GlobalDict.get_name(code))
    if len(l1) == 0:
        return ""
    else:
        return "不包含_" + "_".join(l1)


def get_h_w_d(img_name):
    """
    根据图片获取对应的高度宽度深度
    :param img_name: 图片绝对路径
    :return: (高度,宽度,深度)
    """
    sp = cv2.imread(img_name).shape
    imageShape = []
    for i in sp:
        imageShape.append(str(i))

    return tuple(imageShape)


def get_cursor():
    """
    获取一个数据库的连接对象
    :return:
    """
    db = pymysql.connect(host='192.168.1.73', user='root', passwd='123456', db='brainweb', port=3306, charset='utf8')
    cursor = db.cursor()
    return cursor


def get_data(img):
    """
    单张图片判断是否导出xml
    :param img:
    :return:
    """
    if int(img[3]) == 1:
        shapes = []
        xml_file = XmlFile()
        cursor = get_cursor()
        cursor.execute('SELECT defect_code,defect_info_original FROM brainweb.PictureDefect WHERE task_id={} and pic_id={}'.format(img[0], img[1]))
        defect_list = cursor.fetchall()
        for a in defect_list:
            code = a[0]
            defect = a[1]
            defect = json.loads(defect)
            confidence = float(defect.get("confidence", 1))
            if confidence < thresh:
                logging.warning("当前阈值小于:{}-->直接过滤".format(thresh))
                continue
            if str(code) in no_xml:
                # 赛选出不需要导出的缺陷
                logging.warning("{}不展示".format(GlobalDict.get_name(code)))
                continue
            # if code in ganhao_code:
            #     points = (defect.get("x1"), defect.get("y1"), defect.get("x2"), defect.get("y2"), defect.get("number"))
            # else:
            points = (defect.get("x1"), defect.get("y1"), defect.get("x2"), defect.get("y2"), defect.get("number", "0"))

            if str(code) == "0301":
                label = "ganhao"
            else:
                label = defect.get("number")
            if not label:
                sys.exit("label为空:{}".format(code))
            point = {"points": points, "label": label, "difficult": "0", "is_ai": 1}
            shapes.append(point)
        if len(shapes) == 0:
            # 没有缺陷不导出xml
            return
        xml_file.save_xml(shapes=shapes, img_name=img[2])
    else:
        pass
    return


@easyargs
def export_xml(task_id, path_like="", detect_flag=3):
    """
    导出xml入口函数
    :param task_id:任务id
    :param path_like: 路径条件 "海林北_尚志南\K275528_409"
    :param detect_flag: 线程数 建议50
    :return:
    """
    cursor = get_cursor()
    # 先判断任务是否全部完成
    # sql_count = 'SELECT count(*) FROM brainweb.PictureDetection where DDID={} and detect_flag!={};'.format(task_id, detect_flag)
    # logging.warning("任务{}预期每张图片检测模型数：{}".format(task_id, detect_flag))
    # cursor.execute(sql_count)
    # count = cursor.fetchall()[0][0]
    # if count > 0:
    #     logging.error("任务{}还没有检测结束，预期每张图片检测{}个模型，实际有图片detect_flag统计错误".format(task_id, detect_flag))
    #     logging.warning("执行SQL语句校验结果\n{}".format(sql_count))
    #     return

    sql_count = 'SELECT count(*) FROM brainweb.PictureDetection where DDID={} and detect_flag={};'.format(task_id,
                                                                                                          detect_flag)
    cursor.execute(sql_count)
    count = cursor.fetchall()[0][0]
    if count == 0:
        logging.error("任务{}还没有开始检测，预期每张图片检测{}个模型，实际检测图片数为0".format(task_id, detect_flag))
        logging.warning("执行SQL语句校验结果\n{}".format(sql_count))
        return

    # 图片过滤
    if not path_like:
        cursor.execute('SELECT DDID,id,FramePath,defect_flag FROM brainweb.PictureDetection WHERE DDID= {}'.format(task_id))
    else:
        cursor.execute(
            'SELECT DDID,id,FramePath,defect_flag FROM brainweb.PictureDetection WHERE DDID= {} and FramePath like "%{}%"'.format(task_id, path_like))
    data = cursor.fetchall()
    pool = []
    with ThreadPoolExecutor(max_workers=thread_nun) as executor:
        for i in data:
            t = executor.submit(get_data, img=i)
            pool.append(t)
        for p in pool:
            p.result()
    return


if __name__ == '__main__':
    t1 = time.time()
    # python3 /home/web/GTUtility/GTTools/ai_export.py   task_id
    # python3 /home/web/GTUtility/GTTools/ai_export.py   task_id  --path_like "海林北_尚志南/K275528_409"
    export_xml()
    logging.warning("开了{}个线程导出xml,此次导出{}共花费了:{}分{}秒".format(thread_nun, "杆号", *divmod(time.time() - t1, 60)))

# shapes = [
#     {"points": ("302", "65", "571", "286"), "label": "capmissing", "difficult": "0", "is_ai": 1},
#     {"points": ("604", "130", "1142", "572"), "label": "capmissing", "difficult": "0", "is_ai": 1},
#     {"points": ("906", "195", "1713", "858"), "label": "capmissing", "difficult": "0", "is_ai": 1},
# ]
# xml_file = XmlFile()
# xml_file.save_xml(shapes=shapes, img_name=r"Z:\ins\zh_test\JPEGImages\000001.jpg")
# print(os.path.isfile(r"Z:\ins\zh_test\JPEGImages\0000011.jpg"))
# img_file = ImgFile(img_name=r"Z:\ins\zh_test\JPEGImages\000001.jpg", shapes=shapes)
# img_file.show_img()
