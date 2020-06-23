# -*-coding:utf-8-*-
import logging

FORMAT = '[%(asctime)s][PID:%(process)d] %(levelname)s: %(message)s (%(filename)s@line:%(lineno)d)'
LEVEL = logging.WARNING
logging.basicConfig(level=LEVEL, format=FORMAT)
import os
import codecs
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import json
import sys
import numpy as np
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from lxml import etree
from configparser import ConfigParser
from threading import RLock
from GTUtility.FdiskData import resolvemv_ope
from GTUtility.GTConfig.global_dict import GlobalDict
from GTUtility.GTConfig.config_connections import mysql_exc

# 线程锁,导出xml时 创建文件夹的时候需要加锁
_lock_mkdir = RLock()
CONFIG_FILE_NAME = "export.ini"
config_file = os.path.join(os.path.split(os.path.abspath(__file__))[0], CONFIG_FILE_NAME)
config = ConfigParser()
config.read(config_file, encoding="utf-8")
task_id = int(config.get("SET", "task_id", fallback=0))
img_dir = config.get("SET", "img_dir")
xml_dir = config.get("SET", "xml_dir")
path_like = config.get("SET", "path_like", fallback=0)
detect_flag = config.get("SET", "detect_flag", fallback=1)
thread_nun = int(config.get("SET", "thread_nun", fallback=8))
interval = int(config.get("SET", "interval", fallback=0))
export_code = config.get("SET", "code", fallback="")
export_code_list = export_code.split(",")
attr = int(config.get("SET", "attr", fallback=0))
thresh = float(config.get("SET", "thresh", fallback=0))
#  默认写文件方式
DEFAULT_METHOD = 'w'
# 默认的编码
DEFAULT_ENCODING = 'utf-8'
# xml后缀
XML_EXT = '.xml'
# img后缀
IMG_EXT = '.jpg'
# 颜色配置
frame_color = {1: (0, 255, 0),
               0: (0, 0, 255)}
# 线条的粗细程度,越大则越粗
LINE_SIZE = 3
dir_ext = "20200403_ganhao"


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
            cv2.rectangle(image, (int(one["points"][0]), int(one["points"][1])),
                          (int(one["points"][2]), int(one["points"][3])), frame_color[one["is_ai"]], size)
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
            cv2.rectangle(image, (int(one["points"][0]), int(one["points"][1])),
                          (int(one["points"][2]), int(one["points"][3])), frame_color[one["is_ai"]], size)
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

    def _get_h_w_d(self, img_name):
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
        imageShape = self._get_h_w_d(img_name)
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
                writer.add_box(bndbox[0], bndbox[1], bndbox[2], bndbox[3], bndbox[4], bndbox[5], bndbox[6], bndbox[7],
                               label, difficult)
                # x1,y1,x2,y2,number,class_name,confidence,class_confidence,label, difficult
        writer.save(default_xml_name=xml_name)
        return


class XmlWriter(object):

    def __init__(self, imgfolddir, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None,
                 bnd_flag=None, attr_flag=None, img_name=None):
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
        segmented = ET.SubElement(top, 'segmented')  # 创建一个为segmented的叶子节点
        segmented.text = '0'
        return top

    def add_box(self, xmin, ymin, xmax, ymax, number, class_name, confidence, class_confidence, name, difficult):
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
        xmin = str(min(float(xmin), float(self.imgSize[1])))
        ymin = str(min(float(ymin), float(self.imgSize[0])))
        xmax = str(min(float(xmax), float(self.imgSize[1])))
        ymax = str(min(float(ymax), float(self.imgSize[0])))
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        label = self.bnd_map.get(name) or name
        bndbox['name'] = label
        bndbox['class_name'] = class_name
        bndbox['confidence'] = confidence
        bndbox['class_confidence'] = class_confidence
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
            pose = ET.SubElement(object_item, 'pose')  # 创建一个为pose的叶子节点
            pose.text = "Unspecified"
            truncated = ET.SubElement(object_item, 'truncated')  # 创建一个为truncated的叶子节点
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin'])) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(float(each_object['xmax'])) == int(float(self.imgSize[1]))) or (
                    int(float(each_object['xmin'])) == 1):
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
            class_name = ET.SubElement(bndbox, 'class_name')
            class_name.text = str(each_object['class_name'])
            confidence = ET.SubElement(bndbox, 'confidence')
            confidence.text = str(each_object['confidence'])
            class_confidence = ET.SubElement(bndbox, 'class_confidence')
            class_confidence.text = str(each_object['class_confidence'])

    def get_xml_dir(self):
        # xml_dir = os.path.join(os.path.split(self.imgfolddir)[0], "xml")  # xml目录拼接
        global _lock_mkdir
        # name = exclude_name()
        # xml_dir = os.path.join(self.imgfolddir, "xml" + str(thresh) + name)  # xml目录拼接
        xml_dir = os.path.join(self.imgfolddir, "xml_" + str(thresh) + dir_ext)  # xml目录拼接
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


def get_data(pic_id: int, name: str, img_type: int, path_list: list = []):
    """

    :param pic_id: 图片id
    :param name: 图片保存在磁盘的绝对路径或者是mv的路径以及对应的offset以及size
    :param img_type: 0:本地图片,1:本地mv,2:hdfs下图片,3:hdfs下mv
    :param path_list: 主要是判断是否需要去重
    :return:
    """
    # 判断是否是已经导出的图片
    if not path_list:
        pass
    else:
        if name in path_list:
            print("*********************************{}之前已经导出过,不再导出".format(name))
            return
        else:
            pass
    # 定义一个空的列表来存框框信息,如果最后为空,则不导出xml
    shapes = []
    xml_file = XmlFile()
    # 根据图片id从缺陷表中查询缺陷信息,这里测试过,只用pic_id比使用task_id加pic_id速度更快
    sql = """SELECT defect_code,defect_info_original FROM PictureDefect WHERE pic_id={}""".format(pic_id)
    defect_list = mysql_exc.excute(sql)
    for a in defect_list:
        code = a[0]
        defect = a[1]
        defect = json.loads(defect)
        if not check_special(code, defect):
            continue
        label = defect.get("class_name")
        if not label:
            sys.exit("label为空:{}".format(code))
        points = (defect.get("x1"), defect.get("y1"), defect.get("x2"), defect.get("y2"), defect.get("number", "0"),
                  defect.get("class_name"), defect.get("confidence", "0"), defect.get("class_confidence", "1"))
        point = {"points": points, "label": label, "difficult": "0", "is_ai": 1}
        shapes.append(point)
    if len(shapes) == 0:
        # 没有缺陷不导出xml
        return
    if img_type == 0:
        #  本地图片,老样子导出就行
        name_path, img_name = os.path.split(name)
        end_img_name = os.path.join(img_dir, img_name)
        if not os.path.exists(end_img_name):
            img2img(name, end_img_name)
        xml_name = os.path.join(xml_dir, os.path.splitext(img_name)[0] + XML_EXT)
        xml_file.save_xml(shapes=shapes, img_name=name, xml_name=xml_name)
    elif img_type == 1:
        # 本地mv
        path_dict = parse_(name)
        if not path_dict:
            return
        offset = path_dict.get("offset")
        size = path_dict.get("size")
        mv = path_dict.get("path")
        end_name = os.path.splitext(os.path.split(path_dict.get("path"))[1])[0]
        pic_name = os.path.join(img_dir, end_name + offset + size + IMG_EXT)
        if not os.path.exists(pic_name):
            zero_save_img(mv, pic_name, offset, size)
        xml_name = os.path.join(xml_dir, end_name + offset + size + XML_EXT)
        xml_file.save_xml(shapes=shapes, img_name=pic_name, xml_name=xml_name)
    elif img_type == 3:
        # HDFS 下MV
        path_dict = parse_(name)
        if not path_dict:
            return
        offset = path_dict.get("offset")
        size = path_dict.get("size")
        mv = path_dict.get("path")
        end_name = os.path.splitext(os.path.split(path_dict.get("path"))[1])[0]
        pic_name = os.path.join(img_dir, end_name + offset + size + IMG_EXT)
        if not os.path.exists(pic_name):
            save_img_from_hdfs(mv, pic_name, offset, size)
        xml_name = os.path.join(xml_dir, end_name + offset + size + XML_EXT)
        xml_file.save_xml(shapes=shapes, img_name=pic_name, xml_name=xml_name)
    else:
        return
    return


def parse_(path: str):
    tmp_dict = {}
    try:
        mv_name = path.split("?")[0]
        other_parse = path.split("?")[1]
        query_list = other_parse.split("&")
        for query in query_list:
            k_v = query.split("=")
            k, v = k_v[0], k_v[1]
            tmp_dict[k] = v
        tmp_dict["path"] = mv_name
    except Exception as e:
        logging.error("parse_ error :{}".format(e), exc_info=True)
    finally:
        return tmp_dict


def save_img_from_hdfs(mv, pic_name: str, offset: (int, str), size: (int, str)):
    res = resolvemv_ope(mv, None, is_hdfs=True)
    # 通过读取二进制
    byte_image = res.get_pic_from_mv(offset, size)
    # 需要进行转化再写入
    cv2.imwrite(pic_name, cv2.imdecode(np.asanyarray(bytearray(byte_image), dtype="uint8"), cv2.IMREAD_COLOR))
    return


def img2img(name: str, end_img_name: str):
    count = os.path.getsize(name)
    with open(name, "rb") as f_src:
        with open(end_img_name, 'wb') as f_dst:
            os.sendfile(f_dst.fileno(), f_src.fileno(), offset=None, count=count)
    logging.warning("图片{}--->{}".format(name, end_img_name))
    return


def zero_save_img(mv: str, pic_name: str, offset: (int, str), size: (int, str)):
    """
    使用0拷贝的方式保存图
    :param mv: mv的绝对路径
    :param pic_name: 图片的绝对路径
    :param offset: 偏移量
    :param size: 大小
    :return:
    """
    res = resolvemv_ope(mv, None, is_hdfs=False)
    with res.open_mv() as f_mv:
        with open(pic_name, "wb") as f_pic:
            os.sendfile(f_pic.fileno(), f_mv.fileno(), offset=int(offset), count=int(size))
    logging.warning("mv :{} save :{}".format(mv, pic_name))
    return


def parse_path(path: str):
    """
    解析路径
    :param path:
    :return:一个字典,带详细信息,可根据信息导出对应的图片
    """
    tmp_dict = {}
    try:
        end_parse = urlparse(path)
        end_path = end_parse.path
        query_list = end_parse.query.split("&")
        for query in query_list:
            k_v = query.split("=")
            k, v = k_v[0], k_v[1]
            tmp_dict[k] = v
        tmp_dict["path"] = end_path
    except Exception as e:
        logging.error("parse_path error :{}".format(e), exc_info=True)
    finally:
        return tmp_dict


def main(is_export: int = 0):
    """
    导出xml以及图片的主函数
    :return: 无,以文件的形式保存到磁盘中,到配置的目录中查看
    """

    where = ""
    if not path_like:
        pass
    else:
        where += "and PictureDetection.FramePath like '%{}%'".format(path_like)
    if attr > 0:
        where += "and PictureDetection.img_attr1 ={}".format(attr)
    sql = """SELECT distinct(PictureDetection.id),PictureDetection.FramePath,PictureDetection.img_type FROM PictureDetection inner join  PictureDefect on PictureDetection.id=PictureDefect.pic_id where PictureDetection.DDID={task_id}  and PictureDetection.defect_flag>0 and PictureDefect.defect_code in ({defect_code_tuple}) {where};""".format(
        task_id=task_id, defect_code_tuple=export_code, where=where)
    print("本次导出的sql语句为:{}".format(sql))
    data = mysql_exc.excute(sql)
    data, count_data = interval_selection(data=data, interval=interval)
    logging.warning("此次有{}张图片的数据".format(count_data))
    if not is_export or not count_data:
        pass
    else:
        # 判断目标目录,如果不存在则创建
        if not os.path.exists(img_dir):
            logging.warning("{}不存在,创建".format(img_dir))
            os.makedirs(img_dir)
        if not os.path.exists(xml_dir):
            logging.warning("{}不存在,创建".format(xml_dir))
            os.makedirs(xml_dir)
        t1 = time.time()
        export(data)
        cost = time.time() - t1
        minute, second = divmod(cost, 60)
        logging.warning("导出xml{}个,此次导出共花费了:{}分{}秒,平均每秒{}个导出".format(count_data, minute, second, count_data / cost))
    return


def export(data: list):
    pool = []
    with ThreadPoolExecutor(max_workers=thread_nun) as executor:
        for i in data:
            t = executor.submit(get_data, pic_id=i[0], name=i[1], img_type=i[2], path_list=[])
            pool.append(t)
        for p in pool:
            p.result()
    return


def check_special(code: str, defect: dict):
    """
    检查特殊的
    :param code: 缺陷代码
    :param defect:缺陷信息
    :return: 如果为False,则不导出次缺陷框框信息
    """
    if str(code) in ("01010601", "0307"):
        # 等电位线比较特殊,要单独进行判断
        h = float(defect.get("y2")) - float(defect.get("y1"))
        w = float(defect.get("x2")) - float(defect.get("x1"))
        if not h or not w or not (1.85 > (h / w) > 0.4) or not (1200 > h > 45) or not (1200 > w > 45):
            logging.warning("等电位线滤除:{}".format(defect))
            return False
    # 使用阈值过滤
    confidence = float(defect.get("confidence", 1))
    if confidence < thresh:
        logging.warning("当前阈值小于:{}-->直接过滤".format(thresh))
        return False
    if str(code) not in export_code_list:
        # 只要配置的缺陷类型
        logging.warning("{}不展示".format(GlobalDict.get_name(code)))
        return False
    return True


def interval_selection(data: list, interval: int = 0):
    """
    隔几抽一
    :param data:数据库按照条件抽取出来的数据
    :param interval: 隔多少个抽取一个
    :return:result_list:需要导出的数据列表,count_data:数据总量
    """
    if interval == 0:
        count_data = len(data)
        return data, count_data
    else:
        result_list = []
        num = 0
        for i in data:
            if num % interval == 0:
                result_list.append(i)
            num += 1
        count_data = len(result_list)
        return result_list, count_data


if __name__ == '__main__':
    input_list = (0, 1)
    while 1:
        try:
            input_data = int(input("计数输入0,导出输入1--------->"))
            if input_data not in input_list:
                logging.warning("输入错误:{}---type{}".format(input_data, type(input_data)))
                continue
            main(input_data)
            break
        except Exception as e:
            logging.error("错误请重试,error:{}".format(e), exc_info=True)
