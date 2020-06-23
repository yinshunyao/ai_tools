import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def get_pic_and_xml_dir(end_dir, key_name):
    """
    保存到哪个目录下,保存为什么名字
    :param end_dir: 保存到这个目录下
    :param key_name: 根据此名字产生最后的2个文件夹,分别存放图片与xml
    :return: 2个字符串
    """
    pic_dir = os.path.join(end_dir, "pic_" + key_name)
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    XML_dir = os.path.join(end_dir, "xml_" + key_name)
    if not os.path.isdir(XML_dir):
        os.makedirs(XML_dir)
    return pic_dir, XML_dir


def choice(path, judge_xml_name):
    """

    :param path: 原始的顶层目录
    :param judge_xml_name: 拷贝哪个文件夹里面的xml
    :return:总计带xml的文件夹数量,以及所有包含xml的文件夹列表
    """
    xml_list = []
    parent = os.listdir(path)
    num = 1
    for child in parent:
        child_path = os.path.join(path, child)
        if not os.path.isdir(child_path):
            continue
        try:
            child_child = os.listdir(child_path)
        except Exception as e:
            child_child = []

        for i in child_child:
            if i == judge_xml_name:
                num += 1
                xml_dir = os.path.join(child_path, i)
                xml_list.append(xml_dir)
    print("一共{}个文件夹".format(num))
    return num, xml_list


def save_pic(xml_list, pic_dir, XML_dir):
    """
    拷贝有xml的对应的图片以及xml
    :param xml_list: 所有包含xml的文件夹列表
    :param pic_dir: 拷到图片到哪个路径
    :param XML_dir: 拷贝xml到哪个路径
    :return:
    """
    # name_list = get_already(r"/data3/京沪高铁+合蚌高铁-5月份/pic_sus/pic_sus_use/局部")
    name_list = []
    xml_num = 0
    copy_num = 0
    copy_list = []
    for xml_dir in xml_list:
        par = os.path.dirname(xml_dir)
        names = os.listdir(xml_dir)
        for name in names:
            if not name.lower().endswith("xml"):
                continue
            # if xml_num % 20 != 0 or name in name_list:
            # if xml_num % 137 != 0 or name in name_list:
            # if (xml_num % 657 != 0) or (name in name_list):
            # if (xml_num % 537 != 0) or (name in name_list):
            # if (xml_num % 597 != 0) or (name in name_list):
            # if (xml_num % 567 != 0) or (name in name_list):
            a = 1
            # if (xml_num % 10000000000 == 0) or (name in name_list):
            if a < 0:
                xml_num += 1
            else:
                src_xml_name = os.path.join(xml_dir, name)
                dst_xml_name = os.path.join(XML_dir, name)
                src_pic_name = os.path.join(par, os.path.splitext(name)[0] + ".jpg")
                dst_pic_name = os.path.join(pic_dir, os.path.splitext(name)[0] + ".jpg")
                # copy_to(src_xml_name, dst_xml_name)
                # copy_to(src_pic_name, dst_pic_name)
                xml_num += 1
                copy_num += 1
                copy_tuple = (src_xml_name, dst_xml_name, src_pic_name, dst_pic_name)
                copy_list.append(copy_tuple)
    return xml_num, copy_num, copy_list


def copy_to(src, dst):
    """
    仅仅是拷贝,文件
    :param src: 原始文件绝对路径
    :param dst: 目标文件绝对路径
    :return: 无
    """
    shutil.copyfile(src, dst)
    print("从:{}---->>>>{}".format(src, dst))
    return


def copy_main(path, judge_xml_name, end_dir, key_name):
    """
    主函数
    :param path: 主目录
    :param judge_xml_name:拷贝哪个xml目录名
    :param end_dir:最后保存到哪个目录下
    :param key_name:最后保存文件的后缀
    :return:
    """
    start_time = time.time()
    num, xml_list = choice(path, judge_xml_name)
    pic_dir, XML_dir = get_pic_and_xml_dir(end_dir, key_name)
    xml_num, copy_num, copy_list = save_pic(xml_list, pic_dir, XML_dir)
    pool = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in copy_list:
            t = executor.submit(copy_pic_and_xml,
                                src_xml_name=i[0],
                                dst_xml_name=i[1],
                                src_pic_name=i[2],
                                dst_pic_name=i[3])
            pool.append(t)
        for p in pool:
            p.result()
    print("共花费时间:{}秒,{}个文件夹,一共{}个xml,筛选之后{}个".format(time.time() - start_time, num, xml_num, copy_num))
    return


def copy_pic_and_xml(src_xml_name, dst_xml_name, src_pic_name, dst_pic_name):
    shutil.copyfile(src_xml_name, dst_xml_name)
    print("从:{}---->>>>{}".format(src_xml_name, dst_xml_name))
    shutil.copyfile(src_pic_name, dst_pic_name)
    print("从:{}---->>>>{}".format(src_pic_name, dst_pic_name))
    return


def get_already(path=None):
    if not path or not os.path.exists(path):
        return []
    name_list = os.listdir(path)
    return name_list


if __name__ == '__main__':
    # judge_xml_name = "xml0.32020_02_11_鸟窝"
    # judge_xml_name = "xml0.32020_02_11_鸟窝"
    # judge_xml_name = "xml_0.32020_02_25_鸟窝全部"
    # judge_xml_name = "xml"
    # judge_xml_name = "xml_0.3鸟窝_epoch_69_loss_060_acc_093"
    # judge_xml_name = "xml_0.320200320_suslimp"
    # judge_xml_name = "xml_0.320200320_susdrop"
    # judge_xml_name = "xml_0.320200320_susabnormal"
    # judge_xml_name = "xml_0.320200320_susnormal"
    # judge_xml_name = "xml_0.320200320_suscut2"
    # judge_xml_name = "xml_0.320200327_jyz"
    # judge_xml_name = "xml_0.320200330_jyz"
    # judge_xml_name = "xml_0.320200331_nest"
    # judge_xml_name = "xml_0.320200402_capmissing"
    # judge_xml_name = "xml_0.320200402_capmissing_2"
    judge_xml_name = "xml_0.320200403_ganhao"
    # path = r"V:\test3c\京沪高铁+合蚌高铁-5月份\CR400BF-5025_20180526  09：00-21：45"
    # path = "/mnt/test3c/京沪高铁+合蚌高铁-5月份/CR400BF-5025_20180526  09：00-21：45"
    # path = "/data5/201906252347_京哈高铁_上行_四平东站_隔5取1_不含已识别的"
    # path = r"/data3/京沪高铁+合蚌高铁-5月份/CR400BF-5025_20180526  09：00-21：45"
    path = r"/mnt/74/home/data/brainwebdiskdata-c4线路/201906190151_哈牡客专线_上行_牡丹江_海林北/牡丹江_海林北"
    # path = "/data2/test_data/2c_ganhao_pic_test_300"
    # path = r"/mnt/72/data/京沪高铁+合蚌高铁-5月份/CR400BF-5025_2018052609002145"
    # choice(path)
    # print(dict((('A', 101), ('B', 102), ('C', 203))))
    # print(dict(('A', 101), ('B', 102), ('C', 203)))
    # end_dir = r"/mnt/test3c/京沪高铁+合蚌高铁-5月份"
    # end_dir = r"/data3/京沪高铁+合蚌高铁-5月份"
    # end_dir = r"/data5/jyz_test"
    # end_dir = r"/data5/nest_test"
    # end_dir = r"/data5/sus_test"
    # end_dir = r"/data5/cap_test"
    end_dir = r"/data5/ganhao_4c_test"
    # key_name = "鸟窝33427张20选1"
    # key_name = "隔5取1_不含已识别的"
    # key_name = "jyz_0330"
    # key_name = "nest_0331"
    # key_name = "suslimp_0331"
    key_name = "ganhao_4c_0403"
    copy_main(path, judge_xml_name, end_dir, key_name)
