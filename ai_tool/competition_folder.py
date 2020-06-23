#!/home/web/python/ai/bin/python2
# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 18:41
# @File    : competition_folder.py
# @Software: PyCharm
# @__title__ = ''
# @__author__ = ZhaoHui
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor


def subdir_list(dirname):
    """获取目录下所有子目录名
    @param dirname: str 目录的完整路径
    @return: list(str) 所有子目录完整路径组成的列表
    """
    return list(filter(os.path.isdir, map(lambda filename: os.path.join(dirname, filename), os.listdir(dirname))))


def get_index(start, end, step):
    """
    获取索引列表
    :param start:
    :param end:
    :param step:
    :return:
    """
    tmp_list = []
    end_list = []
    for i in range(start, end, step):
        tmp_list.append(i)
    for j in range(end):
        if j not in tmp_list:
            end_list.append(j)
    # return tmp_list
    return end_list


# 针对数据集，划分train/test数据集
# 输入大目录即可，将进行递归遍历，并保证每个子目录按照相同比例划分
def split_train_val(all_dir, train_dir, rate=5):
    """
    将每个类别的数据按分割率进行分割，生成训练集和测试集数据。
    :param all_dir: 分割的原始文件夹
    :param train_dir: 输出的文件夹
    :param rate: 分割率
    :return: 无
    """
    # 获取所有的子文件夹
    dirs = subdir_list(all_dir)
    # 排序
    dirs = sorted(dirs)

    # 如果没有结果文件夹则新建
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    dirs_num = len(dirs)
    logging.warning("一共:{}个目录".format(dirs_num))
    save_list = get_index(0, dirs_num, rate)
    logging.warning("赛选出:{}个目录".format(len(save_list)))
    pool = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index in save_list:
            t = executor.submit(fn=copy_to, src=dirs[index], dst=os.path.join(train_dir, dirs[index].split('/')[-1]))
            pool.append(t)
        for p in pool:
            p.result()
    return


def copy_to(src, dst):
    shutil.copytree(src, dst)
    logging.warning("从:{}---->>>>{}".format(src, dst))
    return


if __name__ == "__main__":
    split_train_val(
        r"/data1/4C/4c_line_data/0618/201906190127_京哈高铁_下行_四平东站_德惠西站/长春西至德惠西",
        # r"/data1/4C/4c_line_data/0618/201906190127_京哈高铁_下行_四平东站_德惠西站/长春西至德惠西7选6_U和PE只有缺失",
        r"/data1/4C/4c_line_data/0618/201906190127_京哈高铁_下行_四平东站_德惠西站/长春西至德惠西7选6绝缘子带xml",
        rate=7
    )
