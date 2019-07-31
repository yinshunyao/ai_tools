# -*-coding:utf-8-*- 
"""
Author：yinshunyao
Date:2019/7/31 8:18
"""
from distutils.core import setup
from setuptools import setup, find_packages

# 从readme中获取信息
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'ai_tool',
    version = '1.3',
    keywords = ('ai', 'tools', 'iou', 'picture slice'),
    description = 'compute the iou, slice picture etc.',
    long_description = long_description,
    long_description_content_type="text/markdown",
    license = 'MIT License',
    url = 'https://github.com/yinshunyao/ai_tools',
    author = 'yinshunyao',
    author_email = 'yinshunyao@qq.com',
    packages = find_packages(),
    install_requires=["numpy", "opencv-python"],
    platforms = 'any',
)