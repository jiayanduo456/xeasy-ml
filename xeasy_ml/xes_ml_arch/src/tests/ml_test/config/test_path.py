# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import os
import sys
from ..tttttest import w
print(os.getcwd())
import configparser
conf = configparser.ConfigParser()
conf.read('./ml.conf')
print(os.path.dirname(__file__) + os.sep + '../')
#当文件不是包或者定义在项目内时；如果我们要用包的相对路径导入其他包，需要在系统路径内加入当前py文件需导入的文件的相对路径（部分）
print(sys.path)

#sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append('../') #将当前文件所在的目录的上级目录加入sys，这是个全局的，当前文件的意思就是这个py文件，不是最初的启动文件
print(sys.path)
#相比于刚才增加了一个路径


#测试,路径里包含了test_path 的上级目录，所以直接导入下面的文件的话，系统会自己去找
import test_ml_predict

