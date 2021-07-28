# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import os
import shutil
import traceback
import configparser




class CreatNewDemo(object):
    """
    project_init
    when add a new project, run this to create a new project dir under demo
    """

    XES_FILE = 'xes_ml_arch'
    CONF_DIR = 'copy_dir'
    CONF_FILE = 'copy_file'
    PROJ = 'project'
    NEW_DIR = 'new_dir'

    def __init__(self, config, pro_path):
        """
        init config
        :param config: configure file
            copy_dir = config
            copy_file = __main__.py
            new_dir = data,result
        """

        self._model = None
        self._is_change = False
        self.pro_path = pro_path
        self._config = configparser.ConfigParser()
        self._config.read(config)

    def _create_new_dir(self, dir_name):
        """
        create a new dir under demo for a new project
        :param dir_name: string
            the name of new dir
        :return:
        """

        try:
            _dir = self._config.get(dir_name, self.CONF_DIR).replace(' ', '').split(',')
            _file = self._config.get(dir_name, self.CONF_FILE).replace(' ', '').split(',')
            _new = self._config.get(dir_name, self.NEW_DIR).replace(' ', '').split(',')
            self._path = os.path.dirname(__file__)
            file_path = self._path[:-13]
            # file_path = os.path.join(self._path, "monkey")
            _xes_dir = os.path.join(file_path, self.XES_FILE)
            # _src_dir = os.path.join(_xes_dir, self.SRC)
            _proj_dir = os.path.join(self.pro_path, self.PROJ)
            if not os.path.exists(_proj_dir):
                os.mkdir(_proj_dir)
            _new_dir = os.path.join(_proj_dir, dir_name)
            if not os.path.exists(_new_dir):
                os.mkdir(_new_dir)
                for mk in _new:
                    if not os.path.exists(os.path.join(_new_dir, mk)):
                        os.mkdir(os.path.join(_new_dir, mk))
            else:
                print("%s dir already exists. if you want to init this project, delete it frist" % (
                    _new_dir))
                return False

            for j in range(0, len(_file)):
                _tar_file = os.path.join(_new_dir, _file[j])
                shutil.copy(os.path.join(_xes_dir, _file[j]), _tar_file)

            for i in range(0, len(_dir)):
                _copy_dir = os.path.join(_xes_dir, _dir[i])
                _new_dir = os.path.join(_proj_dir, dir_name)
                _tar_dir = os.path.join(_new_dir, _dir[i])

                if os.path.exists(_copy_dir) and not os.path.exists(_tar_dir):
                    shutil.copytree(_copy_dir, _tar_dir)
                else:
                    print("{} is not exist! or new demo dir {} is exist! ".format(_xes_dir,_tar_dir))
            return True
        except:
            print("create new demo {} not succeed".format(dir_name))
            traceback.print_exc()
            return False

    def start(self):
        """
        project init start, create new dir based config file
        :return:
        """

        for key in self._config.sections():
            name = key.replace(' ', '')
            if self._create_new_dir(name):
                print("%s project init succeed" % name)
            else:
                print("%s project init failed" % name)

def create_project(pro_path):
    path = os.path.dirname(__file__)
    conf = os.path.join(path, 'create_demo.conf')
    ins = CreatNewDemo(conf, pro_path)
    ins.start()
    print("Succ init project.")

if __name__ == '__main__':
    conf = './create_demo.conf'
    #ins = CreatNewDemo(conf)
    #ins.start()
    pro_path = os.getcwd()
    create_project(pro_path)