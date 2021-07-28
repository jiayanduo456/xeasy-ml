# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import unittest
import sys
import random
import configparser
import pandas as pd
import numpy as np
import traceback


sys.path.append('../../../..')
from xes_ml_arch.src.model import model_factory


class TestModelFactory(unittest.TestCase):
    def setUp(self):
        self.model_list = ['lr', 'rf', 'line_reg', 'xgb', 'sklearn_xgb', 'dst', 'lightgbm']
        self.conf_file_lr = "../config/xiaozhuan_prediction_lr.conf"
        self.conf_file_rf = "../config/xiaozhuan_prediction_rf.conf"
        self.conf_file_xgb = "../config/xiaozhuan_prediction_xgb.conf"
        self.conf_file_sklearn_xgb = "../config/xiaozhuan_prediction_sklearn_xgb.conf"
        self.conf_file_line_reg = "../config/xiaozhuan_prediction_linereg.conf"
        self.conf_file_lightgbm = "../config/xiaozhuan_prediction_lightgbm.conf"
        self.x = pd.read_csv('../data/test_aaa_test.txt', sep=',')
        self.n_x = None
        self.n_y = None
        self.e_x = pd.read_csv('../data/test_aaa_test.txt', sep=',')
        # self.y = pd.DataFrame([int(random.random() * 100) for x in range(999)])
        self._y = np.random.randint(0,3,size=(99,))
        self._y.reshape(1, 99)
        self.y = np.array([random.random() for x in range(99)])
        #print(self._y,self.y)
        self.y.reshape(1, 99)

    def test_model(self):
        conf_file = configparser.ConfigParser()
        conf_file.read('../config/model.conf')
        for name in self.model_list:
            # file = 'self.conf_file_' + name
            # conf_file.read(eval(file))

            # print type(conf_file)
            # print conf_file.sections()
            model = model_factory.ModelFactory.create_model(conf_file, name, log_path = '../log/log.conf')
            if name == 'xgb':
                self.assertTrue(model.train(self.x, self.y))
                print("xgb init successed")
            else:
                print("%s 开始测试" % (name))
                self.assertTrue(model.train(self.x, self._y))

            self.assertIsNotNone(model.predict(self.x))
            if name == 'line_reg':
                pass
            else:
                self.assertIsNotNone(model.predict_proba(self.x))
            self.assertIsNotNone(model.get_feature_importance(self.x))
            self.assertTrue(model.store_model('../config/zscal_pickle_file_lr.properties'))
            self.assertTrue(model.load_model('../config/zscal_pickle_file_lr.properties'))
            if name == "lr":
                self.assertTrue(model.store_feature_importance(self.x))
                print("lr init successed")
            if name == "dst":
                model.show_dot('./dst', self.x)
                print("决策树流程结构图创建")

    def test_model_exception(self):
        conf_file = configparser.ConfigParser()
        conf_file.read("../config/model_try.conf")
        nofile = None
        feature = None
        model1 = model_factory.ModelFactory.create_model(conf_file, log_path = '../log/log.conf')
        for name in self.model_list:
            print('name: --------',name)
            model = None
            # file = 'self.conf_file_' + name
            # conf_file.read(eval(file))
            try:
                if name not in ['line_reg','xgb']:
                    model = model_factory.ModelFactory.create_model(conf_file, name, log_path= '../log/log.conf')
                    self.assertFalse(model.train(self.e_x, self.y))
                    self.assertFalse(model.load_model('../config/zscal_pickle_file'+'_%s.pickle'%(name)))
                    self.assertFalse(model.store_feature_importance(self.n_x))
                    self.assertIsNone(model.predict(self.x))
                    self.assertIsNone(model.get_feature_importance(feature))
                    self.assertTrue(model.store_model(nofile))
                else:
                    #line_reg No matter what the data format is, it can be trained successfully
                    #xgb obj:binary  train false while y is Continuous categories（int）
                    model = model_factory.ModelFactory.create_model(conf_file, name, log_path= '../log/log.conf')
                    self.assertFalse(model.train(self.e_x, self._y))
                    self.assertFalse(model.load_model('../config/zscal_pickle_file' + '_%s.pickle' % (name)))
                    self.assertFalse(model.store_feature_importance(self.n_x))
                    self.assertIsNone(model.predict(self.x))
                    self.assertIsNone(model.get_feature_importance(feature))
                    self.assertTrue(model.store_model(nofile))
                if name == 'line_reg':
                    pass
                else:
                    self.assertIsNone(model.predict_proba(self.x))
            except:
                traceback.print_exc()
            # model.get_feature_importance(self.n_x)

            try:
                #test create_model function test: the parameter passed in when creating a model can only be config
                model = model_factory.ModelFactory.create_model('../config/zscal_pickle_file', name)
            except:
                traceback.print_exc()

            # model = model_factory.ModelFactory.create_model(nofile, name)
        try:
            model = model_factory.ModelFactory.create_model(conf_file, 'sklearn_xgb', log_path= '../log/log.conf')
        except:
            traceback.print_exc()


if __name__ == '__main__':
    unittest.main()
