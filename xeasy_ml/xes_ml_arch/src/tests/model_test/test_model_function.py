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
from xeasy_ml.xes_ml_arch.src.model import lr
from xeasy_ml.xes_ml_arch.src.model import rf
from xeasy_ml.xes_ml_arch.src.model import linear
from xeasy_ml.xes_ml_arch.src.model import my_xgb
from xeasy_ml.xes_ml_arch.src.model import sklearn_xgb
from xeasy_ml.xes_ml_arch.src.model import desion_tree
from xeasy_ml.xes_ml_arch.src.model import my_lightgbm
from xeasy_ml.xes_ml_arch.src.model import sklearn_xgb_reg



class TestModelFactory(unittest.TestCase):
    def setUp(self):
        self.conf_file = configparser.ConfigParser()
        self.conf_file.read('../config/model.conf')
        self.conf_file_lr = "../config/xiaozhuan_prediction_lr.conf"
        self.conf_file_rf = "../config/xiaozhuan_prediction_rf.conf"
        self.conf_file_xgb = "../config/xiaozhuan_prediction_xgb.conf"
        self.conf_file_sklearn_xgb = "../config/xiaozhuan_prediction_xgb.conf"
        self.conf_file_line_reg = "../config/xiaozhuan_prediction_linereg.conf"
        self.x = pd.read_csv('../data/test_aaa_test.txt', sep=',')
        self.path = '../log/log.conf'
        self.columns = self.x.columns
        # self.y = pd.DataFrame([int(random.random() * 100) for x in range(999)])
        self._y = np.array([int(random.random() * 100) for x in range(99)])
        self._y.reshape(1, 99)
        self.y = np.random.randint(0,3,size=(99,))
        self.y.reshape(1, 99)

    def test_model_lr(self):
        test_lr = lr.LR(self.conf_file, log_path = self.path)
        if test_lr._init_model():
            self.assertTrue(test_lr.train(self.x, self._y))

    def test_model_rf(self):
        test_rf = rf.RF(self.conf_file, log_path = self.path)
        test_rf._init_model()
        self.assertTrue(test_rf.train(self.x, self._y))
        self.assertTrue(test_rf.get_feature_importance(self.columns))

    def test_model_line_reg(self):
        test_linear = linear.Liner(self.conf_file, log_path = self.path)
        test_linear._init_model()
        self.assertTrue(test_linear.train(self.x, self._y))
        res = test_linear.get_feature_importance(self.x)
        self.assertIsNotNone(res)

    def test_model_xgb(self):
        test_lightgbm = my_xgb.MyXgb(self.conf_file, log_path = self.path)
        test_lightgbm._init_model()
        self.assertTrue(test_lightgbm.train(self.x, self._y))
        res = test_lightgbm.predict(self.x)
        self.assertIsNotNone(res)
        # print res
        self.assertIsNotNone(test_lightgbm.predict_proba(self.x))
        # print res1
        self.assertIsNotNone(test_lightgbm.get_feature_importance(self.x))

    def test_model_sklearn_xgb(self):
        test_model_sklearn_xgb = sklearn_xgb.SklearnXGB(self.conf_file, log_path = self.path)
        test_model_sklearn_xgb._init_model()
        self.assertTrue(test_model_sklearn_xgb.train(self.x, self.y))
        res = test_model_sklearn_xgb.predict(self.x)
        self.assertIsNotNone(res)
        # print res
        res1 = test_model_sklearn_xgb.predict_proba(self.x)
        self.assertIsNotNone(res1)
        res2 = test_model_sklearn_xgb.get_feature_importance(self.x)
        self.assertIsNotNone(res2)

    def test_model_dst(self):
        test_model_dst = desion_tree.MyDesionTree(self.conf_file, log_path = self.path)
        test_model_dst._init_model()
        self.assertTrue(test_model_dst.train(self.x, self._y))
        res = test_model_dst.get_feature_importance(self.x)
        self.assertIsNotNone(res)
        try:
            test_model_dst.show_dot('./desion_tree', res)
        except:
            traceback.print_exc()

    def test_model_lightgbm(self):
        test_lightgbm = my_lightgbm.MyLightGBM(self.conf_file, log_path = self.path)
        test_lightgbm._init_model()
        self.assertTrue(test_lightgbm.train(self.x, self._y))
        res = test_lightgbm.predict(self.x)
        self.assertIsNotNone(res)
        # print res
        self.assertIsNotNone(test_lightgbm.predict_proba(self.x))
        # print res1
        self.assertIsNotNone(test_lightgbm.get_feature_importance(self.x))

    def test_model_sklearn_xgb_reg(self):
        test_model_sklearn_xgb = sklearn_xgb_reg.SklearnXGBReg(self.conf_file, log_path = self.path)
        test_model_sklearn_xgb._init_model()
        self.assertTrue(test_model_sklearn_xgb.train(self.x, self.y))
        res = test_model_sklearn_xgb.predict(self.x)
        self.assertIsNotNone(res)
        # print res
        res1 = test_model_sklearn_xgb.predict_proba(self.x)
        self.assertIsNotNone(res1)
        res2 = test_model_sklearn_xgb.get_feature_importance(self.x)
        self.assertIsNotNone(res2)



if __name__ == '__main__':
    unittest.main()
