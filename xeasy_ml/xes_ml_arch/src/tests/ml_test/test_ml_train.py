# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import configparser
import random
import pandas as pd
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.analysis import analysis
from xes_ml_arch.src.cross_validation import cross_validation
from xes_ml_arch.src.ml import train_model_ml
from xes_ml_arch.src.ml_utils import runstatus


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.conf = configparser.ConfigParser()
        self.conf_try = configparser.ConfigParser()
        self.conf_no = configparser.ConfigParser()
        self.conf_try1 = configparser.ConfigParser()
        self.conf_try2 = configparser.ConfigParser()
        self.conf_try3 = configparser.ConfigParser()

        self.conf_file = "./config/ml.conf"
        self.conf.read(self.conf_file)
        self.conf_try.read("./config/ml_try.conf")
        self.conf_no.read("./conf/ml_online.conf")

        self.conf_try1.read("./config/ml_try1.conf")
        self.conf_try2.read("./config/ml_try2.conf")
        self.conf_try3.read("./config/ml_try3.conf")

        self.columns = ["col%s" % x for x in range(10)]
        # self.x = self.columns[:9]
        # self.y = self.columns[9]
        self.get_data()
        self.ins = train_model_ml.TrainML(conf=self.conf,xeasy_log_path = '../log/log.conf')
        self.ins_try = train_model_ml.TrainML(conf=self.conf_try,xeasy_log_path = '../log/log.conf')
        self.ins_no = train_model_ml.TrainML(conf=self.conf_no,xeasy_log_path = '../log/log.conf')
        self.ins_try1 = train_model_ml.TrainML(conf=self.conf_try1,xeasy_log_path = '../log/log.conf')
        self.ins_try2 = train_model_ml.TrainML(conf=self.conf_try2,xeasy_log_path = '../log/log.conf')
        self.ins_try3 = train_model_ml.TrainML(conf=self.conf_try3,xeasy_log_path = '../log/log.conf')

    def test_set_data(self):
        self.assertTrue(self.ins.set_data(self._data))
        self.assertTrue(self.ins_try.set_data(self._data))

    def test_start(self):
        if self.ins.init() == runstatus.RunStatus.SUCC:
            self.assertTrue(self.ins.set_data(self._data))
            self.assertTrue(self.ins.start())
            self.assertTrue(self.ins.set_data(self._data_no))
            self.assertFalse(self.ins.start())
            self.assertTrue(self.ins.set_data(self._data_tt))
            self.assertFalse(self.ins.start())
        self.assertFalse(self.ins_try.init())
        self.assertFalse(self.ins_try.start())
        self.assertFalse(self.ins_no.init())
        self.assertFalse(self.ins_no.start())

    def test_init(self):
        assert (self.ins.init(), runstatus.RunStatus.SUCC)
        self.assertFalse(self.ins_try1.init())
        self.assertFalse(self.ins_try1.start())
        self.assertFalse(self.ins_try2.init())
        self.assertFalse(self.ins_try2.start())
        self.assertFalse(self.ins_try3.init())
        self.assertFalse(self.ins_try3.start())

    def test_set_analysis_ins(self):
        ana = analysis.Analysis(log_path=  '../log/log.conf')
        noa = None
        self.ins.set_analysis_ins(ana)
        try:
            self.ins.set_analysis_ins(noa)
        except:
            pass

    def test_set_cross_validation(self):
        cv = cross_validation.Cross_Validation(log_path=  '../log/log.conf')
        nocv = None
        self.ins.set_cross_validation(cv)
        try:
            self.ins.set_cross_validation(nocv)
        except:
            pass

    def test_init_cv_ins(self):
        self.assertTrue(self.ins._init_cv_ins())
        self.assertFalse(self.ins_no._init_cv_ins())

    def test_train_handle(self):
        self.ins = train_model_ml.TrainML(conf=self.conf,xeasy_log_path = '../log/log.conf')
        self.assertTrue(self.ins.init())
        self.assertTrue(self.ins.set_data(self._data))
        self.assertTrue(self.ins.start())

    def get_data(self):
        self._data = pd.DataFrame(
            [[int(random.random() * 100) for x in range(10)] for y in range(10000)],
            columns=self.columns)
        self._data["col9"] = np.random.choice([0, 1], size=10000)
        self._data_no = pd.DataFrame(
            [[int(random.random() * 100) for x in range(10)] for y in range(10000)],
            columns=self.columns)
        self._data_tt = pd.DataFrame(
            [[int(random.random() * 100) for x in range(4)] for y in range(10000)],
            columns=['col1', 'col2', 'col3', 'col4'])


if __name__ == '__main__':
    unittest.main()
