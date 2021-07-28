# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import pandas as pd
import configparser
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import data_washer


class TestDataSampler(unittest.TestCase):
    def setUp(self):
        self._data_path = "../data/test.txt"
        self._config = "./conf/feature_enginnering.conf"
        self._data = pd.read_csv(self._data_path)
        self._conf = configparser.ConfigParser()
        self._conf_no = configparser.ConfigParser()
        self._conf_false = configparser.ConfigParser()
        self._conf.read(self._config)
        self._conf_no.read("./conf/feature_enginnering_no.conf")
        self._conf_false.read("./conf/feature_enginnering_false.conf")
        # self._conf1.read("./conf/test_discretize_try.conf")
        self._ins = data_washer.DataWasher(data=self._data, log_path = '../log/log.conf',conf=self._conf)
        self._ins_1 = data_washer.DataWasher(data=None, log_path = '../log/log.conf',conf=self._conf)

    def test_start(self):
        self.assertTrue(self._ins.init())
        self._ins.set_train_flag()
        self._ins.excute()
        self.assertTrue(self._ins_1.init())
        self._ins_1.set_data(self._data)
        self._ins.set_train_flag()
        self.assertIsNotNone(self._ins.excute())

    def test_get_data(self):
        self._ins.init()
        data = self._ins.get_data()

    def test_set_data(self):
        self._ins.init()
        self._ins.set_data(self._data)

    def test_set_flag(self):
        self._ins.init()
        self._ins.set_train_flag()
        self._ins.set_test_flag()

    def test_store_zscore_scale(self):
        self._ins._store_zscore_scale()

    def test_control_data(self):
        self._ins.control_data({"col1": None})

        self._ins.control_data({"col1": "fun_add_self"})

if __name__ == '__main__':
    unittest.main()
