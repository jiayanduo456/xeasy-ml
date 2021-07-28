# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import pandas as pd
import configparser
import copy

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import feature_discretizer


class TestDiscretize(unittest.TestCase):
    def setUp(self):
        self._data_path = "../data/test_discretize.txt"
        self._config = "./conf/test_discretize.conf"
        self._data = pd.read_csv(self._data_path)
        self._conf = configparser.ConfigParser()
        self._conf1 = configparser.ConfigParser()
        self._conf.read(self._config)
        self._conf1.read("./conf/test_discretize_try.conf")
        self._ins = feature_discretizer.FeatureDiscretizer(data=self._data, conf=self._conf, log_path = '../log/log.conf')
        self._ins_try = feature_discretizer.FeatureDiscretizer(data=None, conf=self._conf1,log_path = '../log/log.conf')

    def test_excute(self):
        self.assertTrue(self._ins.excute())
        self.assertFalse(self._ins_try.excute())

    def test_get_data(self):
        self._ins.reset(data=self._data, conf=self._conf)
        tmp_data = copy.copy(self._data)
        self._ins.excute()
        data = self._ins.get_data
        # print data
        # print tmp_data
        res = pd.merge(data, tmp_data, on=["userid"], suffixes=["_dis", "_old"])
        print("res:\n", res)
        res_diff = res[res["age_dis"] != res["targ3_dis"]]
        print("res diff:\n", res_diff)


if __name__ == '__main__':
    unittest.main()
