# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import configparser
import unittest
import sys
import pandas as pd

sys.path.append("../../../../")
from xes_ml_arch.src.feature_enginnering import xes_onehot_encoder


class TestOneHot(unittest.TestCase):
    def setUp(self):
        self._data_path = "../data/test_onehot.txt"
        self._config = "./conf/test_onehot.conf"
        self._conf = configparser.ConfigParser()
        self._conf.read(self._config)
        self._config_no = "./conf/test_onehot_no.conf"
        # self._conf_no = configparser.ConfigParser()
        # self._conf_no.read(self._config_no)

        self._data = pd.read_csv(self._data_path)
        # self._data_no = pd.read_csv("../data/test_oneshot_none.txt")
        self._onehot_ins = xes_onehot_encoder.XESOneHotEncoder(data=self._data, conf=self._conf,log_path = '../log/log.conf')
        self._onehot_ins_no = xes_onehot_encoder.XESOneHotEncoder(data=pd.DataFrame(), conf=self._conf,log_path = '../log/log.conf')
        self._onehot_ins_ee = xes_onehot_encoder.XESOneHotEncoder(data=self._data, conf='',log_path = '../log/log.conf')
        self._onehot_ins_nn = xes_onehot_encoder.XESOneHotEncoder(data=self._data, conf=self._config_no,log_path = '../log/log.conf')

    def test_start(self):
        self.assertTrue(self._onehot_ins.init())
        self.assertTrue(self._onehot_ins.execute())
        self.assertFalse(self._onehot_ins_no.execute())
        self.assertFalse(self._onehot_ins_ee.execute())
        self.assertFalse(self._onehot_ins_nn.execute())

    def test_get_data(self):
        self._onehot_ins.reset(data=self._data, one_hot_file=self._conf)
        self._onehot_ins.execute()
        data = self._onehot_ins.get_data

    def test_reset(self):
        data_no = pd.DataFrame()
        self.assertFalse(self._onehot_ins.reset(data=data_no, one_hot_file=''))
        self.assertFalse(self._onehot_ins.reset(data=self._data, one_hot_file=''))


if __name__ == '__main__':
    unittest.main()
