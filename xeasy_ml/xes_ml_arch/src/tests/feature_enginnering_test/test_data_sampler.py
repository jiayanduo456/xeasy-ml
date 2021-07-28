# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import pandas as pd
import configparser
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import data_sampler


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
        self.target = np.random.randint(0, 4, 100)
        # self._conf1.read("./conf/test_discretize_try.conf")
        self._ins = data_sampler.DataSampler(data=self._data,log_path = '../log/log.conf',conf=self._conf)
        self._ins_1 = data_sampler.DataSampler(data=None, log_path = '../log/log.conf',conf=self._conf)
        self._ins_2 = data_sampler.DataSampler(data=self._data, log_path = '../log/log.conf',conf=self._conf, target='col9')
        self._ins_3 = data_sampler.DataSampler(data=self._data, log_path = '../log/log.conf',conf=self._conf_no, target='col9')
        self._ins_4 = data_sampler.DataSampler(data=self._data, log_path = '../log/log.conf',conf=self._conf_false, target='col9')

    def test_excute(self):
        self.assertTrue(self._ins.init())
        self.assertFalse(self._ins.excute())

        self.assertTrue(self._ins_1.init())
        self.assertFalse(self._ins_1.excute())

        self.assertTrue(self._ins_2.init())
        self.assertTrue(self._ins_2.excute())

        self.assertFalse(self._ins_3.init())
        self.assertFalse(self._ins_3.excute())

        self.assertTrue(self._ins_4.init())
        self.assertTrue(self._ins_4.excute())

    def test_get_data(self):
        self._ins_2.init()
        self._ins_2.excute()
        data = self._ins_2.get_data()
        self.assertIsNotNone(data)

    def test_set_data(self):
        _data = None
        self._ins_2.init()
        self.assertTrue(self._ins_2.set_data(self._data))
        self.assertFalse(self._ins_2.set_data(_data))
        self.assertFalse(self._ins_2.set_data(self.target))

    def test_set_target_field(self):
        col = 1
        self.assertTrue(self._ins_2.set_target_field('col1'))
        self.assertFalse(self._ins_2.set_target_field(col))


if __name__ == '__main__':
    unittest.main()
