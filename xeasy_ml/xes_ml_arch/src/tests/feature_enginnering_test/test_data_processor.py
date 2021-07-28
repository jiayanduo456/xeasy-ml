# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import pandas as pd
import configparser
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import data_processor


class TestDataProcessor(unittest.TestCase):
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
        self.feature_processor = data_processor.DataProcessor(conf=self._conf, log_path = '../log/log.conf')
        self.feature_processor_none = data_processor.DataProcessor(conf=self._conf, log_path = '../log/log.conf')
        self.feature_processor_no = data_processor.DataProcessor(conf=self._conf_no, log_path = '../log/log.conf')

    def test_start(self):
        self.assertTrue(self.feature_processor.init())
        self.feature_processor.train_data = self._data
        self.feature_processor.test_data = self._data
        self.assertTrue(self.feature_processor.execute())

    def test_start_false(self):
        self.assertTrue(self.feature_processor.init())
        self.assertFalse(self.feature_processor.execute())
        self.assertFalse(self.feature_processor_no.init())
        self.assertFalse(self.feature_processor.execute())
        self.assertTrue(self.feature_processor_none.init())
        self.assertFalse(self.feature_processor.execute())

    # def test_set_data(self):
    #     self.assertTrue(self.feature_processor.init())
    #     self.assertTrue(self.feature_processor.set_data(self._data))
    #     data_no = None
    #     self.assertFalse(self.feature_processor.set_data(data_no))


if __name__ == '__main__':
    unittest.main()
