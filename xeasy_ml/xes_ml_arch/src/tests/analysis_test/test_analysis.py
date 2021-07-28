# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

# while unittest , look at the 370(line) of analysis.py
import os
import unittest
import sys
import configparser
import random
import pandas as pd
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.analysis import analysis


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.conf = configparser.ConfigParser()
        self.conf_file = "analysis.conf"
        self.conf.read(self.conf_file)
        self.columns = ["col%s" % (x) for x in range(10)]
        self.x = self.columns[:9]
        self.y = self.columns[9]
        self.get_data()
        self.ins = analysis.Analysis(conf=self.conf, log_path = './config/log.conf' , data=self._data, feature_columns=self.x, label_columns=self.y)

    def test_execute(self):
        self.assertTrue(self.ins.execute())

    def test_reset(self):
        self.ins.reset(conf=self.conf, data=self._data, feature_columns=self.x,
                       label_columns=self.y)
        self.assertTrue(self.ins.execute())

    def get_data(self):
        self._data = pd.DataFrame(
            [[int(random.random() * 100) for x in range(10)] for y in range(10000)],
            columns=self.columns)
        self._data["col9"] = np.random.choice([0, 1], size=10000)

if __name__ == '__main__':
    unittest.main()
