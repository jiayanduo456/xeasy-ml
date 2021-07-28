# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import configparser
import random
import pandas as pd
import numpy as np

sys.path.append("../../../../")

from xes_ml_arch.src.analysis import analysis
from xes_ml_arch.src.ml import prediction_ml
from xes_ml_arch.src.ml_utils import runstatus


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.conf = configparser.ConfigParser()
        self.conf_try = configparser.ConfigParser()
        self.conf_file = "./config/ml.conf"
        self.conf.read(self.conf_file)
        self.conf_try.read('./config/ml_try.conf')
        self.columns = ["col%s" % (x) for x in range(10)]
        self.x = self.columns[:9]
        self.y = self.columns[9]
        self.get_data()
        self.ins = prediction_ml.PredictionML(conf=self.conf, xeasy_log_path = '../log/log.conf')
        self.ins_no = prediction_ml.PredictionML(conf=self.conf_file,xeasy_log_path = '../log/log.conf')
        self.ins_try = prediction_ml.PredictionML(conf=self.conf_try,xeasy_log_path = '../log/log.conf')

    def test_set_data(self):
        self.assertTrue(self.ins.set_data(self._data))
        self.assertFalse(self.ins.set_data(self._data_no))
        self.ins_try.set_data(self._data)

    def test_start(self):
        self.ins.set_data(self._data)
        if self.ins.init() == runstatus.RunStatus.SUCC:
            self.ins.start()
        if self.ins_no.init() == runstatus.RunStatus.SUCC:
            self.ins_no.start()
        if self.ins_try.init() == runstatus.RunStatus.SUCC:
            self.ins_try.start()

    def test_init(self):
        assert (self.ins.init(), runstatus.RunStatus.SUCC)

    def get_data(self):
        self._data = pd.DataFrame(
            [[int(random.random() * 100) for x in range(10)] for y in range(10000)],
            columns=self.columns)
        self._data["col9"] = np.random.choice([0, 1], size=10000)
        self._data_no = np.random.randint(0, 5, 99)


if __name__ == '__main__':
    unittest.main()
