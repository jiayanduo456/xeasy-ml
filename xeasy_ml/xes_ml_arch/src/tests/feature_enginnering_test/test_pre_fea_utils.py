# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import pandas as pd
import configparser
import numpy as np
import sys
import json

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import pre_feature_utils


class TestPreFeatureUtils(unittest.TestCase):
    def setUp(self):
        self.conf_1 = configparser.ConfigParser()
        self.conf = configparser.ConfigParser()
        self.conf_d = configparser.ConfigParser()
        self.ff_t = pre_feature_utils.PreFeatureUtils(conf=self.conf_1,log_path = '../log/log.conf')
        self.conf.read('./conf/feature_enginnering.conf')
        self.conf_d.read('./conf/feature_enginnering_no.conf')
        self.ff = pre_feature_utils.PreFeatureUtils(conf=self.conf,log_path = '../log/log.conf')
        self.ff_no = pre_feature_utils.PreFeatureUtils('./conf/feature_enginnering.conf',log_path = '../log/log.conf')
        self.data = pd.read_csv('data/featureFilter.csv')
        self.ff_d = pre_feature_utils.PreFeatureUtils(conf=self.conf, data=self.data,log_path = '../log/log.conf')
        self.ff_f = pre_feature_utils.PreFeatureUtils(conf=self.conf_d,log_path = '../log/log.conf')

    def test_init(self):
        # app = self.conf.get('pre_feature_utils', "single_feature_apply")
        self.assertTrue(self.ff.init())
        self.assertFalse(self.ff_no.init())
        self.assertFalse(self.ff_t.init())

    def test_setdata(self):
        a = [1, 2, 3, 4]
        self.assertFalse(self.ff.set_data(a))

    def test_start(self):
        self.assertTrue(self.ff.init())
        self.assertTrue(self.ff.set_data(self.data))
        res = self.ff.excute()
        print("res:", res)

    def test_start_a(self):
        self.assertTrue(self.ff_d.init())
        res2 = self.ff_d.excute()
        print("res2:", res2)

    def test_start_b(self):
        self.assertTrue(self.ff_f.init())
        res_no = self.ff_f.excute()
        print("res_no:", res_no)

    def test_start_c(self):
        self.assertTrue(self.ff_f.set_data(self.data))
        res1 = self.ff_f.excute()
        print("res1:", res1)


if __name__ == '__main__':
    unittest.main()
