# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import configparser
import unittest
import numpy as np
import pandas as pd
import sys

sys.path.append("../../../..")

from xes_ml_arch.src.feature_enginnering import feature_filter


class TestFeatureFilter(unittest.TestCase):
    def setUp(self):
        # before each test case
        self.data_tt = pd.read_csv("data/featureFilter.csv")
        self.conf_1 = configparser.ConfigParser()
        self.conf = configparser.ConfigParser()
        self.ff_t = feature_filter.FeatureFilter(self.conf_1, log_path = '../log/log.conf')
        self.conf.read('./conf/feature_enginnering.conf')
        self.ff = feature_filter.FeatureFilter(self.conf, log_path = '../log/log.conf')
        self.ff_no = feature_filter.FeatureFilter('./conf/feature_enginnering.conf',log_path = '../log/log.conf')

    def test_start(self):
        self.assertTrue(self.ff.init())
        self.assertTrue(self.ff.set_data(data=self.data_tt))
        self.assertIsNotNone(self.ff.excute())

        self.assertFalse(self.ff_no.init())
        self.assertTrue(self.ff_no.set_data(data=self.data_tt))
        self.ff_no.excute()

        self.assertFalse(self.ff_t.init())
        self.ff_t.set_data(data=self.data_tt)
        self.ff_t.excute()

    def test_get_data(self):
        self.ff.set_data(data=self.data_tt)
        data = self.ff.get_data()
        self.assertIsNotNone(data)

    def test_set_data(self):
        data_no = None
        self.assertTrue(self.ff.init())
        self.assertTrue(self.ff.set_data(self.data_tt))
        data = self.ff.get_data()
        self.assertFalse(self.ff.set_data(data=data_no))
        self.assertIsNotNone(self.ff.excute())

    def test_label_fields(self):
        self.ff.set_data(self.data_tt)
        self.ff.label_fields()

    def test_store_label_properties(self):
        data_no = None
        self.ff._store_label_properties(data_no)

    def test_reset(self):
        self.ff.reset(data=self.data_tt, del_list=['col1'], del_keywords='col2',
                      label_list=np.random.randint(0, 2, 99),
                      selected_list=['col3', 'col4'])


if __name__ == '__main__':
    unittest.main()
