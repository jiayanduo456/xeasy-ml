# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import configparser
import random

import pandas
import pandas as pd
import numpy as np

sys.path.append("../../../..")

from xes_ml_arch.src.cross_validation import cross_validation, data_split
from xes_ml_arch.src.model import base_model, model_factory


class TestCrossVaidation(unittest.TestCase):
    def setUp(self):
        self.conf = configparser.ConfigParser()
        self.conf_file = "cross.conf"
        self.conf.read(self.conf_file)
        self.columns = ["col%s" % (x) for x in range(10)]
        self.x = self.columns[:9]
        self.y = self.columns[9]
        self._data = pd.DataFrame(
            [[int(random.random() * 100) for x in range(10)] for y in range(10000)],
            columns=self.columns)
        self._data["col9"] = np.random.choice([0, 1], size=10000)

        self.create_config()
        self.ins = cross_validation.Cross_Validation(conf=self._cv_config, log_path = '../log/log.conf',data=self._data,
                                                     x_columns=self.x, y_column=self.y)
        self.ins_no = cross_validation.Cross_Validation(conf=self._cv_config, log_path = '../log/log.conf',data=pandas.DataFrame(),
                                                        x_columns=self.x, y_column=self.y)
        self.split_ins = data_split.DataSplit(conf=self._data_spilt_config, log_path = '../log/log.conf', data=self._data)

    def test_data_split(self):
        self.assertTrue(self.split_ins.execute())
        train_data = self.split_ins.train_data
        test_data = self.split_ins.test_data
        # assert (train_data.shape[0] == 9000)
        # assert (test_data.shape[0] == 1000)
        self.assertEqual(train_data.shape[0], 9000)
        self.assertEqual(test_data.shape[0], 1000)
        self.assertTrue(self.split_ins.store_test_data())
        self.assertTrue(self.split_ins.store_train_data())
        self.assertTrue(self.split_ins._load_data())
        self.assertFalse(self.split_ins._store_data(self._data_spilt_config, self._data))
        self.split_ins.reset(None, None)
        self.split_ins.reset(self._data_spilt_config, self._data)

    def test_data_split_no(self):
        self.split_ins_no = data_split.DataSplit(conf=self._data_spilt_config_no, log_path = '../log/log.conf',data=self._data)
        self.split_ins_no.reset()
        self.assertTrue(self.split_ins_no.execute())
        try:
            train_data = self.split_ins_no.train_data
            test_data = self.split_ins_no.test_data
            self.assertTrue(self.split_ins.store_test_data())
            self.assertTrue(self.split_ins.store_train_data())
        except:
            pass

    def test_cv_model(self):
        model = model_factory.ModelFactory.create_model(self._model_config,log_path = '../log/log.conf',
                                                        model_name= base_model.BaseModel.MODEL_SKLEARN_XGB)
        self.assertTrue(self.ins.execute())
        self.assertTrue(self.ins.cv_model(model))
        self.ins.reset(self._cv_config, self._data, 'col1', 'col2')

    def test_cv_model_1(self):
        model = None
        self.assertFalse(self.ins_no.execute())
        self.assertFalse(self.ins.cv_model(model))

    def create_config(self):
        self._model_config = configparser.ConfigParser()
        self._data_spilt_config = configparser.ConfigParser()
        self._cv_config = configparser.ConfigParser()
        self._data_spilt_config_no = configparser.ConfigParser()
        self._data_spilt_config.read("./data_split.conf")
        self._model_config.read("../../../config/demo/model_online.conf")
        self._cv_config.read("./cross_validation.conf")
        self._data_spilt_config_no.read("./data_split_try.conf")


if __name__ == '__main__':
    unittest.main()
