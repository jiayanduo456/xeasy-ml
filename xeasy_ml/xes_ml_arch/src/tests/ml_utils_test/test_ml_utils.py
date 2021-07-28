# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import random
import pandas as pd
import numpy as np

sys.path.append('../../../..')
from xes_ml_arch.src.ml_utils import pre_utils
from xes_ml_arch.src.ml_utils import feature_processor
from xes_ml_arch.src.ml_utils import jsonmanager


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.pre_utils = pre_utils.PredictUtils()
        self.int_label = np.random.randint(0, 9, 99)
        self.label = np.random.randint(0, 9, 99)
        self.long_label = np.random.randint(0, 9, 100)
        self.x = [[random.random() * 100 for x in range(100)] for y in range(30)]
        self.bi_label = np.random.randint(0, 2, 99)
        self.multi_label = np.random.randint(0, 5, 99)
        self.bi_pre_label = np.random.randint(0, 2, 99)
        self.prob_label = np.random.uniform(0, 1, size=(99, 2))
        self.multi_prob_label = np.random.uniform(0, 1, size=(99, 5))
        self.prob_long_label = np.random.uniform(0, 1, size=(100, 2))
        self.prob_label.tolist()
        # self.no_list_x = pd.DataFrame(np.random.randn(2, 4))
        self.no_label = None

    def test_feature_processor(self):
        at = 1
        bt = 1.0
        ct = [1.0]
        resa = feature_processor.FeatureProcessor().convert_to_int(at)
        resb = feature_processor.FeatureProcessor().convert_to_int(bt)
        self.assertTrue(isinstance(resa, int))
        self.assertTrue(isinstance(resb, int))
        resc = feature_processor.FeatureProcessor().convert_to_int(ct)
        self.assertEqual(resc, 0)

    def test_jsonmanager(self):
        test_dic = {"a": 1,  "b": 2,  "d": 4,  "c": 3,  "e": 5,  'f': 'ewd see'}
        res1 = jsonmanager.get_message(test_dic)
        print(res1)
        self.assertTrue(isinstance(res1, str))
        res2 = jsonmanager.get_message_without_whitespace(test_dic)
        print(res2)
        self.assertTrue(isinstance(res1, str))

    def test_store_feature_importance(self):
        no_x = None
        res = self.pre_utils.store_feature_importance(self.x, '../data/test_store_feature_importance.txt')
        self.assertIsNotNone(res)
        try:
            res_no = self.pre_utils.store_feature_importance(no_x, '../data/test_store_feature_importance.txt')
        except:
            res_no = None

    def test_get_precision(self):
        res1 = self.pre_utils.get_precision(self.label, self.int_label)
        self.assertIsNotNone(res1)
        res2 = self.pre_utils.get_precision(list(self.label), list(self.int_label))
        self.assertIsNotNone(res2)

        try:
            self.pre_utils.get_precision(self.no_label, self.label)
        except:
            print("precison:label type can't be converted to a list")
        try:
            self.pre_utils.get_precision(self.label, self.no_label)
        except:
            print("precison:label type can't be converted to a list")
        try:
            self.pre_utils.get_precision(self.label, self.long_label)
        except:
            print("precison: length of label and pre_label is not equal")

    def test_get_model_score(self):
        res1 = self.pre_utils.get_model_score(self.label, self.int_label)
        self.assertTrue(isinstance(res1, dict))
        res2 = self.pre_utils.get_model_score(self.bi_label, self.bi_pre_label)
        self.assertTrue(isinstance(res2, dict))
        res3 = self.pre_utils.get_model_score(list(self.bi_label), list(self.bi_pre_label))
        self.assertTrue(isinstance(res3, dict))
        try:
            self.pre_utils.get_model_score(self.label, self.long_label)
        except:
            print("model_score: length of label and pre_label is not equal")
        try:
            self.pre_utils.get_model_score(self.no_label, self.label)
        except:
            print("model_score:label type can't be converted to a list")
        try:
            self.pre_utils.get_model_score(self.label, self.no_label)
        except:
            print("model_score:label type can't be converted to a list")

    def test_get_roc_score(self):
        res1 = self.pre_utils.get_roc_score(self.bi_label, self.prob_label)
        self.assertTrue(isinstance(res1, dict))
        res2 = self.pre_utils.get_roc_score(list(self.bi_label), list(self.prob_label))
        self.assertTrue(isinstance(res2, dict))
        res3 = self.pre_utils.get_roc_score(self.bi_label, np.array(self.prob_label))
        self.assertTrue(isinstance(res3, dict))
        res4 = self.pre_utils.get_roc_score(self.multi_label, np.array(self.multi_prob_label))
        self.assertTrue(isinstance(res4, dict))
        self.pre_utils.get_roc_score(self.multi_label, np.random.uniform(0, 1, size=(99, 5)))

        try:
            self.pre_utils.get_roc_score(self.bi_label, np.array(self.prob_long_label))
        except:
            print("roc_socore:length of label and pre_label is not equal")

        try:
            self.pre_utils.get_roc_score(self.multi_label, np.array(self.prob_label))
        except:
            print("roc_socore:length of label and pre_label is not equal")
        try:
            self.pre_utils.get_roc_score(self.no_label, self.label)
        except:
            print("roc_socore:label type can't be converted to a list")
        try:
            self.pre_utils.get_roc_score(self.label, self.no_label)
        except:
            print("roc_socore:label type can't be converted to a list")


if __name__ == '__main__':
    unittest.main()
