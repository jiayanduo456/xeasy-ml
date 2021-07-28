# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import unittest
import sys
import random
import pandas as pd

sys.path.append('../../../..')
from xes_ml_arch.src.ml_utils import feature_processor


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.covertint = 3
        self.covertflo = 3.3
        self.covertstr = '3.3'
        self.covertbool = (6,7)

        self.time1 = pd.Series(pd.date_range(start='2019-1-09',periods=10,freq='H'))
        self.time2 = pd.Series(pd.date_range(start='2017-2-21',periods=10,freq='H'))
        self.stamp = pd.Series(feature_processor.FeatureProcessor().time2stamp(self.time2))

        self.year = ['2020+','2031+']
        self.age = ['28-30','28+3']

        self.disli = [random.randint(1,15) for _ in range(100)]
        self.day = 100000

    def test_cover2int(self):
        resa = feature_processor.FeatureProcessor().convert_to_int(self.covertint)
        resb = feature_processor.FeatureProcessor().convert_to_int(self.covertflo)
        resc = feature_processor.FeatureProcessor().convert_to_int(self.covertstr)
        self.assertTrue(isinstance(resa, int))
        self.assertTrue(isinstance(resb, int))
        self.assertEqual(resc,0)

    def test_cover2str(self):
        #resa = feature_processor.FeatureProcessor().convert_to_str(self.covertbool)
        resb = feature_processor.FeatureProcessor().convert_to_str(self.covertflo)
        resc = feature_processor.FeatureProcessor().convert_to_str(self.covertstr)
        self.assertTrue(isinstance(resc, str))
        self.assertTrue(isinstance(resb, str))
        #self.assertEqual(resa, '')

    def test_time2stamp(self):
        resa = feature_processor.FeatureProcessor().time2stamp(self.time1)
        self.assertIsNotNone(resa)

    def test_stamp2time(self):
        resa = feature_processor.FeatureProcessor().stamp2time(self.stamp)
        self.assertIsNotNone(resa)

    def test_timediff(self):
        _diff = feature_processor.FeatureProcessor().time_diff(self.time1,self.time2)
        self.assertIsNotNone(_diff)

    def test_minusdata(self):
        _res = feature_processor.FeatureProcessor().minus_data(self.time1, self.time2)
        self.assertIsNotNone(_res)

    def test_absminusdata(self):
        _res = feature_processor.FeatureProcessor().abs_minus_data(self.time1, self.time2)
        self.assertIsNotNone(_res)

    def test_stayinyear(self):
        res = feature_processor.FeatureProcessor().stay_in_year(self.year)
        self.assertIsNotNone(res)

    def test_age2int(self):
        res = feature_processor.FeatureProcessor().age2int(self.age)
        self.assertNotEqual(res,[-1 for _ in range(len(self.age))])

    def test_discretizefreque(self):
        res = feature_processor.FeatureProcessor().discretize_freque(self.disli)
        self.assertIsNotNone(res)
    def test_getdayrange(self):
        ans = res = feature_processor.FeatureProcessor().get_day_range(self.day)
        self.assertTrue(res > 0)


if __name__ == '__main__':
    unittest.main()
