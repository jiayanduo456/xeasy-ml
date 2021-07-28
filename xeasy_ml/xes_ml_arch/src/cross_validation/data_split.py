# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import traceback
import pandas as pd
from ..systemlog import sysmanagerlog, syserrorlog
from ..ml_utils import runstatus
from ..ml_utils import pre_utils


class DataSplit(object):
    """
    This class is used to split a data set into train and test data set.

    Parameters
    --------
    conf: configparser.ConfigParser, default = None
        Configuration file for data set division.
    data: pandas.DataFrame, default = None
        Data set need split.

    Attributes
    --------
    _is_executed: bool, default = False
         The flag of the data set dividing.

    Examples
    --------
    >>> from xes_ml_arch.src.cross_validation import data_split
    >>> from xes_ml_arch.src.model import model_factory
    >>> import configparser
    >>> import pandas as pd
    >>> import numpy as np
    >>> import random
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> columns = ["col%s" % x for x in range(10)]
    >>> x = columns[:9]
    >>> y = columns[9]
    >>> data = pd.DataFrame(
    >>>           [[int(random.random() * 100) for _x in range(10)] for _y in range(1000)],
    >>>             columns=columns)
    >>> data["col9"] = np.random.choice([0, 1], size=1000)
    >>> ins = data_split.DataSplit(conf=conf, data=data)
    >>> ins.execute()
    >>> train_data = ins.train_data
    >>> test_data = ins.test_data
    >>> ins.store_train_data()
    >>> ins.store_test_data()
    """

    DATA_SPLIT = "data_split"
    RATIO = "test_ratio"
    TRAIN_FILE = "train_file"
    TEST_FILE = "test_file"
    LOAD_DATA_FROM_LOCAL_FILE = "load_data_from_local_file"
    TRUE = "true"

    def __init__(self, conf=None, log_path = None, data=None):
        self._conf = conf
        self.xeasy_log_path = log_path
        self._data = data
        self._train_data = None
        self._test_data = None
        self._is_executed = False

        self.managerlogger = sysmanagerlog.SysManagerLog(__file__, self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__, self.xeasy_log_path)

    def reset(self, conf=None, data=None):
        """
        Reset config and data.

        Parameters
        --------
        conf: configparser.ConfigParser, default = None
            Configuration file for data set division.
        data: pandas.DataFrame, default = None
            Data set need split.
        """
        if conf is not None:
            self._conf = conf
        if data is not None:
            self._data = data
        self._is_executed = False

    def execute(self):
        """
        Start dividing the data set.

        Returns
        --------
        Bool: runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if self._is_executed:
            return runstatus.RunStatus.SUCC
        if not pre_utils.PredictUtils.valid_pandas_data(self._data):
            return runstatus.RunStatus.FAILED
        try:
            if self.LOAD_DATA_FROM_LOCAL_FILE in self._conf.options(self.DATA_SPLIT):
                if self._conf.get(self.DATA_SPLIT,
                                  self.LOAD_DATA_FROM_LOCAL_FILE).lower() == self.TRUE:
                    return self._load_data()
            self._shuf_data()
            # Dividing
            self._split_data()
            self._is_executed = True
            self.store_train_data()
            self.store_test_data()
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.errorlogger.logger.error("cross validation error\n" + traceback.format_exc())
            self.managerlogger.logger.error("cross validation error: %s" % err)
            return runstatus.RunStatus.FAILED

    @property
    def train_data(self):
        """
        Get train data as property.

        Examples
        --------
        usage: ds = data_split.DataSplit()
               train_data =ds.train_data

        Returns
        --------
        self._train_data : pandas.DatFrame
        """
        if pre_utils.PredictUtils.valid_pandas_data(self._train_data):
            return self._train_data
        else:
            raise TypeError("split train data error")

    @property
    def test_data(self):
        """
        Get test data as property.

        Examples
        --------
        usage: ds = data_split.DataSplit()
               test_data =ds.test_data

        Returns
        --------
        self._test_data : pandas.DateFrame
        """
        if pre_utils.PredictUtils.valid_pandas_data(self._test_data):
            return self._test_data
        else:
            raise TypeError("split test data error")

    def store_train_data(self):
        """Store the divided train data set to file."""

        return self._store_data(self.TRAIN_FILE, self._train_data)

    def store_test_data(self):
        """Store the divided test data set to file."""

        return self._store_data(self.TEST_FILE, self._test_data)

    def _split_data(self):
        """Divide orignal data into test data and train data."""
        # get sample ratio
        try:
            ratio = float(self._conf.get(self.DATA_SPLIT, self.RATIO))
        except TypeError:
            self.managerlogger.logger.info(
                "%s not found %s, use default value 0.1" % (self.DATA_SPLIT, self.RATIO))
            ratio = 0.1
        if ratio > 1 or ratio < 0:
            raise ValueError("sample ratio=%s, but range (0, 1) is allowed" % (ratio))
        cure = int(self._data.shape[0] * ratio)
        self._test_data = self._data.iloc[:cure]
        self._train_data = self._data.iloc[cure:]

    def _shuf_data(self):
        """Shuffle data."""
        # self._data = shuffle(self._data)
        self._data = self._data.sample(frac=1)

    def _store_data(self, result_conf, data):
        """
        Store data to file.

        Parameters
        --------
        result_conf: str
            File path in configuration file.
        data: pandas.DataFrame
            Data need to store.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if not pre_utils.PredictUtils.valid_pandas_data(data):
            self.managerlogger.logger.error("train data is null")
            return runstatus.RunStatus.FAILED
        try:
            resutl_path = self._conf.get(self.DATA_SPLIT, result_conf)
        except Exception as err:
            self.managerlogger.logger.error("result path %s is not exist, please check out" % err)
            return runstatus.RunStatus.FAILED

        data.to_csv(resutl_path, index=False, encoding='utf8')
        return runstatus.RunStatus.SUCC

    def _load_data(self):
        """
        Load data from local file system.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            # get file path
            test_data_file = self._conf.get(self.DATA_SPLIT, self.TEST_FILE)
            train_data_file = self._conf.get(self.DATA_SPLIT, self.TRAIN_FILE)
            # read_file
            self._train_data = pd.read_csv(train_data_file)
            self._test_data = pd.read_csv(test_data_file)
            self._is_executed = True
            self.managerlogger.logger.info("load data success")
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("load error: %s" % err)
            self.errorlogger.logger.error("data split load data error! \n" + traceback.format_exc())
            return runstatus.RunStatus.FAILED
