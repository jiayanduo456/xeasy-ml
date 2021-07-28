# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import traceback
import configparser
from ..ml_utils import pre_utils
from ..systemlog import sysmanagerlog, syserrorlog
from ..ml_utils import runstatus


class PreFeatureUtils(object):
    """
    Data Feature preprocessing

    Parameters
    --------
    data : pandas.DataFrame
    conf : configparser.ConfigParser

    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import pre_feature_utils
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("pre_feature.conf")
    >>> data = pd.read_csv("my_data.csv")
    # this
    >>> ins = pre_feature_utils.PreFeatureUtils(data=data, conf=conf)
    >>> ins.init()
    >>> ins.excute()
    # or this
    >>> ins = pre_feature_utils.PreFeatureUtils(conf=conf)
    >>> ins.init()
    >>> ins.set_data(data)
    >>> ins.excute()

    pre_feature.conf
    ----------------
    [pre_feature_utils]
    pre_flag = True    # if use Feature preprocessing or not
    single_feature_apply = {"col0":"time2stamp", "col1":"stamp2time"}
    multi_feature_apply = {'col0-col1':'minus_data', 'col2-col3':'abs_minus_data'}
    # which_feature_apply = {col_name: function name in utils}
    """

    FEATURE_PROC = "feature_processor.FeatureProcessor."
    PRE_FEATURE = 'pre_feature_utils'
    PRE_FLAG = 'pre_flag'
    SINGLE_APP = 'single_feature_apply'
    MULTI_APP = 'multi_feature_apply'
    DASH = ','

    def __init__(self, data=None, conf=None, log_path = None):
        self._conf = conf
        self.xeasy_log_path = log_path
        self._data = data
        self._single_app = {}
        self._multi_app = {}
        self._feature_function = ''
        self._pre_flag = False
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Init feature pre utils object.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            if not isinstance(self._conf, configparser.ConfigParser):
                self.managerlogger.logger.error("init pre feature_utils error: config is illegal")
                return runstatus.RunStatus.FAILED

            self._pre_flag = eval(self._conf.get(self.PRE_FEATURE, self.PRE_FLAG))
            if not self._pre_flag:
                return runstatus.RunStatus.SUCC
            if self.SINGLE_APP in self._conf.options(self.PRE_FEATURE):
                self._single_app = eval(self._conf.get(self.PRE_FEATURE, self.SINGLE_APP))
            if self.MULTI_APP in self._conf.options(self.PRE_FEATURE):
                self._multi_app = eval(self._conf.get(self.PRE_FEATURE, self.MULTI_APP))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("feature pre process init error: %s" % err)
            self.errorlogger.logger.error("feature pre process init error:\n %s" % traceback.format_exc())
        return runstatus.RunStatus.FAILED

    def set_data(self, data):
        """
        Set data.

        Parameters
        --------
        data: pandas.DataFrame
            data to feature pre,

        Returns
        --------
        :return: bool
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: False
        """
        if not pre_utils.PredictUtils.valid_pandas_data(data):
            self.managerlogger.logger.error("feature pre process set data error")
            return runstatus.RunStatus.FAILED
        self._data = data
        return runstatus.RunStatus.SUCC

    def excute(self):
        '''
        Excute.

        Returns
        --------
        self._data: processed data
        '''
        if not self._init():
            return None
        if not self._pre_flag:
            self.managerlogger.logger.info("no need pre feature processer")
            return self._data
        return self.pre_feature_process()

    def _init(self):
        """
        Here, we convert all column names into lowercase letters, confirm that the data
        is a pandas DataFrame and the length is not zero.

        Returns
        ------
        :return : bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if not pre_utils.PredictUtils.valid_pandas_data(self._data):
            self.errorlogger.logger.error("pre feature data is not DataFram: %s" % type(self._data))
            return runstatus.RunStatus.FAILED
        try:
            # Convert all column names to lowercase letters.
            self._data = self._data.copy()
            self._data.rename(columns=lambda x: x.lower(), inplace=True)
            return runstatus.RunStatus.SUCC
        except:
            self.errorlogger.logger.error(traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _single_feature_process(self):
        """
        One column of data feature processor.

        Returns
        ------
        :return : bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            for key in self._single_app:
                key_new = key.replace(' ', '')
                _data_in = self._data[key_new.lower()]
                _fun = self._single_app[key_new]
                _res = eval(self.FEATURE_PROC + _fun)(_data_in)
                if _res is not None:
                    self._data[key_new.lower()] = _res
                else:
                    self.managerlogger.logger.error(eval(self.FEATURE_PROC + _fun) + "error: %s " % key)
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("_signle_feature_process error: %s" % err)
            self.errorlogger.logger.error("_signle_feature_process error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _multi_feature_process(self):
        """
        Multi-column data feature processor.

        Returns
        ------
        :return : bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            for key in self._multi_app:
                key_new = key.replace(' ', '')
                _col_list = key_new.lower().split(self.DASH)
                _fun = self._multi_app[key_new]
                if len(_col_list) == 2:
                    _data_in1 = self._data[_col_list[0]]
                    _data_in2 = self._data[_col_list[1]]
                    _res = eval(self.FEATURE_PROC + _fun)(_data_in1, _data_in2)
                    self._data[key_new.lower() + '_' + _fun] = _res
            return runstatus.RunStatus.SUCC
        except Exception as e:
            self.managerlogger.logger.error("_multi_feature_process error: %s" % e)
            self.errorlogger.logger.error("_multi_feature_process error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def pre_feature_process(self):
        """
        pre_feature_process

        Returns
        --------
        self._data: processed data
        """
        if self._single_app:
            self._single_feature_process()
        if self._multi_app:
            self._multi_feature_process()
        return self._data
