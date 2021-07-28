# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import pickle
import traceback
import os
import pandas as pd
from sklearn import preprocessing
from ..systemlog import sysmanagerlog
from ..systemlog import syserrorlog
from ..ml_utils import runstatus, global_pre


class DataWasher(object):
    """
        Sort column, zscore.

        Parameters
        ----------
        data : pandas.DataFrame, default = None
            data need wash
        conf : configparser.ConfigParser, default = None
        zscal_pickle_file : default = None
        train_flag : bool, default = False
            set train flag as true, train data need wash
        processor : default = None

        Examples
        --------
        >>> from xes_ml_arch.src.feature_enginnering import data_washer
        >>> import configparser
        >>> import pandas as pd
        >>> conf = configparser.ConfigParser()
        >>> conf.read("myconfig.conf")
        >>> data = pd.read_csv("my_data.csv")
        # this
        >>> ins = data_washer.DataWasher(data=data, conf=conf)
        >>> ins.init()
        >>> ins.set_train_flag()
        >>> # or ins.set_test_flag()
        >>> ins.excute()
        # or this
        >>> ins = data_washer.DataWasher(conf=conf)
        >>> ins.init()
        >>> ins.set_data(data)
        >>> ins.set_train_flag()
        >>> ins.excute()
        """
    MEAN = "mean"
    SCALE = "scale"
    PROPERTIES = ".properties"
    DATA_WASHER = "data_washer"
    ZSCAL_PICKLE_FILE = "zscal_pickle_file"
    WASH_FLAG = "wash_flag"

    def __init__(self, data=None, conf=None, log_path = None, zscal_pickle_file=None, train_flag=False,
                 processor=None):
        self._data = data
        self._conf = conf
        self.xeasy_log_path = log_path
        self._zscal_pickle_file = zscal_pickle_file
        self._train_flag = train_flag
        self._scale = None
        self._processer = processor
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Init data washer.

        Returns
        ----------
        :return: bool, runstatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            _zscal_pickle_file = self._conf.get(self.DATA_WASHER, self.ZSCAL_PICKLE_FILE)
            self._zscal_pickle_file = os.path.join(global_pre.RES_PATH, _zscal_pickle_file)
            if not os.path.isdir(os.path.dirname(self._zscal_pickle_file)):
                os.mkdir(os.path.dirname(self._zscal_pickle_file))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("data washer init error: %s" % err)
            self.errorlogger.logger.error("data washer init error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def get_data(self):
        """
        Get data.

        Returns
        ----------
        self._data
        """
        return self._data

    def set_data(self, data):
        """
        Set data need wash.

        Parameters
        ----------
        data: pandas.DataFrame
            data need wash.
        """
        self._data = data

    def set_train_flag(self):
        """
        Set train flag as true, train data need wash
        default is False
        """
        self._train_flag = True

    def set_test_flag(self):
        """
        Set _train_flag as false, data no need wash when test
        default is False

        """
        self._train_flag = False

    def excute(self):
        """
        starting wash data.

        Returns
        ----------
        self._data
        """
        try:
            self._init()
            self.sort_columns()

            if not self._wash_flag:
                self.managerlogger.logger.info("no need data wash")
                return self._data

            self.control_data(self._processer)
            self.zscore_data()
            self.managerlogger.logger.info("data washer excute succeed")
            return self._data
        except Exception as e:
            self.managerlogger.logger.error("data washer excute error: %s" % e)
            self.errorlogger.logger.error("data washer excute error:\n %s " % traceback.format_exc())
            return None

    def sort_columns(self):
        """Sort columns by default."""

        columns = self._data.columns.tolist()
        columns.sort()
        self._data = self._data[columns]

    def zscore_data(self):
        """
        Standardize func.
        self._data : the data to be scaled
        :return: data after scaling
        """
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError

        colnames = self._data.columns.tolist()
        if self._train_flag:
            self._scale = self._scale.fit(self._data)
            pickle.dump(self._scale, open(self._zscal_pickle_file, "w"))
            self._store_zscore_scale()
        after_scale_data = self._scale.transform(self._data)
        self._data = pd.DataFrame(after_scale_data, columns=colnames)

    def _init(self):
        """Init param"""

        # load zscore file or excute standarScaler
        if self.WASH_FLAG in self._conf.options(self.DATA_WASHER):
            self._wash_flag = eval(self._conf.get(self.DATA_WASHER, self.WASH_FLAG))
        if not self._wash_flag:
            return
        if not self._train_flag:
            try:
                self._scale = pickle.load(open(self._zscal_pickle_file, "r"))
                return
            except Exception as ex:
                self.managerlogger.logger.error("load zscal_pickle_file error: %s" % ex)
                self.errorlogger.logger.error("load zscal_pickle_file error:\n %s" % (traceback.format_exc()))
        self._scale = preprocessing.StandardScaler()

    def _store_zscore_scale(self):
        """Save scale result for pmml file, with MEAN and SCALE."""
        try:
            properties_path, properties_file = os.path.split(self._zscal_pickle_file)
            properties_file = os.path.splitext(properties_file)[0] + self.PROPERTIES
            preperties_file = os.path.join(properties_path, properties_file)
            # compine mean amd scale with column like "columnnA = {mean:3, scale:2}"
            res = map(
                lambda x: "%s = {\"%s\":%s, \"%s\":%s}" % (x[0], self.MEAN, x[1], self.SCALE, x[2]),
                zip(self._data.columns.tolist(), self._scale.mean_.tolist(),
                    self._scale.scale_.tolist()))
            with open(preperties_file, "w") as file:
                file.write("\n".join(res))
        except Exception as ex:
            self.managerlogger.logger.error("store properties error: %s" % (str(ex)))
            self.errorlogger.logger.error("store properties error:\n %s" % traceback.format_exc())

    def control_data(self, function_di):
        """
        Customized function. You can add customized functions here.

        Parameters
        ----------
        function_di: {"field":column, "function": function}
        """
        if function_di is None:
            self.managerlogger.logger.debug("processor function is None")
            return
        if len(function_di) == 0 and not isinstance(function_di, dict):
            return
        for key in function_di:
            try:
                tmp_field = key
                tmp_function = function_di[key]
                if tmp_field in self._data.columns.tolist():
                    self._data[tmp_field] = self._data[tmp_field].apply(tmp_function)

                    # unittest
                    # self._data[tmp_field] = self._data[tmp_field].apply(lambda x : x+x)

                    self.managerlogger.logger.info("process %s success" % (key))
            except Exception as ex:
                self.managerlogger.logger.error("function can not process %s" % (key))
                self.errorlogger.logger.error("function can not process %s" % (ex))
