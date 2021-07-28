# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import pandas as pd
import traceback
from ..systemlog import sysmanagerlog, syserrorlog
from ..systemlog import syserrorlog
from ..ml_utils import runstatus
from ..ml_utils import pre_utils


class DataSampler(object):
    """
    Data sampling of train data or test data.

    Parameters
    ---------
    conf: configparser.ConfigParser
        Configuration information
    data: pandas.DataFrame
    sample_rate: float
    target: col of data, label
    base_size:int, default = -1

    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import data_sampler
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> data = pd.read_csv("my_data.csv")
    # this
    >>> ins = data_sampler.DataSampler(data=data, conf=conf, target='col9')
    >>> ins.init()
    >>> ins.excute()
    # or this
    >>> ins = data_sampler.DataSampler(conf=conf, target='col9')
    >>> ins.init()
    >>> ins.set_data(data=data)
    >>> ins.excute()
    """

    DATA = "data"
    VALUE = "value"
    SIZE = "size"
    RATE = "rate"
    DATA_SAMPLE = "data_sample"
    SAMPLE_RATE = "sample_rate"
    SAMPLE_FLAG = "sample_flag"

    def __init__(self, conf=None, log_path = None, data=None, sample_rate=None, target="", base_size=-1):
        self._conf = conf
        self.xeasy_log_path = log_path
        self._data = data
        self._sample_rate = sample_rate
        self._target = target
        self._base_size = base_size
        self._result_sample_data = None
        self._type_data = []
        self._sample_flag = True
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Init data sampler object
        Returns
        ---------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            self._sample_flag = eval(self._conf.get(self.DATA_SAMPLE, self.SAMPLE_FLAG))
            if not self._sample_flag:
                return runstatus.RunStatus.SUCC
            self._sample_rate = eval(self._conf.get(self.DATA_SAMPLE, self.SAMPLE_RATE))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("data sampler init error: %s" % err)
            self.errorlogger.logger.error("data sampler init error:\n %s" % traceback.format_exc())
        return runstatus.RunStatus.FAILED

    def set_target_field(self, target_str):
        """
        Set target.

        Parameters
        ----------
        target_str: str
            name of col, label

        Returns
        ----------
        :return: bool, runstatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if not isinstance(target_str, str):
            return runstatus.RunStatus.FAILED
        self._target = target_str
        return runstatus.RunStatus.SUCC

    def excute(self):
        '''
        Data sampling start.

        Returns
        ----------
        :return: bool, runstatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        '''
        self.managerlogger.logger.info("start sample")
        # init
        if not self._sample_flag:
            self.managerlogger.logger.info("no need to sample")
            self._result_sample_data = self._data
            return runstatus.RunStatus.SUCC
        if self._init() == runstatus.RunStatus.FAILED:
            return runstatus.RunStatus.FAILED
        # sample
        if self._start_sample() == runstatus.RunStatus.FAILED:
            return runstatus.RunStatus.FAILED
        self.managerlogger.logger.info("finish sample")
        return runstatus.RunStatus.SUCC

    def get_data(self):
        """
        Get data.

        Returns
        ----------
        :return: result of data sample
        """
        return self._result_sample_data

    def set_data(self, data):
        """
        Set data.

        Parameters
        ----------
        data: pandas.DataFrame
            data need sample

        Returns
        ----------
        :return: bool, runstatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if not pre_utils.PredictUtils.valid_pandas_data(data):
            return runstatus.RunStatus.FAILED
        self._data = data
        return runstatus.RunStatus.SUCC

    def _init(self):
        """
        Check input and init info of all class

        Returns
        ----------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """

        if not isinstance(self._data, pd.DataFrame):
            self.managerlogger.logger.error("input data is not dataframe")
            return runstatus.RunStatus.FAILED
        if self._target == "":
            self.managerlogger.logger.error("target not find")
            return runstatus.RunStatus.FAILED
        if self._sample_rate is None:
            self.managerlogger.logger.error("sample rate is required")
            return runstatus.RunStatus.FAILED
        return self._init_type_data()

    def _init_type_data(self):
        """
        Init info of all class:
        {data:samples, value:target value, size:numbers of sample, rate:rate of max(size)}

        Returns
        ----------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        # Get the number of samples for each category, and the label value.
        di_type_size = self._get_type_size()
        if len(self._sample_rate) != len(di_type_size):
            self.managerlogger.logger.error(
                "%s \n and %s \n not match" % (str(self._sample_rate), str(di_type_size)))
            return runstatus.RunStatus.FAILED
        try:
            for key in self._sample_rate:
                tmp_data = self._data[self._data[self._target] == key]
                tmp_size = di_type_size[key]
                tmp_rate = self._sample_rate[key]
                tmp_res_di = {self.DATA: tmp_data, self.SIZE: tmp_size, self.RATE: tmp_rate,
                              self.VALUE: key}
                self._type_data.append(tmp_res_di)
            self.managerlogger.logger.info("_init_type_data succeed")
            return runstatus.RunStatus.SUCC
        except Exception as e:
            self.managerlogger.logger.error("_init_type_data error: %s" % e)
            self.errorlogger.logger.error(traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _get_type_size(self):
        """
        Get szie for every class

        Returns
        ---------
        :return: dict
            Return the name of each category and the corresponding quantity of each category.
        """
        count_series = self._data[self._target].value_counts()
        if self._base_size == -1:
            self._base_size = max(count_series.tolist())
        return dict(zip(count_series.index.tolist(), count_series.tolist()))

    def _start_sample(self):
        """
        Sample function.

        Returns
        ---------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        try:
            self._result_sample_data = pd.DataFrame(columns=self._data.columns)
            for di_value in self._type_data:
                sample_size = di_value[self.SIZE]
                sample_rate = di_value[self.RATE]
                sample_data = di_value[self.DATA]

                # Calculate the number of samples after sampling according to the relative
                # sampling ratio, compare with the actual number of samples, and select
                # over-sampling or under-sampling.
                target_size = int(self._base_size * sample_rate)
                if target_size < sample_size:
                    # under-sampling
                    sample_data = sample_data.sample(n=target_size, replace=False, random_state=37)
                elif target_size > sample_size:
                    # over-sampling
                    add_sample_data = sample_data.sample(n=target_size - sample_size, replace=True,
                                                         random_state=37)
                    sample_data = sample_data.append(add_sample_data)

                self._result_sample_data = self._result_sample_data.append(sample_data)
            # set type for all col
            for col in self._result_sample_data.columns:
                try:
                    self._result_sample_data[col] = self._result_sample_data[col].astype("float64")
                except:
                    self.errorlogger.logger.warning("%s can not astype float, which type is %s" % (
                    col, str(self._result_sample_data[col].dtype)))
            return runstatus.RunStatus.SUCC
        except Exception as e:
            self.managerlogger.logger.error("start sample error: %s" % e)
            self.errorlogger.logger.error("start sample error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED
