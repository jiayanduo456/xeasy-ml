# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import pandas as pd
import traceback
from sklearn.preprocessing import OneHotEncoder
from ..systemlog import sysmanagerlog, syserrorlog
from ..ml_utils import runstatus


class XESOneHotEncoder(OneHotEncoder):
    """
    Feature selection, drop unused features and feature label.
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    Parameters
    ----------
    data: pandas.DataFrame, default = None
        input data
    one_hot_file: configparser.ConfigParser, default = None
        config file, format: column:dims
    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import xes_onehot_encoder
    >>> import pandas as pd
    >>> import configparser
    >>> data = pd.read_csv("my_data.csv")
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> ins = xes_onehot_encoder.XESOneHotEncoder(data=data, conf=conf)
    >>> ins.init()
    >>> ins.execute()

    my_one_hot_file.conf
    --------
    subject_id:9  #Number of various values
    grade_id:13   #Number of various values
    """

    ONEHOT = "onehot"
    FIELDS = "fields"
    ONEHOT_FLAG = "onehot_flag"

    DEFAULT_DIM = -1

    def __init__(self, data=None, conf=None, log_path = None):
        super(XESOneHotEncoder, self).__init__()
        self._data = data
        self.xeasy_log_path = log_path
        self._one_hot_feature = {}
        self._conf = conf
        self._onehot_flag = False
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__, self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__, self.xeasy_log_path)

    def reset(self, data, one_hot_file):
        """
        Reset data.

        Parameters
        ----------
        data: pandas.DataFrame
            input data
        one_hot_file: ConfigParser.ConfigParsser
            config file, format: column: dims
        """
        if not isinstance(data, pd.DataFrame):
            self.managerlogger.logger.error("data is not dataframe")
            return False
        if one_hot_file == "":
            self.managerlogger.logger.error("one_hot_file is empty")
            return False
        self._data = data
        self._one_hot_file = one_hot_file

    def set_data(self, data):
        """
        Set data.

        Parameters
        ----------
        data: pd.DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            self.managerlogger.logger.error("data is not dataframe")
            return False
        self._data = data

    def execute(self):
        """
        One hot encoder excute.
        """
        if self._onehot_flag:
            return self._xes_fit_transform_all()
        return False

    def get_data(self):
        """
        Get data.
        """
        return self._data

    def init(self):
        """
        read onehot configration

        Returns
        ----------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            # get conf
            try:
                self._onehot_flag = eval(self._conf.get(self.ONEHOT, self.ONEHOT_FLAG))
            except:
                self.errorlogger.logger.warning("no %s in config" % (self.ONEHOT))
                self._onehot_flag = False
            if not self._onehot_flag:
                return runstatus.RunStatus.SUCC
            onehot_field = self._conf.get(self.ONEHOT, self.FIELDS).replace(" ", "").split(",")

            # process onehot filed
            self._one_hot_feature = {}

            # init self._one_hot_feature
            if len(onehot_field) == 0:
                return runstatus.RunStatus.SUCC
            for field in onehot_field:
                field = field.split(":")
                if len(field) == 2:
                    self._one_hot_feature[field[0]] = int(field[1])
                elif len(field) == 1:
                    self._one_hot_feature[field[0]] = self.DEFAULT_DIM
            self.managerlogger.logger.info("onehot init succeed")
            return runstatus.RunStatus.SUCC
        except Exception as e:
            self.managerlogger.logger.error("onehot init error: %s" % e)
            self.errorlogger.logger.error("onehot init error: \n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _xes_fit_transform_all(self):
        """
        Fit and transform all features in configuration.

        Returns
        ----------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            for key in self._one_hot_feature:
                if key not in self._data.columns.tolist():
                    continue
                self._data = self._xes_fit_transform(self._data, key, self._one_hot_feature[key])
            self.managerlogger.logger.info("xes_fit_transform_all succeed")
            return True
        except Exception as e:
            self.managerlogger.logger.error("xes_fit_transform_all error: %s" % e)
            self.errorlogger.logger.error("xes_fit_transform_all error:\n %s" % traceback.format_exc())
            return False

    def _xes_fit_transform(self, train_data, field, field_dim):
        """
        Onehot encoder.

        Parameters
        ----------
        train_data: pandas.DataFrame
            train data
        field: str
            one hot field
        field_dim: onehot dims

        Returns
        ----------
        train data

        Examples
        ----------
        >>>df = pd.DataFrame([[1,2,1],[1,3,0,], [3,1,3]])
        >>>df.columns = ["f1", "f2", "f3"]
        >>>df
               f1  f2  f3
            0   1   2   1
            1   1   3   0
            2   3   1   3
        >>>self._xes_fit_transform(df, "f3", 3)
                    f1  f2  f3_1  f3_2  f3_3  f3_4
                0   1   2     0     1     0     0
                1   1   3     1     0     0     0
                2   3   1     0     0     0     1
        """

        try:
            # get default values
            sample_length = train_data.shape[0]

            if self.DEFAULT_DIM == field_dim:
                field_dim = int(max(train_data[field].tolist())) +1
            # Add one dimension for outlier handling.
            field_dim = field_dim +1
            new_feature = [[0 for x in range(sample_length)] for y in range(field_dim)]

            # xes onehot Compatible with negative numbers, all negative numbers and out-of-range
            # values will be classified as outliers and put into the same dimension.
            target = train_data[field].tolist()
            for i in range(sample_length):
                if (int(target[i]) >= 0) and (int(target[i]) < field_dim):
                    new_feature[int(target[i])][i] = 1
                else:
                    new_feature[field_dim - 1][i] = 1

            # insert new features of one hot
            for i in range(field_dim):
                train_data.insert(train_data.shape[1], field + "_" + str(i + 1), new_feature[i])

            # remove original feature
            del train_data[field]
            return train_data
        except:
            self.errorlogger.logger.error(traceback.format_exc())
            return train_data
