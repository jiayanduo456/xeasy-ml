# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import pandas as pd
import numpy as np
import traceback
from ..systemlog import sysmanagerlog
from ..systemlog import syserrorlog


class FeatureDiscretizer(object):
    """
    Discretize feature.

    Parameters
    ----------
    data : pandas.DataFrame, default = None
        origin dataset
    conf: configparser.ConfigParser, default = None
        Configuration file

    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import feature_discretizer
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> data = pd.read_csv("my_data.csv")
    # this
    >>> ins = feature_discretizer.FeatureDiscretizer(data=data, conf=conf)
    >>> ins.excute()
    """

    DIS_FUNC = "dis_func"
    ARGS = "args"

    EQ_FREQUE = "eq_freque"
    EQ_FREQUE_VALUE = "eq_freque_value"
    EQ_VALUE = "eq_value"

    FEATURE_DISCRETIZE = "feature_discretize"
    PARAMS = "params"

    def __init__(self, data=None, conf=None, log_path = None):
        self._data = data
        self._conf = conf
        self._feature_conf = {}
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__, log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__, log_path)

    def _init(self):
        """
        Initialization.

        Returns
        --------
        :return: bool
            True: succ
            False: failed
        """
        try:
            if self.FEATURE_DISCRETIZE not in self._conf.sections():
                return False
            # load config
            self._feature_conf = eval(self._conf.get(self.FEATURE_DISCRETIZE, self.PARAMS))
            self.managerlogger.logger.info("feature discretizer init succeed")
            return True
        except Exception as e:
            self.managerlogger.logger.error("Feature Discretizer error: %s" % e)
            self.errorlogger.logger.error("Feature Discretizer error:\n %s" % traceback.format_exc())
            return False

    def _discretize(self):
        """
        Discretize features based on configuration

        Returns
        --------
        :return: bool
            True: succ
            False: failed
        """
        try:
            data_columns = self._data.columns.tolist()
            for key in self._feature_conf:
                if key not in data_columns:
                    continue
                # Use different discretization functions according to configuration.
                func = self._discretize_freque
                if self._feature_conf[key][self.DIS_FUNC] == self.EQ_FREQUE:
                    func = self._discretize_freque
                elif self._feature_conf[key][self.DIS_FUNC] == self.EQ_VALUE:
                    func = self._discretize_value
                elif self._feature_conf[key][self.DIS_FUNC] == self.EQ_FREQUE_VALUE:
                    func = self._discretize_freque_value

                self._data[key] = pd.Series(func(self._data[key].tolist(), self._feature_conf[key][self.ARGS]))
            self.managerlogger.logger.info("feature discretize succeed")
            return True
        except Exception as e:
            self.managerlogger.logger.error("feature discretize error: %s" % e)
            self.errorlogger.logger.error("feature discretize error:\n %s" % traceback.format_exc())
            return False

    def _discretize_freque(self, value_list, args):
        """
        Equal frequency division, evenly divided into multiple segments.

        Parameters
        --------
        value_list:
            Columns that need to be discretized in the data set
        args: int
            Number of discrete classes.

        Returns
        --------
        tmp_res : list
            Discretization result

        Examples
        --------
        >>>value_list = np.random.randint(2, 10, 6)
            array([7, 3, 4, 2, 9, 8])
        >>>args = 2
        >>>self._discretize_freque(value_list, args)
            [1, 0, 0, 0, 1, 1]
        """
        length = len(value_list)
        tmp_res = [[index, value_list[index], 0] for index in range(length)]
        tmp_res.sort(key=lambda x: x[1])
        class_nums = int(float(args))
        res = [int((x * class_nums) / length) for x in range(length)]
        for index in range(len(tmp_res)):
            tmp_res[index][2] = res[index]

        tmp_res.sort(key=lambda x: x[0])
        tmp_res = [x[2] for x in tmp_res]

        return tmp_res

    def _discretize_freque_value(self, value_list, args):
        """
        Equal frequency division, different from the above equal frequency division, the same
        value can be divided into the same category.

        Parameters
        --------
        value_list:
            Columns that need to be discretized in the data set
        args: int
            Number of discrete classes.

        Returns
        --------
        :return: list
        """
        length = len(value_list)
        tmp_res = [[index, value_list[index], 0] for index in range(length)]
        tmp_res.sort(key=lambda x: x[1])
        class_nums = int(float(args))

        # Divide equally into n segments and take the value of the node in each segment.
        step = int(length / class_nums)
        bins = [tmp_res[x][1] for x in range(step, length, step)]

        return np.digitize(value_list, bins=bins).tolist()

    def _discretize_value(self, value_list, args):
        """
        Discretize according to the given threshold.

        Parameters
        --------
        value_list: list
        args: list

        Returns
        --------
        :return: list

        Examples
        --------
        >>> a = [7, 3, 4, 2, 9, 8]
        >>> args = [4, 7, 10]
        >>>self._discretize_value(a, args)
            [2, 0, 1, 0, 2, 2]
        """
        return np.digitize(value_list, bins=args).tolist()

    def reset(self, data=None, conf=None):
        """
        Reset data and conf.

        Parameters
        --------
        data: pandas,DataFrame
        conf: ConfigParser.ConfigParser
        """
        if not isinstance(data, pd.DataFrame):
            self.managerlogger.logger.error("data is not dataframe")
            return False
        self._data = data
        self._conf = conf

    def excute(self):
        """
        Excute Member method,self._init and self._discretize.

        Returns
        --------
        :return: bool
            True: succ
            False: failed
        """
        if not self._init():
            return False
        return self._discretize()

    @property
    def get_data(self):
        """
        useage: feature_discretizer = FeatureDiscretizer()
                data = feature_discretizer.get_data

        Returns
        --------
        self._data
        """
        return self._data
