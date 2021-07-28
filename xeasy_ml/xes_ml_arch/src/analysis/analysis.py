# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import itertools
import pandas as pd
from ..ml_utils import runstatus, global_pre
from ..systemlog import sysmanagerlog, syserrorlog
import traceback
import matplotlib.pyplot as plt
import os
import numpy as np
from functools import reduce
import warnings

warnings.filterwarnings("ignore")


class Analysis(object):
    """
    This class is an analysiser to analysis a data set. You can call various methods below the
    class for visual display and analysis of data set.

    Parameters
    --------
    conf : configparser.ConfigParser, default = None
        config of analysis
    data : pandas.DataFrame, default = None
        data set
    feature_columns : list, default = None
        column name of feature vector. It is a feature list to be analysis.
    label_columns : str, default = ""
        label column name

    Examples
    --------
    >>> from xes_ml_arch.src.analysis import analysis
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
    >>>           [[int(random.random() * 100) for _x in range(10)] for _y in range(10000)], columns=columns)
    >>> data["col9"] = np.random.choice([0, 1], size=10000)
    # this
    >>> ins = analysis.Analysis(conf=conf, data=data, feature_columns=x,
    >>>                        label_columns=y)
    >>> ins.execute()
    # or this
    >>> ins = analysis.Analysis(conf=conf)
    >>> ins.reset(conf=conf, data=data, feature_columns=x,
    >>>                        label_columns=y)
    >>> ins.execute()
    """
    ANALYSIS = "analysis"
    FEATURE_DEL_THRESH = "feature_del_thresh"
    ANALYSIS_RESULT = "analysis_result"
    DELETED_FILE = "deleted_feature.txt"
    BOX = "box"
    HIST = "hist"
    RELATION = "relation"
    NAME_DICT_FILE = "name_dict_file"
    FEATURE_WITH_FEATURE = "feature_with_feature"
    FEATURE_WITH_LABEL = "feature_with_label"

    THRESHOLD_DEFAULT = 0.9

    def __init__(self, conf=None, log_path = None, data=None, feature_columns=None, label_columns=""):
        self._data = data

        self.log_path = log_path

        self._conf = conf
        self._feature_columns = feature_columns
        self._label_column = label_columns
        self._feature_corr = []
        self._target_corr = []
        self._result_path = None

        self.managerlogger = sysmanagerlog.SysManagerLog(__file__, self.log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.log_path)

    def reset(self, conf=None, data=None, feature_columns=None, label_columns=""):
        """
        Set conf, data, feature_columns, label_columns.
        If value is default, then pass.

        Parameters
        --------
        conf : configparser.ConfigParser, default = None
            config of analysis
        data : pandas.DataFrame, default = None
            data set
        feature_columns : list, default = None
            column name of feature vector. It is a feature list to be analysis.
        label_columns : str, default = ""
            label column name
        """
        if conf is not None:
            self._conf = conf
        if data is not None:
            self._data = data
        if feature_columns is not None:
            self._feature_columns = feature_columns
        if label_columns != "":
            self._label_column = label_columns

    def execute(self):
        """
        Start to analysis.

        Returns
        --------
        :return: bool, runstatus.RunStatus
             runstatus.RunStatus.SUCC: True
             runstatus.RunStatus.FAILED: Failed
        """
        try:
            # self._init()
            # get feature correlation and save as document
            self.get_feature_corr()
            # get correlation between features and label, and save as document
            self.get_label_corr()
            # Delete one of the two features whose correlation is higher than the threshold, and save
            # the deleted feature name to the file
            self.get_del_feature()
            # plot box for every feature
            self.plot_box()
            # plot hist figure for every feture
            self.plot_hist()
            self.managerlogger.logger.info("analysis execute succeed!")
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.errorlogger.logger.error("analysis execute error! \n" + traceback.format_exc())
            self.managerlogger.logger.error("analysis execute error!  %s" % err)
            return runstatus.RunStatus.FAILED

    def get_feature_corr(self):
        """
        Get correlation between x1 ang x2 in X, and save as document.
        """
        self._checkout()
        feature_combin = itertools.combinations(self._feature_columns, 2)
        self._feature_corr = []
        for x1, x2 in feature_combin:
            key = (x1, x2)
            if not x1 in self._data.columns or not x2 in self._data.columns:
                continue
            self._feature_corr.append([key, self._data[x1].corr(self._data[x2])])

        # sort and store feature's correlation
        self._feature_corr.sort(key=lambda x: x[1], reverse=True)
        with open(os.path.join(self._result_path, self.FEATURE_WITH_FEATURE), "w") as file_handle:
            file_handle.write(
                "\n".join(["%s:%s %s" % (x[0][0], x[0][1], x[1]) for x in self._feature_corr]))

    def get_label_corr(self):
        """Get correlation between features(x) and label(y), and save as document."""
        self._checkout()
        self._target_corr = []
        for column in self._feature_columns:
            if column not in self._data.columns:
                continue
            corration = self._data[column].corr(self._data[self._label_column])
            corration = 0 if np.isnan(corration) else corration
            self._target_corr.append([column, corration])

        # sort and store correlation between feature and label
        self._target_corr.sort(key=lambda x: abs(float(x[1])), reverse=True)
        with open(os.path.join(self._result_path, self.FEATURE_WITH_LABEL), "w") as file_handle:
            file_handle.write("\n".join(["%s:%s" % (x[0], float(x[1])) for x in self._target_corr]))

    def get_del_feature(self):
        """When correaltion between x1 and x2 beyond a threshold, keep x2 and drop x1."""

        if len(self._feature_corr) == 0:
            raise ValueError("correlation is between x and y is empty")
        try:
            threshold = float(self._conf.get(self.ANALYSIS, self.FEATURE_DEL_THRESH))
        except Exception as err:
            self.managerlogger.logger.error("threshold is not is config, use default threshold value. %s " % err)
            self.errorlogger.logger.error("get_del_feature error:\n %s " % traceback.format_exc())
            threshold = self.THRESHOLD_DEFAULT

        last_result = filter(lambda x: x[1] > threshold, self._feature_corr)
        del_id_list = [x[0][0] for x in last_result]
        with open(os.path.join(self._result_path, self.DELETED_FILE), "w") as file_handle:
            file_handle.write("\n".join(del_id_list))

    def plot_box(self):
        """Plot boxplots for every feature."""

        self._checkout()
        box_png = os.path.join(self._result_path, self.BOX)
        if not os.path.isdir(box_png):
            os.mkdir(box_png)
        for feature in self._feature_columns:
            if feature in self._data.columns and feature != self._label_column:
                # reload(sys)
                # sys.setdefaultencoding('utf8') #python3 no need?
                self._data[[feature, self._label_column]].boxplot(by=self._label_column)
                plt.savefig(os.path.join(box_png, u"%s.png" % (feature)))
                plt.close()

    def plot_hist(self):
        """Plot histogram."""

        self._checkout()
        # Init path to save png.
        hist_png_path = os.path.join(self._result_path, self.HIST)
        if not os.path.isdir(hist_png_path):
            os.mkdir(hist_png_path)
        label_set = set(self._data[self._label_column].tolist())

        # plot png for every feture
        for feature in self._feature_columns:
            if feature in self._data.columns:
                plot_dict = {}
                for label in label_set:
                    plot_dict[label] = self._data[self._data[self._label_column] == label][
                        feature].tolist()
                # reload(sys)
                # sys.setdefaultencoding('utf8')
                # plot histogram
                self._plot_data_n(plot_dict, os.path.join(hist_png_path, u"%s.png" % (feature)))
                # plot distribut of all different target
                self._plot_distrib(plot_dict,
                                   os.path.join(hist_png_path, u"%s_dis.png" % (feature)))
                # plot statistic for one feature data
                self._plot_relation(plot_dict,
                                    os.path.join(hist_png_path, u"%s_rel.png" % (feature)))
                # draw a line graph of the relationship between feature values and labels
                self._plot_relation_2(plot_dict,
                                      os.path.join(hist_png_path, u"%s_rel_2.png" % (feature)))

    def _plot_relation(self, plot_dict, res_path):
        """
        Plot statistic for one feature data.

        Parameters
        --------
        plot_dict: dict
            {key:value},key is a value in set(data[target]), value is list of data[feature]
        res_path: str
            store path
        """
        all_values = reduce(lambda x, y: x + y, [x[1] for x in plot_dict.items()])
        value_list = list(set(all_values))
        value_list.sort()
        plt.figure(figsize=(15, 10))

        dict_vlaues_all = dict([[x, all_values.count(x)] for x in value_list])

        for key in plot_dict:
            tmp_list = [float(plot_dict[key].count(x)) / dict_vlaues_all[x] for x in value_list]
            plt.plot(value_list, tmp_list, label=str(key))
        plt.legend()
        plt.savefig(res_path)
        plt.close()

    def _plot_relation_2(self, plot_dict, res_path):
        """
        Draw a line graph of the relationship between feature values and labels.

        Parameters
        --------
        plot_dict: dict
            {key:value},key is a value in set(data[target]), value is list of data[feature]
        res_path: str
            store path
        """
        all_values = reduce(lambda x, y: x + y, [x[1] for x in plot_dict.items()])
        value_list = list(set(all_values))
        value_list.sort()
        count_value = [all_values.count(x) for x in value_list]
        sum_value = [sum(count_value[:i + 1]) for i in range(len(count_value))]
        dict_vlaues_all = dict(zip(value_list, sum_value))
        plt.figure(figsize=(15, 10))

        for key in plot_dict:
            tmp_count = [plot_dict[key].count(x) for x in value_list]
            tmp_sum = [sum(tmp_count[:i + 1]) for i in range(len(tmp_count))]
            tmp_list = [float(tmp_sum[i]) / dict_vlaues_all[x] for i, x in enumerate(value_list)]
            plt.plot(value_list, tmp_list, label=str(key))
        plt.legend()
        plt.savefig(res_path)
        plt.close()

    def _plot_distrib(self, plot_dict, res_path):
        """
        Plot distribut of all different target.

        Parameters
        --------
        plot_dict: dict
            {key:value},key is a value in set(data[target]), value is list of data[feature]
        res_path: str
            store path
        """
        all_values = reduce(lambda x, y: x + y, [x[1] for x in plot_dict.items()])
        value_list = list(set(all_values))
        value_list.sort()
        x_lim = range(len(value_list))
        width = 0.3

        plt.figure(figsize=(15, 10))
        plt.xticks(x_lim, [str(_) for _ in value_list])
        for key in plot_dict:
            tmp_list = [plot_dict[key].count(x) for x in value_list]
            plt.bar(x_lim, tmp_list, width=width, label=str(key))
            x_lim = [_ + width for _ in x_lim]
        plt.legend()
        plt.savefig(res_path)
        plt.close()

    def _plot_data_n(self, plot_dict, res_path):
        """
        Plot hist figure.

        Parameters
        --------
        plot_dict: dict
            {key:value},key is a value in set(data[target]), value is list of data[feature]
        res_path: str
            store path
        """
        # all_values = reduce(lambda x, y: x + y, plot_dict)
        #
        all_values = reduce(lambda x, y: x + y, [x[1] for x in plot_dict.items()])
        plt.xlim([min(all_values) - 0.1, max(all_values) + 0.1])
        # plt.bar()
        plt.figure(figsize=(15, 10))

        # Judging whether it is classification or regression.
        if len(plot_dict) > 20:
            tmp_value = len(set(all_values))
            bins = int(min(max(tmp_value / 10, 10), 100))
            plt.hist(all_values, bins, alpha=0.3, label=str("all"))
        else:
            for key in plot_dict:
                tmp_value = len(set(plot_dict[key]))
                bins = int(min(max(tmp_value / 10, 10), 100))
                plt.hist(plot_dict[key], bins, alpha=0.3, label=str(key))

        plt.legend(loc='upper right')
        plt.title(res_path)
        plt.savefig(res_path)
        plt.close()

    def _checkout(self):
        """
        checkout data, feature and label

        Notes
        --------
        If you do not meet the specification, exception is thrown.
        """
        if self._data is None:
            raise ValueError("data is None")
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError("data type is not pandas.DataFrame")
        if self._feature_columns is None or self._label_column == "":
            raise ValueError("sample feature or label is empty")

        if not self._label_column in self._data.columns:
            raise AttributeError("data not contains %s, contained fields is [%s]" % (
                self._label_column, str(self._data.columns.tolist())))

        self._result_path = global_pre.RES_PATH
        # unittest
        #self._result_path = self._conf.get(self.ANALYSIS, self.ANALYSIS_RESULT)

        if not os.path.isdir(self._result_path):
            os.mkdir(self._result_path)

    def _init(self, data):
        pass
