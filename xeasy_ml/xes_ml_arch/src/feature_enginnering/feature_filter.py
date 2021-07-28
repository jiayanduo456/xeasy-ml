# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import configparser
import traceback
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ..systemlog import sysmanagerlog
from ..systemlog import syserrorlog
from ..ml_utils import runstatus
from ..ml_utils import pre_utils
from ..ml_utils import global_pre
import pickle
import numpy as np
import os


class FeatureFilter(object):
    """
    Base class for feature filter that includes feature selection, unused features droping
    and feature label.
    Includefeature selection, drop unused features and feature label.

    Parameters
    ----------
    conf: configparser.ConfigParser()
        config
    data_file: str
        data file path
    del_list: list
        columns should be delete
    del_keywords: list
        columns should be delete which contains key word in `del_keywords`
    label_list: list
        columns which should be labeled
    selected_list: list
        columns which should be selected
    data: pandas.DataFrame
        samples

    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import feature_filter
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> data = pd.read_csv("my_data.csv")
    # this
    >>> ins = feature_filter.FeatureFilter(conf=conf)
    >>> ins.init()
    >>> ins.set_data(data)
    >>> ins.excute()
    # or this
    >>> ins = feature_filter.FeatureFilter(conf=conf, data=data)
    >>> ins.init()
    >>> ins.excute()
    # or this
    >>> ins = feature_filter.FeatureFilter(conf=conf, data_file="my_data.csv")
    >>> ins.init()
    >>> ins.excute()
    """
    PREPERTIES_FILE = "./config/label.properties"
    FEATURA_FILTER = "feature_filter"
    DELETE_FIELD_LIST = "delete_field_list"
    DELETE_KEYWORDS = "delete_keyword_list"
    LABEL_LIST = "label_list"
    SELECTED_LIST = "selected_list"

    ENCODEE_FILE_PATH = "label_encoder_file"
    ENCODER_TABLE = ".table"

    def __init__(self, conf=None, log_path = None, data_file=str(), del_list=list(), del_keywords=list(),
                 label_list=list(), selected_list=list(), data=None):
        # conf
        self._conf = conf
        self.xeasy_log_path = log_path
        # data to filter
        self._data = data
        # data file, if data is None ,read_data from file
        self._data_file = data_file
        # fields to delete
        self._del_list = del_list
        # fields which contain keywords to delete
        self._del_keywords = del_keywords
        # fields to label
        self._label_list = label_list
        # fields to select
        self._extract_list = selected_list
        # data columns
        self._columns = []

        # feature label encoder
        self._feature_lebel_encoder = None
        self._feature_lebel_encoder_file = ""
        self._feature_lebel_encoder_table_file = ""

        # logs
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Read information from the configuration file for initialization.
        Returns
        ---------
        :return: bool
        """
        try:
            if not isinstance(self._conf, configparser.ConfigParser):
                self.managerlogger.logger.error("init feature_filter error: config is None")
                return runstatus.RunStatus.FAILED

            # get fields from conf
            self._del_list = self._get_list_from_conf(self.DELETE_FIELD_LIST)
            self._del_keywords = self._get_list_from_conf(self.DELETE_KEYWORDS)
            self._label_list = self._get_list_from_conf(self.LABEL_LIST)
            self._extract_list = self._get_list_from_conf(self.SELECTED_LIST)

            # load encoder
            self._load_encoder_obj()

            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("feature filter init error: %s" % err)
            self.errorlogger.logger.error("feature filter init error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def set_data(self, data):
        """
        set data

        Parameters
        ----------
        data: data to filter, type: pandas.DataFrame
            Initial data for training or testing.

        Returns
        ----------
        :return: bool, runstatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: False
        """
        if not pre_utils.PredictUtils.valid_pandas_data(data):
            self.managerlogger.logger.error("feature filter set data error")
            return runstatus.RunStatus.FAILED
        self._data = data
        return runstatus.RunStatus.SUCC

    def excute(self):
        """
        The basic feature filtering methods include lowercase column names, feature cleaning, feature
        extraction, feature deletion by column or keyword, and target tag encoding.

        Returns:
        ------
        self._data
        """
        try:
            # Convert column names into lists and convert all column names to lowercase.
            self._init()
            # Clean the features
            self.clean_features()
            # Extract features
            self.extract_features()
            # Delete features not used by colunms.
            self.drop_features()
            # Delete fields which contains the keywords.
            self.drop_features_by_keywords()
            # Encode target labels with value between 0 and n_classes-1.
            self.label_fields()
            return self._data
        except Exception as ex:
            self.errorlogger.logger.error("feature filter excute error:\n %s" % traceback.format_exc())
            self.managerlogger.logger.error("feature filter excute error: %s" % ex)

    def get_data(self):
        """
        Get filted data
        Returns
        ----------
        self._data
        """
        return self._data

    def reset(self, data=None, del_list=None, del_keywords=None, label_list=None,
              selected_list=None):
        """
        Set data, del_list, del_keywords, etc.

        Parameters
        ----------
        data: pandas.DataFrame
            orignal data
        del_list: list
            columns need to be deleted
        del_keyword: list
            key words of the columns need to be deleted
        label_list: list
            columns need to be labeled
        """
        if data is not None:
            self._data = data
            self._columns = self._data.columns
        if del_list is not None:
            self._del_list = del_list
        if del_keywords is not None:
            self._del_keywords = del_keywords
        if label_list is not None:
            self._label_list = label_list
        if selected_list is not None:
            self._extract_list = selected_list

    def clean_features(self, empty_val=-1):
        """Clean the features, currently including the replacement of outliers, the filling of
        missing values, and the deletion of duplicate columns."""

        # replace exception
        self._data = self._data.replace("\\N", 0)
        # fill null
        self._data = self._data.fillna(empty_val)
        self._data.columns = self._data.columns.str.strip(" ")

        # remove duplicated columns
        self._data = self._data.loc[:, ~self._data.columns.duplicated()]
        self._columns = self._data.columns.tolist()

    def drop_features(self):
        """
        Delete fields not used. This function will update self._data, self._columns.

        Examples
        --------
        Given a dataset with three features and four samples, we let the drop_features.
        >>>_del_list=["feature1"]
        >>>df = pd.DataFrame([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        >>>df.columns=["feature1","feature2","feature3"]
        ['feature1' 'feature2' 'feature3']
        >>>drop_features()
        ['feature2' 'feature3']
        """

        # warning: filter function return object in python 3.x
        self._del_list = list(filter(None, self._del_list))
        if len(self._del_list) == 0:
            return
        # case insensitive
        self._del_list = self.case_insensitive(self._del_list)
        self._columns = self.case_insensitive(self._columns)

        del_list = self.intersect_cloumns(self._del_list, self._columns)

        if len(del_list) > 0:
            # Delete by column.
            self._data = self._data.drop(list(del_list), 1)
        self._columns = self._data.columns.tolist()

    def drop_features_by_keywords(self):
        """
        Delete fields which contains the keywords. This function will update self._data.

        Examples
        --------
        Given a dataset with three features and four samples, we let the drop_fields.
        >>>_del_list=["testfeature"]
        >>>df = pd.DataFrame([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        >>>df.columns=["testfeature1","testfeature2","feature3"]
        ['feature1' 'feature2' 'feature3']
        >>>drop_features_by_keywords()
        ['feature3']
        """
        self._del_keywords = filter(None, self._del_keywords)
        #  case insensitive
        self._del_keywords = self.case_insensitive(self._del_keywords)
        self._columns = self.case_insensitive(self._columns)
        del_list = []
        for _col in self._columns:
            for _keyword in self._del_keywords:
                if _col.find(_keyword) != -1:
                    del_list.append(_col)
        if len(del_list) == 0:
            return

        self._data = self._data.drop(list(del_list), 1)
        self._columns = self._data.columns.tolist()

    def label_fields(self):
        '''
        Label some field.
        Encode target labels with value between 0 and n_classes-1.
        '''
        if len(self._label_list) == 0:
            return

        res = {}
        encoder_flag = False

        if not self._feature_lebel_encoder:
            self._feature_lebel_encoder = {}
            encoder_flag = True

        try:

            for _f in self._label_list:
                if _f not in self._columns:
                    continue

                # After adding labels that need to be labeled, you need to store the data.
                if _f not in self._feature_lebel_encoder and not encoder_flag:
                    encoder_flag = True

                if encoder_flag:
                    lebel_encoder = LabelEncoder()
                    # Fit label encoder and return encoded labels.
                    self._data[_f] = [x + 1 for x in lebel_encoder.fit_transform(self._data[_f])]
                    res[_f] = dict(zip(lebel_encoder.classes_.tolist(),
                                       range(1, len(lebel_encoder.classes_.tolist()) + 1)))
                    self._feature_lebel_encoder[_f] = lebel_encoder
                else:
                    lebel_encoder = self._feature_lebel_encoder.get(_f)
                    try:
                        self._data[_f] = [x + 1 for x in lebel_encoder.transform(self._data[_f])]
                    except:
                        class_arr = lebel_encoder.classes_.tolist()
                        try:
                            tmp_label_val = np.searchsorted(class_arr, self._data[_f].tolist(),
                                                            sorter=np.argsort(class_arr))

                            self._data[_f] = [x + 1 for x in tmp_label_val]
                        except:
                            continue

            if encoder_flag:
                self._store_label_properties(res)

        except Exception as e:
            self.errorlogger.logger.error("label_fields error:\n %s " % traceback.format_exc())
            self.managerlogger.logger.error("label_fields error:\n %s " % e)

    def extract_features(self):
        """
        Extract features. This function will update self._data and self._columns.

        Examples
        --------
        Given a dataset with three features and four samples, we let the extract_features.
        >>>_del_list=["feature1"]
        >>>df = pd.DataFrame([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        >>>df.columns=["feature1","feature2","feature3"]
        ['feature1' 'feature2' 'feature3']
        >>>extract_feature()
        ['feature1']
        """

        try:
            self._extract_list = list(filter(None, self._extract_list))
            if len(self._extract_list) == 0:
                return
            # case insensitive
            self._extract_list = self.case_insensitive(self._extract_list)
            self._columns = self.case_insensitive(self._columns)

            extract_list = self.intersect_cloumns(self._extract_list, self._columns)

            # if selected_list is empty，skip this function
            if len(extract_list) == 0:
                return
            self._data = self._data[list(extract_list)]
            self._columns = self._data.columns.tolist()
        except Exception as ex:
            self.errorlogger.logger.error("extract_features error:\n %s " % traceback.format_exc())
            self.managerlogger.logger.error("extract_features error:\n %s " % ex)

    def _store_label_properties(self, res):
        """
        Store label properties.

        Parameters
        ----------
        res: dict
            result of filter

        """
        try:
            self._valid_path(self._feature_lebel_encoder_table_file)
            self._valid_path(self._feature_lebel_encoder_file)

            res = map(lambda x: "%s = %s" % (x[0], json.dumps(x[1])), res.items())
            with open(self._feature_lebel_encoder_table_file, "w") as file:
                file.write("\n".join(res))

            pickle.dump(self._feature_lebel_encoder, open(self._feature_lebel_encoder_file, "wb"))
        except Exception as e:
            self.managerlogger.logger.warning("store label properties error: %s" % e)

    def _init(self):
        # Convert column names into lists and convert all column names to lowercase.
        if self._data is None:
            self._data = pd.read_csv(self._data_file, sep=",", encoding="utf-8",
                                     error_bad_lines=False, low_memory=True)
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError
        # case insensitive, using lowercase by default
        self._data.columns = self.case_insensitive(self._data.columns)
        self._columns = self._data.columns.tolist()

    def case_insensitive(self, list):
        """
        Parameters
        ----------
        data: list
            list need to be converted to lowercase

        Returns
        ----------
        :return: lowercase list
        """
        return [x.lower() for x in list]

    def intersect_cloumns(self, list1, list2):
        """
        Parameters
        ----------
        list1: list need intersected
        list2: list need intersected

        Returns
        ----------
        resultList: list
            sorted list
        """
        set1 = set(list1)
        set2 = set(list2)

        resultList = list(set1.intersection(set2))
        resultList.sort()
        return resultList

    def _get_list_from_conf(self, option):
        """Read options from conf, and replace ' ', split by comma"""
        res = list()
        if option in self._conf.options(self.FEATURA_FILTER):
            keyword_str = self._conf.get(self.FEATURA_FILTER, option).lower()
            res = keyword_str.replace(" ", "").split(global_pre.Global.COMMA)
        return res

    def _load_encoder_obj(self):
        """Read label encoding information from configuration file."""
        try:
            self._feature_lebel_encoder_file = self._conf.get(self.FEATURA_FILTER,
                                                              self.ENCODEE_FILE_PATH)
            self._feature_lebel_encoder_table_file = "%s%s" % (
                self._feature_lebel_encoder_file, self.ENCODER_TABLE)
            self._feature_lebel_encoder = pickle.load(open(self._feature_lebel_encoder_file, "rb"))
        except:
            self._feature_lebel_encoder = None
            self.errorlogger.logger.warning("load encoder error")

    def _valid_path(self, file_path):
        """
        Verify that the file exists.

        Parameters
        ----------
        file_path : str
        """
        if not file_path:
            raise ValueError("file path is invalid: %s" % (file_path))
        path, file_name = os.path.split(file_path)

        if not os.path.isdir(path):
            os.makedirs(path)
