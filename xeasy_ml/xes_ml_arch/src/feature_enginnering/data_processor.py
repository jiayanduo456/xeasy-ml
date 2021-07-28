# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from ..systemlog import sysmanagerlog, syserrorlog
from . import data_washer
from . import feature_filter
from . import data_sampler
from . import xes_onehot_encoder
from . import pre_feature_utils
from ..ml_utils import runstatus
from ..ml_utils import global_pre
import configparser
import traceback


class DataProcessor(object):
    """
    Processor train data or test data.

    Parameters
    --------
    conf : configparser.ConfigParser, default = None
        Contains configuration information such as data set path, id field, target field, data processing method, etc.

    Attributes
    --------
    train_data : pandas.DataFrame, default = None
        The raw text data to learn. A 2D table, each row represents a sample, and each column represents
        the characteristics of a given sample.
    train_data_id : str, default = None
        The id field of train sample.
    train_data_feature : list, default = None
        The feature list of train sample.
    train_data_target : str, default = None
        The target field of train sample.

    test_data : pandas.DataFrame, default = None
        Test samples.
    test_data_id : str, default = None
        The id field of test sample.
    test_data_feature : list, default = None
        The feature list of test sample.
    test_data_target : str, default = None
        The target field of test sample.

    data_id_columns : list, default = None
        Id columns.
    data_feature_columns : list, default = None
        Feature columns.
    data_target_column : str, default = None
        Target column.
    _conf : configparser.ConfigParser, default = None
        Contains configuration information such as data set path, id field, target field, data processing
        method, etc.
    _feature_filter : :class : '~feature_filter.FeatureFilter'
        Feature selection, drop unused features and feature label.
    _data_sample : :class : '~data_sampler.DataSampler'
        Data sampling of train data or test data.
    _feature_washer : :class : '~data_washer.DataWasher'
        Sort column, zscore.
    _onehot_encoder : :class : '~xes_onehot_encoder.XESOneHotEncoder'
        Feature selection, drop unused features and feature label.

    Examples
    --------
    >>> from xes_ml_arch.src.feature_enginnering import data_processor
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> feature_processor = data_processor.DataProcessor(conf=conf)
    >>> feature_processor.init()
    >>> train_data = pd.read_csv("my_traindata.csv")
    >>> test_data = pd.read_csv("my_testdata.csv")
    >>> feature_processor.train_data = train_data
    >>> feature_processor.test_data = test_data
    >>> feature_processor.execute()
    """
    # main config path
    default_config_file = "../config/demo/feature_enginnering.conf"
    # test config path
    # default_config_file = "./conf/feature_enginnering.conf"
    BASE_CONFIG = "base_config"
    ID_FIELDS = "id_fields"
    TARGET_FIELDS = "target_fields"
    TRAIN_DATA_FILE = "train_data_file"
    TEST_DATA_FILE = "test_data_file"

    def __init__(self, conf=None,log_path = None):

        self.xeasy_log_path = log_path
        # train data, 3 parts: id, feature, target
        self.train_data = None
        self.train_data_id = None
        self.train_data_feature = None
        self.train_data_target = None

        # test data, 3 parts: id, feature, target
        self.test_data = None
        self.test_data_id = None
        self.test_data_feature = None
        self.test_data_target = None

        # id columns, feature columns, target column
        self.data_id_columns = None
        self.data_feature_columns = None
        self.data_target_column = None

        # config
        self._conf = conf

        # data processor object
        self._feature_filter = None
        self._data_sample = None
        self._feature_washer = None
        self._onehot_encoder = None

        self.managerlogger= sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Init feature processor object, including data sampling, data washing, feature preprocessing, etc.

        Returns
        --------
        :return : bool , runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        Notes
        --------
        If there is no usable conf parameter, read class default_config_file.
        """
        if not isinstance(self._conf, configparser.ConfigParser):
            self.managerlogger.logger.error("conf error: conf is not ConfigParser instance")
            config = configparser.ConfigParser()
            config.read(self.default_config_file)
            self._conf = config
        try:
            self.data_id_columns = self._conf.get(self.BASE_CONFIG, self.ID_FIELDS).lower().split(
                global_pre.Global.COMMA)
            self.data_target_column = self._conf.get(self.BASE_CONFIG, self.TARGET_FIELDS).lower()

            if not self._init_data_sample() or not self._init_data_washer() or \
                    not self._init_feature_filter() or not self._init_pre_feature() or \
                    not self._init_onehot_encoder():
                self.managerlogger.logger.error("feature processor init error")
                return runstatus.RunStatus.FAILED

            return runstatus.RunStatus.SUCC
        except Exception as ex:
            self.managerlogger.logger.error("data processor object init erorr: %s" % ex)
            self.errorlogger.logger.error("data processor object init erorr \n" + traceback.format_exc())
        return runstatus.RunStatus.FAILED

    """
    # set_data no use
    def set_data(self, data):
        # :param data:
        # :return: runstatus.RunStatus
        if not pre_utils.PredictUtils.valid_pandas_data(data):
            self.managerlogger.logger.logger.info("data is illegal")
            return runstatus.RunStatus.FAILED
        self._data = data
        return runstatus.RunStatus.SUCC
    """

    def execute(self):
        """
        Start data processor.

        Returns
        --------
        :return: bool, runstatus.RunStatus
             runstatus.RunStatus.SUCC: True
             runstatus.RunStatus.FAILED: Failed
        """
        if self.train_data is None and (self.test_data is not None and self._test_data_handle()) \
                or (self.train_data is not None and self._train_data_handle() and
                    self.test_data is not None and self._test_data_handle()):
            self.managerlogger.logger.info("data processor succeed! ")
            return runstatus.RunStatus.SUCC
        else:
            self.managerlogger.logger.error("data processor error! ")
            return runstatus.RunStatus.FAILED

    def _init_pre_feature(self):
        # init class object PreFeatureUtils.
        self._feature_preprocesser = pre_feature_utils.PreFeatureUtils(conf=self._conf, log_path = self.xeasy_log_path)
        if self._feature_preprocesser.init() == runstatus.RunStatus.SUCC:
            return True
        else:
            self.managerlogger.logger.error("init feature preprocesser error")
            return False

    def _init_data_washer(self):
        # init class object DataWasher.
        self._feature_washer = data_washer.DataWasher(conf=self._conf, log_path = self.xeasy_log_path)
        if self._feature_washer.init() == runstatus.RunStatus.SUCC:
            return True
        else:
            self.managerlogger.logger.error("init data washer error")
            return False

    def _init_feature_filter(self):
        # init class object FeatureFilter.
        self._feature_filter = feature_filter.FeatureFilter(conf=self._conf, log_path = self.xeasy_log_path)
        if self._feature_filter.init() == runstatus.RunStatus.SUCC:
            return True
        else:
            self.managerlogger.logger.error("init feature filter error")
            return False

    def _init_data_sample(self):
        # init class object DataSampler.
        self._data_sample = data_sampler.DataSampler(conf=self._conf,
                                                     log_path = self.xeasy_log_path,
                                                     target=self.data_target_column)
        if self._data_sample.init() == runstatus.RunStatus.SUCC:
            return True
        else:
            self.managerlogger.logger.error("init data sample error")
            return False

    def _init_onehot_encoder(self):
        # init class object XESOneHotEncoder.
        self._onehot_encoder = xes_onehot_encoder.XESOneHotEncoder(None, conf=self._conf,log_path = self.xeasy_log_path)
        if self._onehot_encoder.init() == runstatus.RunStatus.SUCC:
            return True
        else:
            self.managerlogger.logger.error("init data sample error")
            return False

    def _train_data_handle(self):
        """
        Handle train data. Including feature preprocessing, data filtering, converting features
        into one hot encoding, data sampling and data washer.
        After these processes, we can get the data for training

        Returns
        --------
        :return: bool
            True: succeed
            False: failed
        """
        try:
            # feature preprocessing
            if self._feature_preprocesser.set_data(self.train_data) == runstatus.RunStatus.FAILED:
                return False
            self.train_data = self._feature_preprocesser.excute()

            # filter data
            if self._feature_filter.set_data(self.train_data) == runstatus.RunStatus.FAILED:
                return False
            self.train_data = self._feature_filter.excute()
            if self.train_data is None:
                self.managerlogger.logger.error("data filter error")
                return False

            # Converting features into one hot encoding.
            self._onehot_encoder.set_data(self.train_data)
            if self._onehot_encoder.execute():
                self.train_data = self._onehot_encoder.get_data()

            # store processed data
            self._store_data(self.TRAIN_DATA_FILE, self.train_data)

            # sample data
            if self._data_sample.set_data(self.train_data) == runstatus.RunStatus.FAILED:
                return False
            if self._data_sample.excute() == runstatus.RunStatus.FAILED:
                self.managerlogger.logger.error("data sample faild")
                return False

            # split train data to 3 parts
            self.train_data = self._data_sample.get_data()
            self.train_data_id = self.train_data[self.data_id_columns]
            self.train_data_target = self.train_data[self.data_target_column]
            self.train_data_feature = self.train_data.drop(self.data_id_columns, 1).drop(
                self.data_target_column, 1)

            # data washer
            self._feature_washer.set_train_flag()
            self._feature_washer.set_data(self.train_data_feature)
            self.train_data_feature = self._feature_washer.excute()

            # get train data columns
            self.data_feature_columns = self.train_data_feature.columns
            if self.train_data_feature is not None:
                return True

        except Exception as err:
            self.managerlogger.logger.error("train data processor train data handle error: %s" % err)
            self.errorlogger.logger.error("_train_data_handle error: %s" % traceback.format_exc())

    def _test_data_handle(self):
        """
        Handle test data
        Returns
        --------
        :return: bool
            True : success
            False : faild
        """
        try:
            # feature preprocessing
            if self._feature_preprocesser.set_data(self.test_data) == runstatus.RunStatus.FAILED:
                return False
            self.test_data = self._feature_preprocesser.excute()

            # filter data
            self._feature_filter.set_data(self.test_data)
            self.test_data = self._feature_filter.excute()

            # onehot
            self._onehot_encoder.set_data(self.test_data)
            if self._onehot_encoder.execute():
                self.test_data = self._onehot_encoder.get_data()

            # store processed data
            self._store_data(self.TEST_DATA_FILE, self.test_data)

            # split test data to 3 parts
            self.test_data_id = self.test_data[self.data_id_columns]
            self.test_data_feature = self.test_data.drop(self.data_id_columns, 1)
            if self.data_target_column in self.test_data.columns:
                self.test_data_feature = self.test_data_feature.drop(self.data_target_column, 1)
                self.test_data_target = self.test_data[self.data_target_column]

            # data washer
            self._feature_washer.set_test_flag()
            self._feature_washer.set_data(self.test_data_feature)
            self.test_data_feature = self._feature_washer.excute()

            # get test data columns
            self.data_feature_columns = self.test_data_feature.columns

            if self.test_data_feature is not None:
                return True
        except Exception as err:
            self.managerlogger.logger.error("test data processor data error: %s" % err)
            self.errorlogger.logger.error("_test_data_handle error: %s" % traceback.format_exc())
            return False

    def _store_data(self, optios, data):
        """
        Store processed data to file.
        Parameters
        ----------
        optios: str
            file to store, config options.
        data: pandas.DataFrame
            processed data.

        Returns
        ---------
        :return: bool
            True: succ
            False: failed
        """
        try:
            path = self._conf.get(self.BASE_CONFIG, optios)
            data.to_csv(path, index=False)
            return True
        except Exception as ex:
            self.managerlogger.logger.error("store data error: %s" % ex)
            self.errorlogger.logger.error("_store_data error:  %s " % traceback.format_exc())
        return False
