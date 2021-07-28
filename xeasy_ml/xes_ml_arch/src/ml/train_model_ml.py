# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import traceback
from . import base_ml
import configparser
from ..constance.predict_constance import PredictConstance
from ..ml_utils import runstatus, global_pre
import os


class TrainML(base_ml.BaseML):
    """
    This basic class encapsulates the training of the model.

    Parameters
    --------
    conf : configparser.ConfigParser
        Model configuration file.

    Examples
    --------
    >>> from xes_ml_arch.src.ml import train_model_ml
    >>> from xes_ml_arch.src.ml_utils import global_pre
    >>> import configparser
    >>> import pandas as pd
    >>> BASE_CONFIG = "base_config"
    >>> RESULT_PATH = "result_path"
    >>> conf = configparser.ConfigParser()
    >>> ml_config = configparser.ConfigParser()
    >>> ml_config.read("./config/demo/ml_online.conf")
    >>> global_pre.RES_PATH = ml_config.get(BASE_CONFIG, RESULT_PATH)
    >>> ml = train_model_ml.TrainML(conf=ml_config)
    >>> ml.init()
    >>> ml.set_data(pd.read_csv("./data/sample.txt"))
    >>> ml.start()

    """

    def __init__(self, conf=None,xeasy_log_path = None):
        super(TrainML, self).__init__(config=conf,xeasy_log_path = xeasy_log_path)

    def start(self):
        """
        Start to train a model.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            runstatus.RunStatus.FAILD: faild
        """
        self.managerlogger.logger.info("start ml train model...")
        if not self._train_handle(None):
            # self.managerlogger.logger.error("train faild")
            return runstatus.RunStatus.FAILED
        self.managerlogger.logger.info("train succeed")

        if not self._get_result():
            # self.managerlogger.logger.error("get result faild")
            return runstatus.RunStatus.FAILED
        self.managerlogger.logger.info("get result succeed")

        if not self._store_predict_result():
            # self.managerlogger.logger.error("store result faild")
            return runstatus.RunStatus.FAILED
        self.managerlogger.logger.info("store result succeed")
        self.managerlogger.logger.info("finished ml train model!")
        return runstatus.RunStatus.SUCC

    def _train_handle(self, data):
        """
        Model train.

        Parameters
        --------
        data: pandas.DataFrame
            sample data

        Returns
        --------
        :return: bool
            True: success
            False: failed
        """
        # split data
        if not self._init_data_spliter():
            return False

        # set data and feature processor
        self._feature_processor.train_data = self._data_spliter.train_data
        self._feature_processor.test_data = self._data_spliter.test_data
        if not self._feature_processor.execute():
            return False

        # analysis data
        self._analysiser.reset(data=self._feature_processor.train_data,
                               feature_columns=self._feature_processor.data_feature_columns,
                               label_columns=self._feature_processor.data_target_column)
        if not self._analysiser.execute():
            return False

        # search for the best params
        if self._optimizing._enable_gridsearch == True:
            opt = self._optimizing.excute(self._model, self._model._model_params,
                                          self._feature_processor.train_data_feature,
                                          self._feature_processor.train_data_target)
            if not opt:
                return False
            self._model._model = self._optimizing.best_estimator_

        if runstatus.RunStatus.SUCC == self._model.train(self._feature_processor.train_data_feature,
                                                         self._feature_processor.train_data_target):
            self._model.store_model()

            # when model is dst get bintree
            try:
                tmp_file = os.path.join(global_pre.RES_PATH, "model_dest")
                self._model.show_dot(tmp_file, self._feature_processor.train_data_feature.columns)
            except:
                pass

            self._model.store_feature_importance(self._feature_processor.train_data_feature.columns)
        else:
            self.managerlogger.logger.error("model train faild")
            return False
        # Cross-validation
        self._cv_model()

        return True

    def _init_data_spliter(self):
        """
        Initialize the data set splitter.

        Returns
        --------
        :return: bool
            True: success
            False: faild
        """
        try:
            split_conf = self._conf.get(PredictConstance.BASE_CONFIG,
                                        PredictConstance.DATA_SPLITE_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(split_conf)
            self._data_spliter.reset(conf=conf, data=self._data)
            if self._data_spliter.execute() == runstatus.RunStatus.SUCC:
                self.managerlogger.logger.info("init data spliter successful!")
                return True
        except Exception as err:
            self.managerlogger.logger.error("init data spliter error: %s" % err)
            self.errorlogger.logger.error("init data spliter error:\n %s" % traceback.format_exc())
        return False

    def _cv_model(self):
        """
        Cross-validate the trained model.

        Returns
        --------
        :return: bool
            True: success
            False: fialed
        """
        self._cross_validation.reset(data=self._feature_processor.train_data,
                                     x_columns=self._feature_processor.data_feature_columns,
                                     y_column=self._feature_processor.data_target_column,
                                     id_fields=self._feature_processor.data_id_columns)

        try:
            # model_config = self._conf.get(PredictConstance.BASE_CONFIG,
            #                               PredictConstance.MODEL_CONFIG)
            # conf = ConfigParser.ConfigParser()
            # conf.read(model_config)
            # cv_model = model_factory.ModelFactory.create_model(config=conf)
            cv_model = self._model
            if self._cross_validation.execute() == runstatus.RunStatus.SUCC:
                if self._cross_validation.cv_model(cv_model):
                    self.managerlogger.logger.info("cv success")
            return True
        except Exception as err:
            self.managerlogger.logger.error("init model error: %s" % err)
            self.errorlogger.logger.error("init model error: \n %s" % traceback.format_exc())
            return False
