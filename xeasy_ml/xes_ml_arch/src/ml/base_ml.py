# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import pandas as pd
import configparser
import traceback
import os
from ..analysis import analysis
from ..cross_validation import cross_validation, data_split
from ..feature_enginnering import data_processor
from ..systemlog import sysmanagerlog, syserrorlog
from ..constance.predict_constance import PredictConstance
from ..model import model_factory
from ..optimizing import optimizing
from ..ml_utils import runstatus, pre_utils, global_pre


class BaseML(object):
    """
    This basic class is a base framework for machine learning, used to solve a machine learning problem.

    Parameters
    --------
    conf : configparser.ConfigParser
        Model configuration file.

    Attributes
    ----------
    _data: pandas.DataFrame
        Machine learning data set.

    _conf: config object, instance of ConfigParser.ConfigParser

    _model: ml nodel object lr,rf,xgb...

    _predict_res: ml predict result

    _feature_processor: feature processor object

    _analysiser: ml analysiser

    _cross_validation: ml cross validation

    _data_spliter: data split object

    _result_path: the path to store ml result

    _log: system log

    _logerror: error debug log
    """

    def __init__(self, config=None,xeasy_log_path = None):
        # data
        self._data = None
        self.xeasy_log_path = xeasy_log_path

        # config  object
        self._conf = config

        # result path
        self._result_path = None

        # result
        self._predict_res = None

        # feature enginnering object
        self._feature_processor = None

        # model object
        self._model = None

        # analysis object
        self._analysiser = None

        # cross validataion object
        self._cross_validation = None

        # data spliter object
        self._data_spliter = data_split.DataSplit(log_path = self.xeasy_log_path)

        # optimizing
        self._optimizing = None
        # log object
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def init(self):
        """
        Initialize model,feature processer andanalysiser.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            runstatus.RunStatus.FAILD: faild
        """
        try:
            self.managerlogger.logger.info("start ml...")
            if not isinstance(self._conf, configparser.ConfigParser):
                return runstatus.RunStatus.FAILED

            # store path for results
            self._result_path = global_pre.RES_PATH
            # unittest
            # self._result_path = '../result'

            if not os.path.isdir(self._result_path):
                os.mkdir(self._result_path)

            # init model, feature engineering and analysiser
            if self._init_model() and self._init_feature_processer()  and self._init_analysiser() and \
                    self._init_cv_ins() and self._init_optimizing():
                self.managerlogger.logger.info("ml init success")
                return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("machine_learning init error: %s" % err)
            self.errorlogger.logger.error("machine_learning init error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def start(self):
        pass

    def set_feature_processor_ins(self, instance):
        """
        Set feature processor object.

        Parameters
        --------
        instance: data_processor.DataProcessor
            Instance of data_processor.DataProcessor, to process data

        Returns
        --------
        :return: runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            return runstatus.RunStatus.FAILED: faild
        """
        if not isinstance(instance, data_processor.DataProcessor):
            self.managerlogger.logger.error("instance is not type of DataProcessor")
            return runstatus.RunStatus.FAILED
        self._feature_processor = instance
        return runstatus.RunStatus.SUCC

    def set_analysis_ins(self, instance):
        """
        Set analysiser.

        Parameters
        --------
        instance: analysis.Analysis
            Used to analysis a data set

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            return runstatus.RunStatus.FAILED: faild
        """
        if not isinstance(instance, analysis.Analysis):
            self.managerlogger.logger.error("instance is not type of analysis.Analysis")
            return runstatus.RunStatus.FAILED
        self._analysiser = instance
        return runstatus.RunStatus.SUCC

    def set_optimizing_ins(self, instance):
        """
        Set the optimizer.

        Parameters
        --------
        instance: optimizing.Optimizing
            Used to search the best params.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            return runstatus.RunStatus.FAILED: faild
        """
        if not isinstance(instance, optimizing.Optimizing):
            self.managerlogger.logger.error("instance is not type of optimizing.Optimizing")
            return runstatus.RunStatus.FAILED
        self._optimizing = instance
        return runstatus.RunStatus.SUCC

    def set_cross_validation(self, instance):
        """
        Set cross-validation object.

        Parameters
        --------
        instance: cross_validation.Cross_Validation
            Used to cross-validation.
        """
        if not isinstance(instance, cross_validation.Cross_Validation):
            self.managerlogger.logger.error("instance is not type of cross_validation.Cross_Validation")
        self._cross_validation = instance

    def set_data(self, data):
        """
        Set data.

        Parameters
        --------
        data: pandas.DataFrame
            Data set.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            runstatus.RunStatus.FAILED: failed
        """
        if not isinstance(data, pd.DataFrame):
            return runstatus.RunStatus.FAILED
        self._data = data
        return runstatus.RunStatus.SUCC

    def _init_model(self):
        """
        Initialize model object.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild
        """
        try:
            model_config = self._conf.get(PredictConstance.BASE_CONFIG,
                                          PredictConstance.MODEL_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(model_config)
            self._model = model_factory.ModelFactory.create_model(config=conf)
            return True
        except Exception as err:
            self.managerlogger.logger.error("init model error: %s" % err)
            self.errorlogger.logger.error("init model error: \n %s" % traceback.format_exc())
            return False

    def _init_optimizing(self):
        """
        Initialize _init_optimizing object.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild
        """
        try:
            model_config = self._conf.get(PredictConstance.BASE_CONFIG,
                                          PredictConstance.MODEL_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(model_config)
            self._optimizing = optimizing.Optimizing(conf,log_path = self.xeasy_log_path)
            if self._optimizing.init() == runstatus.RunStatus.SUCC:
                return True
        except Exception as err:
            self.managerlogger.logger.error("init optimizing error: %s" % err)
            self.errorlogger.logger.error("init optimizing error: \n %s" % traceback.format_exc())
            return False


    def _init_feature_processer(self):
        """
        Initialize feature processer object.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild
        """
        try:
            model_config = self._conf.get(PredictConstance.BASE_CONFIG,
                                          PredictConstance.FEATURE_ENGINEERING_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(model_config)
            self._feature_processor = data_processor.DataProcessor(conf=conf,log_path = self.xeasy_log_path)
            if self._feature_processor.init() == runstatus.RunStatus.SUCC:
                return True
            else:
                return False
        except Exception as err:
            self.managerlogger.logger.error("init model error: %s" % err)
            self.errorlogger.logger.error("init model error:\n %s" % traceback.format_exc())
            return False

    def _init_analysiser(self):
        """
        Initialize analysiser.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild

        """
        try:
            analysis_config = self._conf.get(PredictConstance.BASE_CONFIG,
                                             PredictConstance.ANALYSIS_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(analysis_config)
        except Exception as err:
            self.managerlogger.logger.error("analysiser config error: %s" % err)
            return False
        self._analysiser = analysis.Analysis(conf=conf,log_path = self.xeasy_log_path)
        return True

    def _init_cv_ins(self):
        """
        Initialize cross validation.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild
        """
        try:
            cross_val_conf = self._conf.get(PredictConstance.BASE_CONFIG,
                                            PredictConstance.CV_CONFIG)
            conf = configparser.ConfigParser()
            conf.read(cross_val_conf)
        except Exception as err:
            self.managerlogger.logger.error("init cv error: %s" % err)
            return False
        self._cross_validation = cross_validation.Cross_Validation(conf=conf,log_path = self.xeasy_log_path)
        return True

    def _get_result(self):
        """
        Get predict result.

        Returns
        --------
        :return: bool
            True: Success
            False: Faild
        """
        try:
            # get test data
            test_id = self._feature_processor.test_data_id
            test_feature = self._feature_processor.test_data_feature
            test_target = self._feature_processor.test_data_target

            # process data
            test_feature = test_feature.astype("float64", errors='ignore')

            # predict
            predict_res = self._model.predict(test_feature)
            predict_res_df = pd.DataFrame(predict_res, columns=[PredictConstance.PRE])
            proba_res = self._model.predict_proba(test_feature)
            proba_res_df = pd.DataFrame([str(x) for x in proba_res],
                                        columns=[PredictConstance.PROBA])

            res = [test_id, predict_res_df, proba_res_df]
            # get model score
            if test_target is not None:
                res.append(test_target)
                model_auc = pre_utils.PredictUtils.get_roc_score(test_target, proba_res)
                model_score = pre_utils.PredictUtils.get_model_score(test_target, predict_res)
                model_score.update(model_auc)
                with open(os.path.join(self._result_path, PredictConstance.TEST_SCORE), "w") as ftp:
                    ftp.write(str(model_score))

            # joint predict result
            self._joint_predict_result(res)

            return True
        except Exception as err:
            self.managerlogger.logger.error("base ml get result error: %s" % err)
            self.errorlogger.logger.error("base ml get result error:\n %s" % traceback.format_exc())
            return False

    def _joint_predict_result(self, result):
        """
        Joint the result.

        Parameters
        --------
        result: list(pandas.Dataframe)
            Results of prediction.

        Returns
        --------
        :return: bool
            True: Suceess
            False: Faild
        """
        if len(result) == 0:
            return
        try:
            [x.reset_index(drop=True, inplace=True) for x in result]
            self._predict_res = pd.concat(result, axis=1)
            return True
        except Exception as err:
            self.managerlogger.logger.error("joint_predict_result error: %s" % err)
            self.errorlogger.logger.error("joint_predict_result error:\n %s" % traceback.format_exc())
            return False

    def _store_predict_result(self):
        """
        Store result to file.

        Returns
        --------
        :return: bool
            True: Suceess
            False: Faild
        """
        try:
            self._predict_res.to_csv(os.path.join(self._result_path, PredictConstance.PREDICT_FILE),
                                     index=False)
            return True
        except Exception as err:
            self.managerlogger.logger.error("joint_predict_result error: %s" % err)
            self.errorlogger.logger.error("joint_predict_result error:\n %s" % traceback.format_exc())

            return False

    def _train_handle(self, data):
        """
        Train a model.

        Parameters
        --------
        data: pandas.DataFrame
            Input data.
        """
        pass

    def _predict_handle(self):
        """
        Predict handle.
        """
        pass
