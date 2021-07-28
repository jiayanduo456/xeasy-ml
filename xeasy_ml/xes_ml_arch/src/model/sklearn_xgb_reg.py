# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT
from .base_model import BaseModel
import time
import traceback
from ..ml_utils import runstatus

try:
    from xgboost import XGBRegressor
except:
    pass
    # raise ImportError("no model named xgboost")


class SklearnXGBReg(BaseModel):
    """A xgboost regress.
       Encapsulation form base model

       Parameters
       ----------
       config: the instance of ConfigParser.ConfigParser()

       Examples
       --------
       >>> from xes_ml_arch.src.model import sklearn_xgb_reg
       >>> import configparser
       >>> import pandas as pd
       >>> import numpy as np
       >>> import random
       >>> config = configparser.ConfigParser()
       >>> config.read("sklearn_xgb.conf")
       >>> x = pd.read_csv('data.txt', sep=',')
       >>> y = np.array([int(random.random() * 100) for _x in range(99)]) #label = num of samples
       >>> ins = sklearn_xgb.SklearnXGB(config=config)
       >>> ins.init()
       >>> ins.train(x, y)
       ...
       ...
    """

    MODEL_ID = BaseModel.MODEL_SKLEARN_XGB_REG

    def __init__(self, config=None, log_path = None):
        super(SklearnXGBReg, self).__init__(config=config, log_path = log_path)

    def _init_model(self):
        """Init mode"""
        self._model = XGBRegressor(**self._model_params)
        print(self._model_params)

    def train(self, x, y):
        """Model train function.

        Parameters
        ----------
        x: input sample data.
        y: label.

        Returns
        ---------
        Bool: True(train successed) or False(train faild).
        """
        try:
            t_start = time.time()
            self.managerlogger.logger.info("start xgboost.sklearn..")
            self._model.fit(x, y)
            self.managerlogger.logger.info("finished xgboost.sklearn!")
            t_end = time.time()
            self.managerlogger.logger.info("xgboost.sklearn train time: %s" % (t_end - t_start))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("xgboost.sklearn train error: %s " % err)
            self.errorlogger.logger.error("xgboost.sklearn train error:\n %s " % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def predict(self, x, thresh=0.5):
        """Forecast input data.

        Parameters
        ----------
        x：input sample data.
        thresh: label classification threshold.

        Returns
        -------
        Prediction label(list).
        """

        try:
            return self._model.predict(x)
        except Exception as err:
            self.managerlogger.logger.error("xgboost.sklearn predict error: %s " % err)
            self.errorlogger.logger.error("xgboost.sklearn predict error:\n %s " % traceback.format_exc())
            return None

    def predict_proba(self, x):
        """Calculate data predict probability value.

        Parameters
        ----------
        x: input sample data.

        Returns
        ---------
        list of probability value [[ ]]
        """
        try:
            return self._model.predict(x)
        except Exception as err:
            self.managerlogger.logger.error("xgboost.sklearn predict_proba error: %s " % err)
            self.errorlogger.logger.error("xgboost.sklearn predict_proba error:\n %s " % traceback.format_exc())
            return None

    def get_feature_importance(self, feature):
        """Get feature weights of user data.

        Parameters
        ----------
        feature: Target characteristics(string of list).

        Returns
        ---------
        if faild return None,else return the score of feature importrance.(list, [[feature, score]]).
        """
        try:
            res = zip(self._model.feature_importances_, feature)
            return res
        except Exception as err:
            self.managerlogger.logger.error("xgboost.sklearn get_feature_importance error: %s " % err)
            self.errorlogger.logger.error(
                "xgboost.sklearn get_feature_importance error:\n %s " % traceback.format_exc())
            return None
