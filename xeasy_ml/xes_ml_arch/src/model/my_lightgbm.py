# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from .base_model import BaseModel
import time
import traceback
import numpy as np
from ..ml_utils import runstatus

try:
    import lightgbm as lgb
except:
    pass


class MyLightGBM(BaseModel):
    """A lightgbm classifier. The encapsulation form is base model.

    Parameters
    --------
    config: configparser.ConfigParser()
        Configuration file for model initialization.

    Examples
    --------
    >>> from xes_ml_arch.src.model import my_lightgbm
    >>> import configparser
    >>> import pandas as pd
    >>> import numpy as np
    >>> import random
    >>> config = configparser.ConfigParser()
    >>> config.read("lightgbm.conf")
    >>> x = pd.read_csv('data.txt', sep=',')
    >>> y = np.array([int(random.random() * 100) for _x in range(99)])
    >>> ins = my_lightgbm.MyLightGBM(config=config)
    >>> ins.init()
    >>> ins.train(x, y)
    ...
    ...
    """
    GBM_N_ESTIMATORS = "num_boost_round"

    MODEL_ID = BaseModel.MODEL_LIGHTGBM

    def __init__(self, config=None, log_path = None):
        super(MyLightGBM, self).__init__(config=config, log_path = log_path)
        self._lgb_param_ = []

    def _init_model(self):
        """Initialize model."""

        self._init_config()
        self._model = None

        self._model = lgb.LGBMClassifier(**self._model_params)

    def _init_config(self):
        pass

    def train(self, x, y):
        """Train lightgbm.

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            sample data
        y: array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            t_start = time.time()
            self.managerlogger.logger.info("start lightGBM..")
            self._model.fit(x, y)

            t_end = time.time()
            self.managerlogger.logger.info("finished lightGBM!")
            self.managerlogger.logger.info("lightGBM train time: %s" % (t_end - t_start))

            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("lightGBM train error: %s " % err)
            self.errorlogger.logger.error("lightGBM train error:\n %s " % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def predict(self, x):
        """Calculate data predict value.

        Parameters
        --------
        x: pandas.DataFrame with shape (n_samples, n_features)
            Features of test data.

        Returns
        --------
        res: array like of shape (n_sample, )
            Predict result.
        """
        if self._model is None:
            return None
        # x_data = lgb.Dataset(x)
        # Label classification threshold.
        thresh =0.5
        res = self._model.predict(x)
        res=[0 if x < thresh else 1 for x in res]
        return np.array(res)

    def predict_proba(self, x):
        """Calculate data predict probability value.

        Parameters
        --------
        x: pandas.DataFrame with shape (n_samples, n_features)
            Features of test data.

        Returns
        --------
        res: list
            List of probability value belonging to a certain class.
        """

        if self._model is None:
            return None
        # x_data = lgb.Dataset(x)
        res = self._model.predict(x)
        return res

    def get_feature_importance(self, feature):
        """Get weights of user data features.

        Returns
        --------
        res: list
            Feature importance, list like : [[feature, score]].
        """
        try:
            # feature_importance = []
            # importance = self._model.get_score(importance_type='gain')
            # for key in importance:
            #     feature_importance.append([importance[key], key])
            res = zip(self._model.booster_.feature_importance(importance_type='split'), self._model.booster_.feature_name())
            #res = zip(self._model.feature_importance(), feature)

            return res
        except Exception as err:
            self.managerlogger.logger.error("lightGBM get feature importance error: %s" % err)
            self.errorlogger.logger.error("lightGBM get feature importance error:\n %s" % traceback.format_exc())
            return None

    def fit(self,x,y):
        """Training with data set."""

        return  self.train(x,y)