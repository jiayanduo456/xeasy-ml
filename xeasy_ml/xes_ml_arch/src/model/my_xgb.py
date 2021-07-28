# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from .base_model import BaseModel
import time
import traceback
import numpy as np
from ..ml_utils import runstatus

try:
    import xgboost as xgb
except:
    pass


class MyXgb(BaseModel):
    """A xgboost classifier.

    Parameters
    -----------
    config: configparser.ConfigParser(), default = None
            Model configuration file.

    Examples
    -----------
       >>> from xes_ml_arch.src.model import my_xgb
       >>> import configparser
       >>> import pandas as pd
       >>> import numpy as np
       >>> import random
       >>> config = configparser.ConfigParser()
       >>> config.read("xgb.conf")
       >>> x = pd.read_csv('data.txt', sep=',')
       >>> y = np.array([int(random.random() * 100) for _x in range(99)])
       >>> ins = my_xgb.MyXgb(config=config)
       >>> ins.init()
       >>> ins.train(x, y)
       ...
       ...
    """

    XGB_N_ESTIMATORS = "num_boost_round"
    MODEL_ID = BaseModel.MODEL_XGB

    def __init__(self, config=None, log_path = None):
        super(MyXgb, self).__init__(config=config, log_path = log_path)
        self._xgb_param = [('eval_metric', 'auc'), ('eval_metric', 'ams@0')]

    def _init_model(self):
        """Init model"""

        self._init_config()
        self._model = None

    def _init_config(self):
        pass

    def train(self, x, y):
        """Model train func.

        Parameters
        ----------
        x: input data.
        y: label.

        Returns
        -------
        bool: True(train suceessed) or False(train faild)
        """

        try:
            t_start = time.time()
            self.managerlogger.logger.info("start xgboost..")
            train_data = xgb.DMatrix(x, label=y, missing=-999)
            num_round = 10
            if self.XGB_N_ESTIMATORS in self._xgb_param:
                num_round = self._xgb_param[self.XGB_N_ESTIMATORS]
                self._xgb_param.pop(self.XGB_N_ESTIMATORS)
            bst = xgb.train(self._model_params, train_data, num_round, [])
            t_end = time.time()
            self.managerlogger.logger.info("finished xgboost!")
            self.managerlogger.logger.info("xgboost train time: %s" % (t_end - t_start))

            self._model = bst
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("xgboost train error: %s " % err)
            self.errorlogger.logger.error("xgboost train error:\n %s " % traceback.format_exc())
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

        if self._model is None:
            return None
        x_data = xgb.DMatrix(x)
        res = self._model.predict(x_data)
        res = [0 if x < thresh else 1 for x in res]
        return np.array(res)

    def predict_proba(self, x):
        """Calculate data predict probability value.

        Parameters
        ----------
        x: input sample data.

        Returns
        ---------
        list of probability value [[ ]]
        """

        if self._model is None:
            return None
        x_data = xgb.DMatrix(x)
        return self._model.predict(x_data)

    def get_feature_importance(self, feature):
        """Get feature weights of user data.

        Parameters
        ----------
        feature: Target characteristics(string of list).

        Returns
        ---------
        the score of feature importrance.(list, [[feature, score]]).
        """

        try:
            feature_importance = []
            importance = self._model.get_score(importance_type='gain')
            for key in importance:
                feature_importance.append([importance[key], key])
            return feature_importance
        except Exception as err:
            self.managerlogger.logger.error("xgboost get feature importance error: %s" % err)
            self.errorlogger.logger.error("xgboost get feature importance error:\n %s" % traceback.format_exc())
            return None
