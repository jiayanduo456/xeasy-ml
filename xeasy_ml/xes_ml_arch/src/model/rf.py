# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from .base_model import SklearnModel
from sklearn.ensemble import RandomForestClassifier
import time
import traceback
from ..ml_utils import runstatus


class RF(SklearnModel):
    """A RandomForestClassifier classifier.
       Encapsulation form sklearn model

       Parameters
       ----------
       config: the instance of ConfigParser.ConfigParser().

       Examples
       ---------
       >>> from xes_ml_arch.src.model import rf
       >>> import configparser
       >>> import pandas as pd
       >>> import numpy as np
       >>> import random
       >>> config = configparser.ConfigParser()
       >>> config.read("rf.conf")
       >>> x = pd.read_csv('data.txt', sep=',')
       >>> y = np.array([int(random.random() * 100) for _x in range(99)])
       >>> ins = rf.RF(config=config)
       >>> ins.init()
       >>> ins.train(x, y)
       ...
       ...
    """

    def __init__(self, config=None, log_path = None):
        super(RF, self).__init__(config=config, log_path = log_path)
        self._n_estimators = 200
        self._max_feature = "auto"
        self._max_deepth = None

    def _init_model(self):
        """Init model.

        Returns
        -------
        Boolean: True(train successed) or False(train faild)
        """

        try:
            self._model = RandomForestClassifier(**self._model_params)
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("init rf model error: %s" % err)
            self.errorlogger.logger.error("init rf model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def train(self, x, y):
        """Model train func.

        Parameters
        ----------
        x: input data.
        y: label.

        Returns
        -------
        Boolean: True(train suceessed) or False(train faild)
        """

        try:
            t_start = time.time()
            self.managerlogger.logger.info("start RF..")
            self._model.fit(x, y)
            self.managerlogger.logger.info("finished RF!")
            t_end = time.time()
            self.managerlogger.logger.info("RF train time: %s" % (t_end - t_start))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("rf train error: %s" % err)
            self.errorlogger.logger.error("rf train error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def get_feature_importance(self, feature):
        """Get feature weights of user data.

        Parameters
        ----------
        feature: Target characteristics(string of list).

        Returns
        ---------
        The score of feature importrance.(list, [[feature, score]]).
        """

        try:
            res = zip(self._model.feature_importances_, feature)
            return res
        except Exception as err:
            self.managerlogger.logger.error("rf get feature importance error: %s" % err)
            self.errorlogger.logger.error("rf get feature importance error:\n %s" % traceback.format_exc())
            return None
