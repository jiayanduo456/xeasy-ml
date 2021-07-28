# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from .base_model import SklearnModel
from sklearn import svm
import time
import traceback
from ..ml_utils import runstatus


class LinerSVR(SklearnModel):
    """A liner classifier.
        Encapsulation form sklearn model.

    Parameters
    ----------
    config: the instance of ConfigParser.ConfigParser().

    Examples
    --------
        >>> from xes_ml_arch.src.model import linear
        >>> import configparser
        >>> import pandas as pd
        >>> import numpy as np
        >>> import random
        >>> config = configparser.ConfigParser()
        >>> config.read("liner.conf")
        >>> x = pd.read_csv('data.txt', sep=',')
        >>> y = np.array([int(random.random() * 100) for _x in range(99)])
        >>> ins = linear.Liner(config=config)
        >>> ins.init()
        >>> ins.train(x, y)
        ...
        ...
    """

    def __init__(self, config=None, log_path = None):
        super(LinerSVR, self).__init__(config=config, log_path = log_path)
        self._n_estimators = 200
        self._max_feature = "auto"
        self._max_deepth = None

    def _init_model(self):
        """Init model.

        Returns
        --------
        Boolean: True while suceess else False.
        """
        try:
            self._init_config()
            self._model = svm.LinearSVR(**self._model_params)
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("init linear model error: %s" % err)
            self.errorlogger.logger.error("init linear model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _init_config(self):
        pass

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
            self.managerlogger.logger.info("start linear..")
            self._model.fit(x, y)
            self.managerlogger.logger.info("finished liner!")
            t_end = time.time()
            self.managerlogger.logger.info("liner train time: %s " % (t_end - t_start))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("linear train error: %s " % err)
            self.errorlogger.logger.error("linear train error:\n %s " % traceback.format_exc())
            return runstatus.RunStatus.FAILED

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
            res = zip(self._model.coef_, feature)
            return res
        except Exception as err:
            self.managerlogger.logger.error("linear get feature importance error: %s" % err)
            self.errorlogger.logger.error(
                "linear get feature importance error:\n %s" % traceback.format_exc())
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
        return self._model.predict(x)
