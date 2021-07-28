# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


from .base_model import SklearnModel
from sklearn.linear_model import LogisticRegression
import time
import traceback
from ..ml_utils import runstatus


class LR(SklearnModel):
    """A LogisticRegression classifier. The encapsulation form is sklearn model.

    Parameters
    --------
    config: configparser.ConfigParser()
        Configuration file for model initialization.

    Examples
    --------
    >>> from xes_ml_arch.src.model import lr
    >>> import configparser
    >>> import pandas as pd
    >>> import numpy as np
    >>> import random
    >>> config = configparser.ConfigParser()
    >>> config.read("lr.conf")
    >>> x = pd.read_csv('data.txt', sep=',')
    >>> y = np.array([int(random.random() * 100) for _x in range(99)])
    >>> ins = lr.LR(config=config)
    >>> ins.init()
    >>> ins.train(x, y)
    ...
    ...
    """
    def __init__(self, config=None, log_path = None):
        SklearnModel.__init__(self, config=config, log_path = log_path)

    def _init_model(self):
        """Initialize model.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            self._model = LogisticRegression(**self._model_params)
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("init lr model error: %s" % err)
            self.errorlogger.logger.error("init lr model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def train(self, x, y):
        """Train a model use a LogisticRegression model.

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            sample data
        y: array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            runstatus.RunStatus.Faild: Faild
        """
        try:
            t_start = time.time()
            self.managerlogger.logger.info("start lr..")
            self._model.fit(x, y)
            self.managerlogger.logger.info("finished lr!")
            t_end = time.time()
            self.managerlogger.logger.info("lr train time: %s" % (t_end - t_start))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("lr train error: %s " % err)
            self.errorlogger.logger.error("lr train error:\n %s " % traceback.format_exc())
            return runstatus.RunStatus.FAILED
