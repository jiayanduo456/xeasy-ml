# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from ..systemlog import sysmanagerlog, syserrorlog
from ..ml_utils import runstatus
from ..ml_utils import global_pre
import configparser
import traceback
import pandas as pd
from ..ml_utils import pre_utils
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
try:
    from xgboost import XGBClassifier
except:
    pass

class Optimizing():

    OPTIMIZING_CONF = "optimizing_conf"
    ENABLE_GRIDSEARCH = "enable_gridSearch"
    CV = "cv"
    N_JOBS = "n_jobs"

    def __init__(self,config = None,log_path = None):
        # config
        self._conf= config
        self.xeasy_log_path = log_path
        # model params
        self._cv = 5
        self._n_jobs = 1
        self._model_params = {}
        self._estimator = None
        self._enable_gridsearch = None
        self._best_params = None
        self.best_estimator_ = None
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__, self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__, self.xeasy_log_path)

    def init(self):
        """Read optimizer parameter configuration information

        Returns
        -------
        Boolean: if "enable_gridSearch","cv", "n_jobs" exists return runstatus.RunStatus.SUCC
                otherwise runstatus.RunStatus.FAILED.
        """
        if not isinstance(self._conf, configparser.ConfigParser):
            self.managerlogger.logger.error("conf error: conf is not ConfigParser instance")
            config = configparser.ConfigParser()
            config.read(self.default_config_file)
            self._conf = config
        try:
            self._enable_gridsearch = eval(self._conf.get(self.OPTIMIZING_CONF, self.ENABLE_GRIDSEARCH))
            self._cv = eval(self._conf.get(self.OPTIMIZING_CONF, self.CV))
            self._n_jobs = eval(self._conf.get(self.OPTIMIZING_CONF, self.N_JOBS))
            return runstatus.RunStatus.SUCC
        except Exception as ex:
            self.managerlogger.logger.error("optimizing object init erorr: %s" % ex)
            self.errorlogger.logger.error("optimizing object init erorr \n" + traceback.format_exc())
        return runstatus.RunStatus.FAILED



    def excute(self, estimator = None, params = None, x = None, y = None ):
        """Model training search for the best params.

        Parameters
        ----------
        estimator: base model.
        params: parameter list which to search.
        x: input data.
        y: label.

        Returns
        -------
        The best value of the parameter in the parameter list.
        """

        self._estimator = estimator._model
        self._model_params = params
        try:
            grid = RandomizedSearchCV(self._estimator,self._model_params, cv=self._cv, n_jobs=self._n_jobs)
            grid.fit(x,y)
            if grid:
                self._best_params = grid.best_params_
                self.best_estimator_ = grid.best_estimator_
                self.managerlogger.logger.info("optimizing excute succeed! ")
                self.managerlogger.logger.info(grid.best_params_)
                return runstatus.RunStatus.SUCC
            else:
                self.managerlogger.logger.error("optimizing excute error! ")
                return runstatus.RunStatus.FAILED
        except Exception as ex:
            self.managerlogger.logger.error("optimizing excute erorr: %s" % ex)
            self.errorlogger.logger.error("optimizing excute erorr \n" + traceback.format_exc())
        return runstatus.RunStatus.FAILED
