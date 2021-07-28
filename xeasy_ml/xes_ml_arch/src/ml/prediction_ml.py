# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from . import base_ml
import traceback
from ..ml_utils import runstatus


class PredictionML(base_ml.BaseML):
    """
    This basic class encapsulates the functions of the prediction part, and you can call the method
    of the class to make predictions on the test set.

    Parameters
    --------
    conf : configparser.ConfigParser, default = None
        Configuration file for prediction of the test data set.

    Examples
    --------
    >>> from xes_ml_arch.src.ml import prediction_ml
    >>> import configparser
    >>> import pandas as pd
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> pml = prediction_ml.PredictionML(conf=conf)
    >>> data = pd.read_csv("my_data.csv")
    >>> pml.set_data(data)
    >>> pml.start()
    """

    def __init__(self, conf=None,xeasy_log_path = None):
        self._test_data = None
        super(PredictionML, self).__init__(config=conf, xeasy_log_path = xeasy_log_path)

    def start(self):
        """
        Start predict data handle.
        """
        self.managerlogger.logger.info("start ml predict...")
        if runstatus.RunStatus.SUCC == self._predict_handle():
            self.managerlogger.logger.info("finished ml predict!")
        else:
            self.managerlogger.logger.error("ml predict failed!")

    def _init_model(self):
        """
        Load the trained model.

        Returns
        -------
        :return: bool
            True : Succ
            False : failed
        """
        if not super(PredictionML, self)._init_model():
            return False
        # load model
        if runstatus.RunStatus.FAILED == self._model.load_model():
            self.managerlogger.logger.error("load model error")
            return False
        self.managerlogger.logger.info("successfly load model to predict: %s" % self._model.MODEL_ID)
        return True

    def _predict_handle(self):
        '''
        Model predict handle.

        Returns
        -------
        :return: bool
            True : Succ
            False : failed
        '''
        try:
            self._feature_processor.test_data = self._data
            if runstatus.RunStatus.FAILED == self._feature_processor.execute():
                self.managerlogger.logger.error("predict feature processor error")
                return False
            self.managerlogger.logger.info("successfly predict model: %s" % self._model.MODEL_ID)
            # get predict result
            if runstatus.RunStatus.FAILED == self._get_result():
                self.managerlogger.logger.error("predict get result error")
                return False
            self.managerlogger.logger.info("successfly get result of predict : %s" % self._model.MODEL_ID)

            # store result to file
            if runstatus.RunStatus.FAILED == self._store_predict_result():
                self.managerlogger.logger.error("store predict result error")
                return False
            self.managerlogger.logger.info("successfly store result of predict : %s" % self._model.MODEL_ID)
            return True

        except:
            self.managerlogger.logger.debug(traceback.format_exc())
            self.managerlogger.logger.error("predict handle error")
            return False