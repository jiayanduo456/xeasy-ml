# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import traceback
import os
import pandas as pd
from ..systemlog import sysmanagerlog, syserrorlog
from ..ml_utils import runstatus
from ..model import base_model
from ..ml_utils import pre_utils, global_pre
from ..constance.predict_constance import PredictConstance


class Cross_Validation(object):
    """
    Cross-validation class, and the methods provided include: training set test set division,
    cross training set test set division, model cross-validation effect.
.
    Parameters
    --------
    conf : configparser.ConfigParser, default = None
        Configuration information.
    data : pandas.DataFrame, default = None
        Data used for cross-validation.
    x_columns : List, default = None
        Name list of feature fields.
    y_column : str, default = ""
        Name of label field.
    id_fields : list, default = []
        Name list of id_fields.

    Attributes
    --------
    _k_fold : int, default = 7
        The value of k-fold cross-validation. Generally speaking, if the training data set is relatively small, increase the value of k. If the
        training set is relatively large, decrease the value of k.
    _is_executed : bool, default = False
        Cross-validation flag.

    Examples
    --------
    >>> from xes_ml_arch.src.cross_validation import cross_validation
    >>> from xes_ml_arch.src.model import model_factory
    >>> import configparser
    >>> import pandas as pd
    >>> import numpy as np
    >>> import random
    >>> conf = configparser.ConfigParser()
    >>> conf.read("myconfig.conf")
    >>> model_conf = configparser.ConfigParser()
    >>> model_conf.read("model_config.conf")
    >>> model_name = 'sklearn_xgb'
    >>> columns = ["col%s" % x for x in range(10)]
    >>> x = columns[:9]
    >>> y = columns[9]
    >>> data = pd.DataFrame(
    >>>           [[int(random.random() * 100) for _x in range(10)] for _y in range(1000)],
    >>>            columns=columns)
    >>> data["col9"] = np.random.choice([0, 1], size=1000)
    >>> model = model_factory.ModelFactory.create_model(model_conf, model_name=model_name)
    # this
    >>> ins = cross_validation.Cross_Validation(conf=conf, data=data, x_columns=x, y_column=y)
    >>> ins.execute()
    >>> ins.cv_model(model)
    # or this
    >>> ins = cross_validation.Cross_Validation(conf=conf)
    >>> ins.reset(data=data,x_columns=x,y_column=y)
    >>> ins.execute()
    >>> ins.cv_model(model)
    """

    CROSS_VALIDATION = "cross_validation"
    RATIO = "train_test_tatio"
    K_FOLD = "k_fold"
    RESULT_FILE = "result_file"
    TRAIN_FILE = "train_file"
    TEST_FILE = "test_file"
    PREDICT_FILE = "predict_file"
    VERIFICATION_MODE = 'mode'
    IS_SHUFFLE = 'is_shuffle'
    GROUP = 'group'
    SEED = 'seed'
    VALID = 'valid'

    @staticmethod
    def _valid_data(data):
        """
        Static function to check data.

        Returns
        --------
        :return: bool
            True: succeed
            False: failed
        """
        if data is None:
            return False
        if not isinstance(data, pd.DataFrame):
            return False
        if data.shape[0] == 0:
            return False
        return True

    def __init__(self, conf=None,log_path = None, data=None, x_columns=None, y_column="", id_fields=list()):
        self._conf = conf
        self._data = data
        self.xeasy_log_path = log_path
        self._x_columns = x_columns
        self._y_column = y_column

        self._cv_data_list = []
        # k fold default:7
        self._k_fold = 7

        self._is_executed = False

        self._id_fields = id_fields

        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def reset(self, conf=None, data=None, x_columns=None, y_column="", id_fields=list()):
        """
        Set conf, data, x_columns and y_columns. If the value is default, then pass

        Parameters
        --------
        conf : configparser.ConfigParser, default = None
            Configuration information.
        data : pandas.DataFrame, default = None
            Data used for cross-validation.
        x_columns : List, default = None
            Name list of feature fields.
        y_column : str, default = ""
            Name of label field.
        id_fields : list, default = []
            Name list of id_fields.
        """
        if conf is not None:
            self._conf = conf
        if data is not None:
            self._data = data
        if x_columns is not None:
            self._x_columns = x_columns
        if y_column != "":
            self._y_column = y_column
        if len(id_fields) > 0:
            self._id_fields = id_fields
        self._is_executed = False

    def cv_model(self, model):
        """
        Model cross-validation effect.

        Parameters
        --------
        model: model object implementing 'fit'
            The object to use to fit the data.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if not isinstance(model, base_model.BaseModel):
            self.managerlogger.logger.error("mdoel is not BaseModel")
            return runstatus.RunStatus.FAILED

        try:
            cv_result = []
            all_true_label = []
            all_pre_label = []
            all_prob_label = []
            all_base_info = pd.DataFrame(columns=self._id_fields + [self._y_column])
            try:
                mode = self._conf.get(self.CROSS_VALIDATION, self.VERIFICATION_MODE)
            except TypeError:
                mode = 'k_fold'
            # calculate k parts score
            if mode == 'muti_k_fold':
                for num, data in enumerate(self._cv_data_list):
                    # train and pre
                    try:
                        if self._conf.get(self.CROSS_VALIDATION, self.VALID):
                            model.train((data[0][self._x_columns], data[1][self._x_columns]),
                                        (data[0][self._y_column], data[1][self._y_column]))
                        else:
                            model.train(data[0][self._x_columns], data[0][self._y_column])
                    except TypeError:
                        model.train(data[0][self._x_columns], data[0][self._y_column])
                    # store model
                    cur_test = num // 5
                    pre_prob_result = model.predict_proba(self._muti_cv_list[cur_test][self._x_columns])
                    pre_class_result = model.predict(self._muti_cv_list[cur_test][self._x_columns])

                    # append result
                    all_prob_label.extend(pre_prob_result.tolist())
                    all_pre_label.extend(pre_class_result.tolist())
                    all_true_label.extend(self._muti_cv_list[cur_test][self._y_column].tolist())
                    all_base_info = all_base_info.append(
                        self._muti_cv_list[cur_test][self._id_fields + [self._y_column]])

                    # calculate auc and score
                    model_auc = pre_utils.PredictUtils.get_roc_score(
                        self._muti_cv_list[cur_test][self._y_column].tolist(),
                        pre_prob_result)
                    model_score = pre_utils.PredictUtils.get_model_score(
                        self._muti_cv_list[cur_test][self._y_column].tolist(), pre_class_result)
                    # keep result
                    model_score.update(model_auc)
                    cv_result.append(model_score)
            else:
                for train_data, test_data in self._cv_data_list:
                    # train and pre
                    try:
                        if self._conf.getboolean(self.CROSS_VALIDATION, self.VALID):
                            model.train((train_data[self._x_columns], test_data[self._x_columns]),
                                        (train_data[self._y_column], test_data[self._y_column]))
                        else:
                            model.train(train_data[self._x_columns], train_data[self._y_column])
                    except TypeError:
                        model.train(train_data[self._x_columns], train_data[self._y_column])
                    pre_prob_result = model.predict_proba(test_data[self._x_columns])
                    pre_class_result = model.predict(test_data[self._x_columns])

                    # append result
                    all_prob_label.extend(pre_prob_result.tolist())
                    all_pre_label.extend(pre_class_result.tolist())
                    all_true_label.extend(test_data[self._y_column].tolist())
                    all_base_info = all_base_info.append(test_data[self._id_fields + [self._y_column]])

                    # calculate auc and score
                    model_auc = pre_utils.PredictUtils.get_roc_score(test_data[self._y_column].tolist(),
                                                                     pre_prob_result)
                    model_score = pre_utils.PredictUtils.get_model_score(
                        test_data[self._y_column].tolist(), pre_class_result)

                    # keep result
                    model_score.update(model_auc)
                    cv_result.append(model_score)

            predict_res_df = pd.DataFrame(all_pre_label, columns=[PredictConstance.PRE])
            proba_res_df = pd.DataFrame([str(x) for x in all_prob_label],
                                        columns=[PredictConstance.PROBA])

            # store predict to file
            res = [all_base_info, predict_res_df, proba_res_df]
            [x.reset_index(drop=True, inplace=True) for x in res]
            predict_res = pd.concat(res, axis=1)
            output_file = self._conf.get(self.CROSS_VALIDATION, self.PREDICT_FILE)
            output_file = os.path.join(global_pre.RES_PATH, output_file)
            predict_res.to_csv(output_file, index=False)

            # calculate all parts score
            all_auc = pre_utils.PredictUtils.get_roc_score(all_true_label, all_prob_label)
            all_score = pre_utils.PredictUtils.get_model_score(all_true_label, all_pre_label)

            all_score.update(all_auc)
            cv_result.append(all_score)

            # keep cv result
            output_file = self._conf.get(self.CROSS_VALIDATION, self.RESULT_FILE)
            output_file = os.path.join(global_pre.RES_PATH, output_file)
            with open(output_file, "w") as file_handle:
                file_handle.write("\n".join([str(x) for x in cv_result]))

            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error.error("cv model error:\n %s" % traceback.format_exc())
            self.managerlogger.logger.error("cv model error: %s" % err)
            return runstatus.RunStatus.FAILED

    def execute(self):
        """
        Construct a data set for K-fold cross-validation.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: True
            runstatus.RunStatus.FAILED: Failed
        """
        if self._is_executed:
            return runstatus.RunStatus.SUCC
        if not Cross_Validation._valid_data(self._data):
            return runstatus.RunStatus.FAILED
        try:
            if self._conf.get(self.CROSS_VALIDATION, self.IS_SHUFFLE):
                self._shuf_data()
        except TypeError:
            self.managerlogger.logger.info(
                "%s not found %s, default apply shuffle" % (self.CROSS_VALIDATION, self.IS_SHUFFLE))
        try:
            try:
                mode = self._conf.get(self.CROSS_VALIDATION, self.VERIFICATION_MODE)
                if mode == 'k_fold':
                    self._k_fold_data()
                elif mode == 'group_k_fold':
                    try:
                        group = self._conf.get(self.CROSS_VALIDATION, self.GROUP)
                        if len(group) != 1:
                            self.managerlogger.logger.info("group length must be 1, use default mode kfold")
                            self._k_fold_data()
                        elif group not in self._id_fields:
                            self.managerlogger.logger.info(
                                "%s not found in colums, use default mode kfold" % group)
                            self._k_fold_data()
                        else:
                            self._group_k_fold_data(group)
                    except:
                        self.managerlogger.logger.info(
                            "group not found, use default mode kfold")
                        self._k_fold_data()
                elif mode == 'train_test_split':
                    self._train_test_split()
                elif mode == 'muti_k_fold':
                    self._muti_k_fold_data()
                else:
                    self.managerlogger.logger.info(
                        "%s not found mode name, use default mode kfold" % self.CROSS_VALIDATION)
                    self._k_fold_data()
            except:
                self.managerlogger.logger.info(
                    "mode not found, use default mode kfold")
                self._k_fold_data()
            self._is_executed = True
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("cross validation error: %s" % err)
            self.managerlogger.logger.error.error("cross validation error\n" + traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def _train_test_split(self):
        """
        devide data to only 1 parts according to ratio
        :return:
        """
        try:
            ratio = self._conf.get(self.CROSS_VALIDATION, self.RATIO)
            if ratio <= 0 or ratio >= 1:
                ratio = 0.2
                self.managerlogger.logger.info(
                    "%s must be bigger than 0 and smaller than 1 %s, use default value 0.2" % (
                    self.CROSS_VALIDATION, self.RATIO))
        except TypeError:
            ratio = 0.2
            self.managerlogger.logger.info(
                "%s not found %s, use default value 0.2" % (self.CROSS_VALIDATION, self.RATIO))
        if not Cross_Validation._valid_data(self._data):
            raise ValueError("train data is not valid")
        # devide data
        end_index = int(self._data.shape[0] * ratio)
        train_data = self._data.iloc[:end_index]
        test_data = self._data.iloc[end_index:]
        self._cv_data_list = [(train_data, test_data)]

    def _k_fold_data(self):
        """
        Divide data into k parts.
        """
        try:
            k = int(self._conf.get(self.CROSS_VALIDATION, self.K_FOLD))
            if k <= 1:
                k = 7
                self.managerlogger.logger.info(
                    "%s must be greater than 1 %s, use default value 7" % (self.CROSS_VALIDATION, self.K_FOLD))
        except TypeError:
            k = 7
            self.managerlogger.logger.info(
                "%s not found %s, use default value 7" % (self.CROSS_VALIDATION, self.K_FOLD))
        if not Cross_Validation._valid_data(self._data):
            raise ValueError("train data is not valid")
        # devide data to k parts
        length = self._data.shape[0]
        range_list = [(int((i * length) / k), int(((i + 1) * length) / k)) for i in range(k)]

        # keep k's parts result, [(train, test),]
        self._cv_data_list = []
        for star_index, end_index in range_list:
            test_data = self._data.iloc[star_index:end_index]
            train_data = self._data.iloc[:star_index].append(self._data.iloc[end_index:])
            self._cv_data_list.append((train_data, test_data))

    def _group_k_fold_data(self, group):
        """
        devide data to k parts
        :return:
        """
        try:
            k = int(self._conf.get(self.CROSS_VALIDATION, self.K_FOLD))
            if k <= 1:
                k = 7
                self.managerlogger.logger.info(
                    "%s must be greater than 1 %s, use default value 7" % (self.CROSS_VALIDATION, self.K_FOLD))
        except TypeError:
            k = 7
            self.managerlogger.logger.info(
                "%s not found %s, use default value 7" % (self.CROSS_VALIDATION, self.K_FOLD))
        if not Cross_Validation._valid_data(self._data):
            raise ValueError("train data is not valid")
        # get groups
        group_list = self._data[group].unique()
        group_index = []
        for value in group_list:
            group_index.append(self._data[self._data[group] == value].index)
        # divide data according to group/k
        self._cv_data_list = []
        for i in range(k):
            range_list = []
            for j in range(len(group_list)):
                cur_group_index = group_index[j]
                length = len(cur_group_index)
                range_list.extend(cur_group_index[int((i * length) / k) : int(((i + 1) * length) / k)])
            test_data = self._data.iloc[range_list]
            train_data = self._data.iloc[~self._data.index.isin(range_list)]
            self._cv_data_list.append((train_data, test_data))

    def _muti_k_fold_data(self):
        """
        devide data to k * k parts
        :return:
        """
        try:
            k = int(self._conf.get(self.CROSS_VALIDATION, self.K_FOLD))
            if k <= 1:
                k = 5
                self.managerlogger.logger.info(
                    "%s must be greater than 1 %s, use default value 5" % (self.CROSS_VALIDATION, self.K_FOLD))
        except TypeError:
            k = 5
            self.managerlogger.logger.info(
                "%s not found %s, use default value 5" % (self.CROSS_VALIDATION, self.K_FOLD))
        if not Cross_Validation._valid_data(self._data):
            raise ValueError("train data is not valid")
        # devide data to k parts
        self._cv_data_list = []
        self._muti_cv_list = []
        length = self._data.shape[0]
        range_list = []
        for i in range(k):
            range_list.append((int((i * length) / k), int(((i + 1) * length) / k)))
            new_list = list(range(0, int((i * length) / k)))
            new_list.extend(list(range(int(((i + 1) * length) / k), length)))
            new_length = len(new_list)
            for j in range(k):
                star_index = int((j * new_length) / k)
                end_index = int(((j + 1) * new_length) / k)
                test_data = self._data.iloc[new_list[star_index : end_index]]
                train_data = self._data.iloc[new_list[:star_index]].append(self._data.iloc[new_list[end_index:]])
                self._cv_data_list.append((train_data, test_data))
        for star_index, end_index in range_list:
            test_data = self._data.iloc[star_index:end_index]
            self._muti_cv_list.append(test_data)


    def _shuf_data(self):
        """Shuffle the data set."""
        # self._data = shuffle(self._data)
        try:
            seed = int(self._conf.get(self.CROSS_VALIDATION, self.SEED))
            self._data = self._data.sample(frac=1, random_state=seed).reset_index(drop=True)
        except:
            self._data = self._data.sample(frac=1).reset_index(drop=True)
