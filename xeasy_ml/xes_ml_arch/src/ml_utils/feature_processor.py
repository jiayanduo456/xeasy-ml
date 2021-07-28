# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import pandas as pd
import time
import datetime
import numpy as np
from ..ml_utils.global_pre import Global


class FeatureProcessor(object):
    """Handle one feature."""

    LOCATION = "feature_processor.FeatureProcessor."
    TIME_FORM = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def convert_to_int(param):
        """Convert one feature to int.

        Parameters
        ----------
        param: input data to be converted.

        Returns
        -------
        if param's type is int, return param; else if int(param) can be implement return int(param);
        else return 0.

        Examples
        -------
        >>>param = 4
        >>>convert_to_int(param)
        4

        >>>param = 4.4
        >>>convert_to_int(param)
        4

        >>>param = 'test'
        >>>convert_to_int(param)
        0
        """
        if isinstance(param, int):
            return param
        else:
            try:
                return int(param)
            except:
                return 0

    @staticmethod
    def convert_to_str(param):
        """Convert one feature to str.

        Parameters
        ----------
        param: input data to be converted.

        Returns
        -------
        if param's type is str, return param; else if str(param) can be implement return str(param);
        else return ''.

        Examples
        -------
        >>>param = 'test'
        >>>convert_to_str(param)
        'test'

        >>>param = 3
        >>>convert_to_str(param)
        '3'

        >>>param = True
        >>>convert_to_str(param)
        ''
        """

        if isinstance(param, str):
            return param
        else:
            try:
                return str(param)
            except:
                return ''

    @staticmethod
    def time2stamp(data):
        """Time convert to timestamp.

        Parameters
        -------
        data: 1d array-like; Every element in it is a time series string.
                pandas.core.series.Series.

        Returns
        -------
        list of Seconds to represent the floating-point number of time.

        Examples
        -------
        >>> data[i] = "2019-02-18 23:40:00"
        >>> time2stamp(data)[i]
        1550504400.0
        """

        if not isinstance(data, pd.core.series.Series):
            return None
        for i in range(len(data)):
            try:
                # 转换成时间数组
                timeArray = time.strptime(str(data[i]), "%Y-%m-%d %H:%M:%S")
                # 转换成时间戳
                timestamp = time.mktime(timeArray)
                data[i] = timestamp
            except:
                data[i] = None
        return data

    @staticmethod
    def stamp2time(data):
        """Timestamp convert to time.

        Parameters
        -------
        data:  A list of timestamp date(float).

        Returns
        -------
        A list of time(str).

        Examples
        -------
        >>> data[i] = 1550504400.0
        >>> stamp2time(data)[i]
        "2019-02-18 23:40:00"
        """

        if not isinstance(data, pd.core.series.Series):
            return None
        try:
            data = data.astype('int')
        except:
            return
        for i in range(len(data)):
            try:
                # 转换成localtime
                time_local = time.localtime(int(data[i]))
                # 转换成新的时间格式(2016-05-05 20:28:54)
                dt = time.strftime(Global.TIME_FORM, time_local)
                data[i] = dt
            except:
                data[i] = None
        return data

    @staticmethod
    def time_diff(data1, data2):
        """Real time convert to timastamp then Calculating the difference.

        Parameters
        -------
        data1: 1d arrat-like; Every element in it is a time series string.
                pandas.core.series.Series
        data2: 1d arrat-like; Every element in it is a time series string.
                pandas.core.series.Series

        Returns
        -------
        _diff: 1d array-like; Time difference of corresponding time(float).
                np.array.

        Examples
        -------
        >>>data1 = pd.date_range('20200101',periods=6, freq='1h30min')
        >>>data2 = pd.date_range('20190101',periods=6, freq='1h45min')
        >>>time_diff(data1, data2)
        array([31536000., 31535100., 31534200., 31533300., 31532400., 31531500.])
        """

        if not (isinstance(data1, pd.core.series.Series) and
                isinstance(data2, pd.core.series.Series)):
            return None
        if len(data1) != len(data2):
            return None
        _diff = np.zeros(len(data1))
        for i in range(len(data1)):
            try:
                # 转换成时间数组
                timeArray1 = time.strptime(str(data1[i]), Global.TIME_FORM)
                # 转换成时间戳
                timestamp1 = time.mktime(timeArray1)
                # 转换成时间数组
                timeArray2 = time.strptime(str(data2[i]), Global.TIME_FORM)
                # 转换成时间戳
                timestamp2 = time.mktime(timeArray2)
                _diff[i] = int(abs(int(timestamp1) - int(timestamp2)))
            except:
                _diff[i] = None
        return _diff

    @staticmethod
    def minus_data(data1, data2):
        """Time difference.

        Parameters
        ----------
        data1: 1d arrat-like; Every element in it is a timestamp float.
                pandas.core.series.Series
        data2: 1d arrat-like; Every element in it is a timestamp float.
                pandas.core.series.Series

        Returns
        -------
        result data1 - data2
        """
        if not (isinstance(data1, pd.core.series.Series) and
                isinstance(data2, pd.core.series.Series)):
            return None
        if len(data1) != len(data2):
            return None

        _res = np.zeros(len(data1))
        for i in range(len(data1)):
            try:
                _res[i] = float(data1[i]) - float(data2[i])
            except:
                _res[i] = None
        return _res

    @staticmethod
    def abs_minus_data(data1, data2):
        """Absolute value of time difference.

        Parameters
        ----------
        data1: 1d arrat-like; Every element in it is float.
                pandas.core.series.Series
        data2: 1d arrat-like; Every element in it is float.
                pandas.core.series.Series

        Returns
        -------
        result |data1 - data2|
        """
        if not (isinstance(data1, pd.core.series.Series) and isinstance(data2,
                                                                        pd.core.series.Series)):
            return None
        if len(data1) != len(data2):
            return None
        _res = np.zeros(len(data1))
        for i in range(len(data1)):
            try:
                _res[i] = abs(float(data1[i]) - float(data2[i]))
            except:
                _res[i] = None
        return _res

    @staticmethod
    def stay_in_year(years):
        """Converts str(years type) to int.

        Parameters
        ----------
        years:1d array-like; Every element in it is str with '+'.
                pandas.core.element.Series.

        Returns
        -------
        res: 1d array-like; Every element in it is a year series int while year can be convert to int else 0.
              pandas.core.series.Series.
        """

        res = []
        for year in years:
            try:
                res.append(int(year.replace("+", "")))
            except:
                return res.append(0)
        return res

    @staticmethod
    def age2int(ages):
        """Convert str(age type) to int.

        Parameters
        ----------
        ages: 1d array-like;Every element in it is string.
             pandas.core.series.Series.

        Returns
        -------
        res: 1d array-like;Every element in it is int.
             pandas.core.series.Series.
        """

        res = []
        for age in ages:
            try:
                if "-" in age:
                    age = [int(x) for x in age.split("-")]
                    res.append(float(sum(age)) / len(age))
                elif "+" in age:
                    res.append(int(age.replace("+", "")))
                else:
                    res.append(int(age))
            except:
                res.append(-1)
        return res

    @staticmethod
    def discretize_freque(value_list, args=3):
        """Segment evenly according to the input value.

        Parameters
        ----------
        value_list: 1d array-like; pandas.core.series.Series.
        args: Segment parameter(int).

        Returns
        -------
        Segment label; 1d array-like.

        Examples
        -------
        >>>list = [2,345345,34,5,34,53,4,3,53,45,3,5]
        >>>discretize_freque(list,args = 3)
        [0, 2, 1, 1, 1, 2, 0, 0, 2, 2, 0, 1]
        """
        value_list = list(value_list)
        length = len(value_list)
        tmp_res = [[index, value_list[index], 0] for index in range(length)]
        tmp_res.sort(key=lambda x: x[1])
        class_nums = int(float(args))
        res = [int((x * class_nums) / length) for x in range(length)]
        for index in range(len(tmp_res)):
            tmp_res[index][2] = res[index]

        tmp_res.sort(key=lambda x: x[0])
        tmp_res = [x[2] for x in tmp_res]

        return tmp_res

    @staticmethod
    def get_day_range(day_num):
        """ Range of days.

        Parameters
        ----------
        day_num: max days.

        Returns
        -------
        max range of days. int.
        """
        try:
            day = int(day_num)
            if 7 < day < 31:
                return 30
            elif day <= 7:
                return day
            elif 31 <= day < 366:
                return 365
            elif 366 <= day < 731:
                return 730
            else:
                return 1095
        except:
            return -1
