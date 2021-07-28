# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from ..systemlog import syserrorlog

class DataSet(object):
    def __init__(self):
        self._data = None

        self._id_fields = None
        self._target_field = None
        self._feature_fields = None

        self._data_id = None
        self._data_feature = None
        self._data_label = None

    def data_set_id(self):
        if self._data is None or self._id_fields is None:
            pass
