# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import os
import logging
import logging.config

from ..ml_utils import runstatus
from ..ml_utils import jsonmanager


class SysLog():
    __info_dict = None

    def __init__(self, filename, name, path = None):
        """Initialization parameter information"""

        if not path:
            path = os.path.join(os.path.dirname(__file__),'../../config/log.conf')
        self.log_path = path
        logging.config.fileConfig(self.log_path)

        # test log config path
        # logging.config.fileConfig('../config/log.conf')
        self.name = name
        self.filename = filename
        self.logging = logging.getLogger(self.name)

        self.__info_dict = {}

    def __del__(self):
        if self.__info_dict is not None:
            self.__info_dict.clear()

    @property
    def logger(self):
        """According to self.name, find the corresponding log configuration in log.conf and return it"""
        return self.logging

    def info(self, message):
        """Log level: Info; Content write function.

        Parameters
        ----------
        message: Written documents(string).

        Returns
        -------
        Write message and file doctor current log level to the output log file.
        """
        self.logging.info('%s : %s' % (self.filename, message))

    def error(self, message):
        """Log level: error; Content write function.

        Parameters
        ----------
        message: Written documents(string).

        Returns
        -------
        Write message and file doctor current log level to the output log file.
        """

        self.logging.error('%s : %s' % (self.filename, message))

    def debug(self, message):
        """Log level: debug; Content write function.

        Parameters
        ----------
        message: Written documents(string).

        Returns
        -------
        Write message and file doctor current log level to the output log file.
        """
        self.logging.debug('%s : %s' % (self.filename, message))

    def warning(self, message):
        """Log level: error; Content write function.

        Parameters
        ----------
        message: Written documents(string).

        Returns
         -------
        Write message and file doctor current log level to the output log file.
        """
        self.logging.warning('%s : %s' % (self.filename, message))

    def push_info(self, info_key, info_value):
        """Judge whether the log information is repeated.

        Parameters
        ----------
        info_key：log level.
        info_value: message.

        Returns
        -------
        If it already exists, return False; Otherwise recorded in the self.__info_dict.
        """
        if info_key in self.__info_dict:
            self.warning("key[%s] already exist in log" % (info_key))
            return runstatus.RunStatus.FAILED

        self.__info_dict[info_key] = info_value

    def print_info(self):
        """Output the log under the current file to the console"""

        self.logging.info(str(self.filename) + ':' + str(self.__info_dict))

    def object_info(self, message):
        """Convert Python data structure to JSON data structure and write it to log file; log level: info"""
        json_str = jsonmanager.get_message(message)
        self.info(json_str)

    def object_debug(self, message):
        """Convert Python data structure to JSON data structure and write it to log file; log level: debug"""
        json_str = jsonmanager.get_message(message)
        self.debug(json_str)

    def object_warning(self, message):
        """Convert Python data structure to JSON data structure and write it to log file; log level: warning"""
        json_str = jsonmanager.get_message(message)
        self.warning(json_str)

    def object_error(self, message):
        """Convert Python data structure to JSON data structure and write it to log file; log level: error"""
        json_str = jsonmanager.get_message(message)
        self.error(json_str)
