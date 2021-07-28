# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import configparser

from ..systemlog import sysmanagerlog
from ..systemlog import syserrorlog


class ConfigManager():
    """Reading of log system synchronization configuration file"""

    def __init__(self, configfile, log_path = None):
        """
        Initialization parameter.

        Parameters
        --------
        configfile: System Configuration file path of user.
        """

        self.configfile = configfile
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,log_path)
        self.configP = self.init_config(self.configfile, self.managerlogger)

    def init_config(self, configfile, logger):
        """
        Loading configuration information.

        Parameters
        ----------
        configfile：System Configuration file path

        Returns
        -------
        Configuration information file.
        """
        self.managerlogger.logger.info('Start innit the config.....')

        conf = configparser.ConfigParser()
        conf.read(configfile)

        self.managerlogger.logger.info('End init the config.....')
        return conf

    def get_key(self, group, key):
        """Gets the option value of the named part"""
        return self.configP.get(group, key)

    def get_keys(self, group):
        """Gets the content of the configuration file section(group)；
            contens: tuple of list.

        Parameters
        ----------
        group: section name of configuration file.

        Returns
        -------
        tuple of list
        """

        return self.configP.items(group)

    def get_sections(self):
        return self.configP.sections()

    def get_float(self, section, option):
        return self.configP.getfloat(section, option)

    def has_option(self, section, option):
        return self.configP.has_option(section, option)

    def __iter__(self):
        for section in self.configP.sections():
            yield section
