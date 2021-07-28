# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from . import syslog


class SysManagerLog(syslog.SysLog):
    def __init__(self, filename, log_path = None):
        """Instantiate the system manager log class"""
        syslog.SysLog.__init__(self, filename, name='system', path = log_path)
