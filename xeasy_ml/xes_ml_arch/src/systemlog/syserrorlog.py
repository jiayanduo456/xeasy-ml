# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from . import syslog


class SysErrorLog(syslog.SysLog):
    """Instantiate the system error log class"""
    def __init__(self, filename, log_path = None):
        syslog.SysLog.__init__(self, filename, name='error', path = log_path)
