# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


from . import syslog


class AccessLog(syslog.SysLog):
    """Instantiate the access log class"""
    def __init__(self, filename, log_path = None):
        syslog.SysLog.__init__(self, filename, name='access', path = log_path)
