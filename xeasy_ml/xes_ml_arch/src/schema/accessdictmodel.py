# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import json
import datetime
from ..ml_utils import configmanager


class AccessDictModel():
    """Get config profile information"""
    def __init__(self, log_path = None):
        self.acc_Logger_Dict = {}
        self.log_option_list = list()
        self.log_option_property_dict = dict()

        #Configuration file instance
        standard_log_conf = configmanager.ConfigManager('./config/log_schemas.conf', log_path)

        #get options
        for k, v in standard_log_conf.get_keys('default_schema'):
            self.log_option_list.append(k)
            self.log_option_property_dict[k] = v
        self.reset()

    def format_body(self):
        """Convert dictionary type data to JSON type.

        Returns
        -------
        josn string.
        """

        json_str = json.dumps(self.acc_Logger_Dict, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        json_str = json_str.replace('\n', '').replace('\t', '')
        return json_str

    def get_log_dic(self):
        return self.acc_Logger_Dict

    def set_log_dic_key(self, key, value):
        self.acc_Logger_Dict[key] = value
        self.set_log_dic_key_time(key)

    def set_log_dic_key_time(self, key):
        key_time = str(key) + '_time'
        time_value = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.acc_Logger_Dict[key_time] = time_value

    def set_data_log_dic_key(self, req_or_resp, key, value):
        for data_dict in self.acc_Logger_Dict['data']:
            if data_dict.get(req_or_resp) is None:
                continue
            else:
                inner_dict = data_dict.get(req_or_resp)
                inner_dict[key] = value
                break

    def reset(self):
        """The configuration information is converted into a dictionary whose value represents the data type of
        the configuration information;For example:'string'== V, self.acc_ Logger_ Dict[k] = ''.

        Returns
        -------
        the dict of configuration information data type.
        """

        for k in self.log_option_property_dict:
            v = self.log_option_property_dict[k]
            if 'int' == v:
                self.acc_Logger_Dict[k] = 0
            elif 'string' == v:
                self.acc_Logger_Dict[k] = ''
            elif 'dict' == v:
                if self.acc_Logger_Dict.get(k) is None:
                    self.acc_Logger_Dict[k] = dict()
                else:
                    self.acc_Logger_Dict[k].clear()
            elif 'list' == v:
                if 'data' == k:
                    if self.acc_Logger_Dict.get(k) is None:
                        self.acc_Logger_Dict[k] = list()
                        request_dict = dict()
                        request_dict["req"] = dict()
                        self.acc_Logger_Dict[k].append(request_dict)
                        response_dict = dict()
                        response_dict["resp"] = dict()
                        self.acc_Logger_Dict[k].append(response_dict)
                    else:
                        for data_dict in self.acc_Logger_Dict[k]:
                            inner_dict = data_dict.get('req')
                            if inner_dict is not None:
                                inner_dict.clear()
                            inner_dict = data_dict.get('resp')
                            if inner_dict is not None:
                                inner_dict.clear()
                else:
                    if self.acc_Logger_Dict.get(k) is None:
                        self.acc_Logger_Dict[k] = list()
                    else:
                        del self.acc_Logger_Dict[k][:]
            else:
                self.acc_Logger_Dict[k] = None
