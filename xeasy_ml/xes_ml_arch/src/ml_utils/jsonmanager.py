# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import json

def get_message(message):
    """Generating Python objects into JSON objects. Element of json is string.

    Parameters
    ----------
    message: input date,dict.

    Returns
    -------
    date of json.
    """
    # json_str = json.dumps(message, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    # json_str = json_str.replace('\n', '').replace('\t', '')
    json_str = json.dumps(message, default=lambda o: o.__dict__, sort_keys=True)
    return json_str


def get_message_without_whitespace(message):
    """Generating Python objects into JSON objects with no space. Element of json is string.

    Parameters
    ----------
    message: input date,dict.

    Returns
    -------
    date of json.
    """

    json_str = json.dumps(message, default=lambda o: o.__dict__, sort_keys=True)
    json_str = json_str.replace('\n', '').replace('\t', '').replace(' ', '')
    return json_str
