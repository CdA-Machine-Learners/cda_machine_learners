'''

Commonly needed functions. TODO: should probably be relocated

'''

import logging

def get_nested(config, keys):
    ''' Ex: get_nested(my_dict, ['some', 'nested', 'key'] '''
    temp = config
    for k in keys:
        if k in temp:
            temp = temp[k]
        else:
            return None
    return temp

def mk_logger(name, level):
    ''' A logger builder helper, so each feature can namespace its own logs.'''
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s => %(message)s [%(pathname)s:%(lineno)d]')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
