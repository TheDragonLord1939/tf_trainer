# coding=utf8

import os
import re
from realtime.util.exception import *

#----------------------------------------------------------------------------------#
class ConfLoader(object):
    def __init__(self, filename):
        self.filename = filename

    def parse(self, ):
        kv = dict()
        if not os.path.exists(self.filename):
            return kv
        with open(self.filename, mode='r', encoding='utf8') as f:
            base_dir = os.path.dirname(os.path.abspath(self.filename))
            for lineno, l in enumerate(f):
                if re.match('^\s*#', l):
                    continue
                elements = l.strip().split()
                if not elements:
                    continue
                if len(elements) < 2:
                    errmsg = 'format incorrect: lineno:%s %s' % (lineno, l)
                    raise ConfException(errmsg)
                key, value = self.clean_elem(elements)
                if key == 'include':
                    conf_loader = ConfLoader(
                        os.path.join(base_dir, value)
                        )
                    kv.update(conf_loader.parse())
                else:
                    kv[key] = value
        return kv

    def clean_elem(self, elements):
        key = elements[0]
        value = ' '.join(elements[1:])
        return key.strip(), value.strip()

#----------------------------------------------------------------------------------#
class Conf(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise FileNotFound(filename)
        self.conf_loader = ConfLoader(filename)
        self.conf = self.conf_loader.parse()
        self.local_conf = {}
        
    def get_values(self, key):
        val = self.local_conf.get(key) or self.conf.get(key)
        if val:
            return [p.strip() for p in val.split(',')]
        return []

    def get(self, key, val=''):
        value = self.local_conf.get(key) or self.conf.get(key)
        return value or val

    def get_all(self):
        all_kv = self.conf
        all_kv.update(self.local_conf)
        return all_kv

    def set(self, key, val):
        self.conf[key] = val
 
    def __getattr__(self, name):
        try:
            return super(Conf, self).__getattr__(name)
        except:
            return self.get(name)

#----------------------------------------------------------------------------------#
if __name__ == '__main__':
    c = Conf('test.cfg')
    print(c.get_all())

#end
