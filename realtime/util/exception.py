#!/usr/bin/env python
# coding=utf-8
# author: weianjun(weiaj@shareit.com)
# date: 2020.11.26

class FileNotFound(Exception):
    def __init__(self, file_name):
        self.file_name = file_name

    def __str__(self):
        return 'file=[%s] not found' % (self.file_name)

class KeyNotFound(Exception):
    def __init__(self, key_name):
        self.key_name = key_name

    def __str__(self):
        return 'key=[%s] not found' % (self.key_name)
