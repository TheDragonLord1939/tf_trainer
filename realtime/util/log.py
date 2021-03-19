# coding=utf8

import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import BaseRotatingHandler
from scribe_logger.logger import ScribeLogHandler
from realtime.util.conf import Conf

#----------------------------------------------------------------------------------------------------#
class MultiProcessRotatingFileHandler(TimedRotatingFileHandler):
    """
    handler for logging to a file for multi process...
    """
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False):
        TimedRotatingFileHandler.__init__(self, filename, when, interval, backupCount, encoding, delay, utc)
        self.lastRolloverAt = 0
        if not os.path.exists(self.baseFilename):
            self.dev, self.ino = -1, -1
        else:
            stat = os.stat(self.baseFilename)
            #here is a bug ???
            #self.dev, self.ino = stat[ST_DEV], stat[ST_INO]
            self.dev, self.ino = stat.st_dev, stat.st_ino

    def emit(self, record):
        """
        Emit a record.

        First check if the underlying file has changed between the given time, and if it
        has, close the old stream and reopen the file to get the
        current stream.
        """
        current_time = int(time.time())
        if current_time > self.lastRolloverAt and current_time < self.lastRolloverAt + 600:
            #TODO, at this time , we should watch the basefile
            if not os.path.exists(self.baseFilename):
                stat = None
                changed = 1
            else:
                stat = os.stat(self.baseFilename)
                changed = (stat.st_dev != self.dev) or (stat.st_ino != self.ino)
            if changed and self.stream is not None:
                self.stream.flush()
                self.stream.close()
                self.stream = self._open()
                if stat is None:
                    stat = os.stat(self.baseFilename)
                self.dev, self.ino = stat.st_dev, stat.st_ino
        BaseRotatingHandler.emit(self, record)
        
    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        #here we do not remove the dfn simply, we should guess it.
        if os.path.exists(dfn):
            stat = os.stat(dfn)
            last_mt = int(stat.st_mtime)
            current_time = int(time.time())
            if last_mt + 3600 < current_time:
                os.remove(dfn)
                os.rename(self.baseFilename, dfn)
        else:
            try:
                os.rename(self.baseFilename, dfn)
            except:
                #TODO, fix it
                pass
        if self.backupCount > 0:
            # find the oldest log file and delete it
            #s = glob.glob(self.baseFilename + ".20*")
            #if len(s) > self.backupCount:
            #    s.sort()
            #    os.remove(s[0])
            for s in self.getFilesToDelete():
                os.remove(s)
        #print "%s -> %s" % (self.baseFilename, dfn)
        #self.mode = 'w'
        self.stream = self._open()
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstNow = time.localtime(currentTime)[-1]
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    newRolloverAt = newRolloverAt - 3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    newRolloverAt = newRolloverAt + 3600
        self.lastRolloverAt = self.rolloverAt
        self.rolloverAt = newRolloverAt

#----------------------------------------------------------------------------------------------------#
def create_logger(conf, logger_name):
    log_format = getattr(conf, 'log_format', '') \
        or '%(asctime)s %(levelname)-5s %(message)s'
    console_log_level = getattr(conf, 'console_log_level', logging.ERROR) \
        or logging.ERROR
    log_level = getattr(conf, 'log_level', logging.INFO) or logging.INFO
    scribe_log_level = getattr(conf, 'scribe_log_level', log_level) \
        or log_level
    #if isinstance(scribe_log_level, basestring):
    if isinstance(scribe_log_level, str):
        scribe_log_level = eval(scribe_log_level)
    #if isinstance(console_log_level, basestring):
    if isinstance(console_log_level, str):
        console_log_level = eval(console_log_level)
    #if isinstance(log_level, basestring):
    if isinstance(log_level, str):
        log_level = eval(log_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(log_format))
    ch.setLevel(console_log_level)
    logger.addHandler(ch)

    log_file = conf.get('%s_file' % (logger_name))
    if log_file:
        fh = MultiProcessRotatingFileHandler(log_file, 'midnight')
        fh.setFormatter(logging.Formatter(log_format))
        fh.setLevel(log_level)
        logger.addHandler(fh)

    # 通过配置文本文件加载的conf, hasattr(conf, 'any_key')始终为真, 故使用getattr代替
    if getattr(conf, 'log_category', None):
        log_host = getattr(conf, 'log_host', None) or '127.0.0.1'
        log_port = int(getattr(conf, 'log_port', None) or 1464)
        scribe = ScribeLogHandler(
            category=conf.log_category, host=log_host, port=log_port)
        scribe.setLevel(scribe_log_level)
        scribe.setFormatter(logging.Formatter(log_format))
        logger.addHandler(scribe)

    # 将warning打印到日志便于分析问题
    if getattr(conf, 'log_capture_warning', None) in ['True', '1', True, 1]:
        logging.captureWarnings(True)
    return logger

#----------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    conf = Conf('test.cfg')
    test_logger = create_logger(conf, 'test_logger')
    test_logger.info('load test.log OK')
    time.sleep(1)
