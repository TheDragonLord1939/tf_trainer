#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.header import Header

# set from=noreply@ushareit.com
# set smtp=smtp.ushareit.com
# set smtp-auth-user=noreply@ushareit.com
# set smtp-auth-password=mhGtM67XXUf2uN7X
# set smtp-auth=login

class MailSender:
    def __init__(self):
        self.__fromaddr = 'noreply@ushareit.com'
        self.__username = 'noreply@ushareit.com'
        self.__password = 'mhGtM67XXUf2uN7X'

    # @param | list   | recevier_list    | 收件人列表
    # @param | string | subject          | 邮件主题
    # @parma | string | text_message     | 邮件正文内容
    # @param | list   | attach_file_list | 附件列表
    def send(self, recevier_list, subject, text_message, attach_file_list):
        mail = MIMEMultipart() 

        mail['From']    = self.__fromaddr                    # 发件人
        mail['To']      = ";".join(recevier_list)            # 收件人
        mail['Subject'] = subject                            # 邮件主题

        mail.attach(MIMEText(text_message, 'html', 'utf-8'))# 邮件正文内容

        for attach_file in attach_file_list:
            attach_message = MIMEText(open(attach_file, 'rb').read(), 'base64', 'utf-8')
            attach_message["Content-Type"]        = 'application/octet-stream'
            attach_message["Content-Disposition"] = 'attachment; filename="%s"' % attach_file.split("/")[-1]
            mail.attach(attach_message)
        
        try:
            self.__server = SMTP_SSL('smtp.ushareit.com', 465)
            self.__server.login(self.__username, self.__password)
            self.__server.sendmail(self.__fromaddr, recevier_list, mail.as_string())
            self.__server.quit()
        except Exception as e:
            print('connect smtp server error: %s' % str(e))
            return -1
        return 0

# @param | string | receviers    | 收件人，多个邮箱地址使用,分割
# @param | string | subject      | 邮件主题
# @parma | string | msg          | 邮件正文内容
# @param | list   | attach_files | 附件列表，多个文件使用,分割
def sendmail(receviers, subject, msg = None, attach_files = None):
    recevier_list = receviers.split(",")

    # 待发送邮件内容
    msg_text = ""
    if msg:
       msg_text = msg 

    # 待发送文件
    attach_file_list = []
    if attach_files:
        attach_file_list = attach_files.split(",")

    mail = MailSender()
    mail.send(recevier_list, subject, msg_text, attach_file_list)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("receviers and subject needed!")
    elif len(sys.argv) == 3:
        sendmail(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        sendmail(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        sendmail(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    
