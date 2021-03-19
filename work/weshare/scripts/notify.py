#!/usr/bin/env python
# coding=utf-8

import sys
import base64
import requests

# https://doc.ushareit.me/web/#/4?page_id=3156

g_team_dict = {}
g_team_dict["guorui"]  = {"mail": "guorui@ushareit.com",  "phone": "18911235520"}
g_team_dict["chengtx"] = {"mail": "chengtx@ushareit.com", "phone": "18612981582"}
g_team_dict["weipan"]  = {"mail": "weipan@ushareit.com",  "phone": "15541165158"}
g_team_dict["husong"]  = {"mail": "husong@ushareit.com",  "phone": "13075565850"}
g_team_dict["chenxk"]  = {"mail": "chenxk@ushareit.com",  "phone": "18511691698"}
g_team_dict["langhb"]  = {"mail": "langhb@ushareit.com",  "phone": "17355324586"}
g_team_dict["longxb"]  = {"mail": "longxb@ushareit.com",  "phone": "13661252526"}
g_team_dict["jiangda"] = {"mail": "jiangda@ushareit.com", "phone": "18810922512"}


class Notify():
    def __init__(self, sender = "SHAREIT_ALG"):
        self.__base_url = "http://prod.openapi-notify.sgt.sg2.api/notify"

        token = self.__get_token()
        if token == None:
            raise Exception("send_mail fail! can't get token")

        self.__headers = {"Content-Type": "application/json", "Authorization": token}
        self.__sender  = sender


    def info(self, subject, message, receivers, attach_files = None):
        addrs = self.__get_addrs(receivers)
        if addrs:
            self.send_mail(subject, message, addrs, attach_files)
        else:
            print("wrong receivers! %s" % receivers)


    def error(self, subject, message, receivers, attach_files = None):
        addrs  = self.__get_addrs(receivers)
        phones = self.__get_phones(receivers)
        if addrs:
            self.send_mail("[ERROR] " + subject, message, addrs, attach_files)
            if phones:
                self.send_ding("e66e188a6aa76d0f6dfab32bda5fdf9f927bb277684c2ae40a2cafa332f938f4", "[ERROR] " + message, phones)
        else:
            print("wrong receivers! %s" % receivers)


    def fatal(self, subject, message, receivers, attach_files = None):
        addrs    = self.__get_addrs(receivers)
        phones   = self.__get_phones(receivers)
        shareids = self.__get_shareids(receivers)
        if addrs:
            self.send_mail("[FATAL] " + subject, message, addrs, attach_files)
            if phones:
                self.send_ding("e66e188a6aa76d0f6dfab32bda5fdf9f927bb277684c2ae40a2cafa332f938f4", "[FATAL] " + message, phones)
            if shareids:
                self.send_phone(shareids)
        else:
            print("wrong receivers! %s" % receivers)


    def __get_addrs(self, receivers):
        receiver_list = receivers.split(",")
        addr_list     = [g_team_dict[receiver]["mail"] for receiver in receiver_list if receiver in g_team_dict]
        if addr_list:
            return ",".join(addr_list)
        return None

    def __get_phones(self, receivers):
        receiver_list = receivers.split(",")
        addr_list     = [g_team_dict[receiver]["phone"] for receiver in receiver_list if receiver in g_team_dict]
        if addr_list:
            return ",".join(addr_list)
        return None

    def __get_shareids(self, receivers):
        receiver_list = receivers.split(",")
        shareid_list  = [receiver for receiver in receiver_list if receiver in g_team_dict]
        if shareid_list:
            return ",".join(shareid_list)
        return None


    def __get_token(self):
        try:
            req_url = "https://sentry.ushareit.me/dex/token"
    
            headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8'}
    
            req_data = {}
            req_data["username"]   = "B0BEB4EEBCD74F3CAC846313893312EA"
            req_data["password"]   = "5444B6DA95B78247652EBB925101BA6C"
            req_data["grant_type"] = "password"
            req_data["scope"]      = "openid groups"
            req_data["client_id"]  = "sgt-notify-openapi"
    
            resp = requests.post(req_url, req_data)
            if resp.status_code != 200:
                print("get_token fail! %s" % resp.text)
                return None
    
            result = resp.json()
            token = "Bearer " + result.get("id_token")
            return token
        except Exception as e:
            print("get_token error: %s" % str(e))
            return None


    def send_mail(self, subject, message, addrs, attach_files = None):
        try:
            req_url = self.__base_url + "/email/send"

            req_data = {}
            req_data["from"]    = self.__sender         # 发件人
            req_data["subject"] = subject               # 主题
            req_data["text"]    = message               # 邮件内容
            req_data["to"]      = addrs.split(",")      # 收件人
            if attach_files:
                attach_file_list = []
                for attach_file in attach_files.split(","):
                    attach_file_dict = {}
                    with open(attach_file, "rb") as fp:
                        attach_file_dict["filename"] = str(attach_file.split("/")[-1])
                        attach_file_dict["content"]  = str(base64.b64encode(fp.read()))
                    attach_file_list.append(attach_file_dict)

                req_data["attach_files"] = attach_file_list

            resp = requests.post(req_url, json = req_data, headers = self.__headers)
            if resp.status_code != 200:
                print("send_mail to %s fail! %s" % (addrs, resp.text))
            else:
                print("send_mail to %s succ. %s" % (addrs, resp.text))
        except Exception as e:
            print("send_mail error: %s" % str(e))


    def send_ding(self, access_token, message, phones):
        try:
            req_url = self.__base_url + "/dingrobot/send"

            req_data = {}
            req_data["msgtype"]      = "text"
            req_data["text"]         = {"content": message}
            req_data["at"]           = {"atMobiles": phones.split(","), "isAtAll": False}
            req_data["access_token"] = access_token

            resp = requests.post(req_url, json = req_data, headers = self.__headers)
            if resp.status_code != 200:
                print("send_ding to %s fail! %s" % (phones, resp.text))
            else:
                print("send_ding to %s succ. %s" % (phones, resp.text))

        except Exception as e:
            print("send_ding error: %s" % str(e))


    # 功能不完善，不建议使用
    def send_phone(self, shareit_ids):
        try:
            req_url = self.__base_url + "/telephone/send"

            req_data = {}
            req_data["template_id"]     = ""
            req_data["template_params"] = []
            req_data["receiver"]        = shareit_ids.split(",")

            resp = requests.post(req_url, json = req_data, headers = self.__headers)
            if resp.status_code != 200:
                print("send_phone to %s fail! %s" % (shareit_ids, resp.text))
            else:
                print("send_phone to %s succ. %s" % (shareit_ids, resp.text))
        except Exception as e:
            print("send_phone error: %s" % str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="send mail or dingding.")
    parser.add_argument("--sender", type=str, default=None)                                     # 发件人
    parser.add_argument("--level",  type=str, default="info", help="info, error support")       # 消息级别
    parser.add_argument("subject", type=str)
    parser.add_argument("message", type=str)
    parser.add_argument("receivers", type=str, help='use shareit id, multiple receiver split by comma')
    args = parser.parse_args()

    notify = Notify(args.sender)
    if args.level == "info":
        notify.info(args.subject, args.message, args.receivers)
    elif args.level == "error":
        notify.error(args.subject, args.message, args.receivers)
    else:
        print("%s not support" % args.level)
