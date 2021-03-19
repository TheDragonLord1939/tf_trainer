"""
author: SPRS_weipan
Copyright (c) Shareit.com Co., Ltd. All Rights Reserved.
"""
#!/usr/bin/python
#coding=utf-8
import os
import sys
import argparse


def read_embedding_file(embedding_file, langs):
    embedding_dict = {}
    for lang in langs:
        embedding_dict[lang] = []
    embedding_dict["other"] = []
    
    embedding_dim = 0
    try:
        with open(embedding_file) as fp:
            for line in fp:
                line_list = line.strip().split("\t")
                if len(langs) > 0:
                    item_langs = line_list[0].split(",")
                    item_vec = " ".join(line_list[1:])
                    vec_dim = len(line_list) - 2
                else:
                    item_langs = ["other"]
                    item_vec = " ".join(line_list[0:])
                    vec_dim = len(line_list) - 1

                if embedding_dim == 0:                  # 第一行决定维度
                    embedding_dim = vec_dim
                elif embedding_dim != vec_dim:          # 如果某一行的维度不正确，则pass
                    continue

                for lang in item_langs: 
                    if lang in langs:
                        embedding_dict[lang].append(item_vec)
                    else:
                        embedding_dict["other"].append(item_vec)


    except Exception as e:
        print("read_embedding_file error: %s" % str(e))
    return embedding_dict, embedding_dim


def save_embedding(item_embedding_dict, out_file, langs):
    for lang in langs + ["other"]:
        with open(out_file + "/" + lang + "_vectors.txt", "w") as fp:
            fp.write("\n".join(item_embedding_dict[lang]))


def main(embedding_file, out_file, debug_label_emb):
    # 读入embedding文件
    if len(debug_label_emb.split(",")) == 2:
        langs = ["mr", "ta", "kn", "te", "hi"]
        embedding_dict, embedding_dim = read_embedding_file(embedding_file, langs)
    else:
        langs = []
        embedding_dict, embedding_dim = read_embedding_file(embedding_file, langs)

    if not embedding_dict:
        return
    
    save_embedding(embedding_dict, out_file, langs)


if __name__ == "__main__":
    embedding_file     = sys.argv[1]
    out_file           = sys.argv[2]
    debug_label_emb    = sys.argv[3]
    main(embedding_file, out_file, debug_label_emb)
