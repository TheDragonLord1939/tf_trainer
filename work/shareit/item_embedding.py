#!/usr/bin/python
#coding=utf-8
import os
import sys
import argparse

parser = argparse.ArgumentParser(description = "save item embedding")
parser.add_argument("--old_embedding", type=str, default=None, help='old item embedding file')
parser.add_argument("item_index_file", type=str, help='item index file')
parser.add_argument("embedding_file",  type=str, help='index embedding file')
parser.add_argument("out_file",        type=str, help='item embedding output file')
args = parser.parse_args()


old_embedding_file = args.old_embedding
item_index_file    = args.item_index_file
embedding_file     = args.embedding_file
out_file           = args.out_file


def read_embedding_file(embedding_file):
    embedding_dict = {}
    embedding_dim  = 0
    try:
        with open(embedding_file) as fp:
            for line in fp:
                line_list = line.strip().split(" ")
                item_id   = line_list[0]
                item_vec  = " ".join(line_list[1:])

                vec_dim = len(line_list) - 1
                if embedding_dim == 0:                  # 第一行决定维度
                    embedding_dim = vec_dim
                elif embedding_dim != vec_dim:          # 如果某一行的维度不正确，则pass
                    continue
                
                embedding_dict[item_id] = item_vec
    except Exception as e:
        print("read_embedding_file error: %s" % str(e))
    return embedding_dict, embedding_dim


def read_index_file(index_file):
    item_index_dict = {}
    try:
        with open(index_file) as fp:
            for line in fp:
                item_id, item_index = line.strip().split("\t")
                item_index_dict[item_id] = item_index
    except Exception as e:
        print("read_index_file error: %s" % str(e))
    return item_index_dict 


def combine_embedding(item_index_dict, index_embedding_dict, old_embedding_dict):
    item_embedding_dict = {}
    update_num = 0
    try:
        item_embedding_dict.update(old_embedding_dict)                  # 使用历史数据

        for (item_id, item_index) in item_index_dict.items():
            item_embedding = index_embedding_dict.get(item_index, None)
            if item_embedding:
                item_embedding_dict[item_id] = item_embedding
                update_num += 1
    except Exception as e:
        print("combine_embedding error: %d" % str(e))

    return item_embedding_dict, update_num


def save_embedding(item_embedding_dict, out_file):
    write_list = []

    for item_id, item_vec in item_embedding_dict.items():
        write_list.append("%s %s" % (item_id, item_vec))

    with open(out_file, "w") as fp:
        fp.write("\n".join(write_list))


def main():
    # 读入历史item的embedding
    old_embedding_dict = {}
    old_embedding_dim  = 0
    if old_embedding_file:
        old_embedding_dict, old_embedding_dim = read_embedding_file(old_embedding_file)
    print("history item num: %d embedding: %d" % (len(old_embedding_dict), old_embedding_dim))

    # 读入item_index文件
    item_index_dict = read_index_file(item_index_file)
    print("item num: %d" % len(item_index_dict))
    if not item_index_dict:
        return

    # 读入index的embedding 
    index_embedding_dict, embedding_dim = read_embedding_file(embedding_file)
    print("index num: %d embedding: %d" % (len(index_embedding_dict), embedding_dim))
    if not index_embedding_dict:
        return
    if (old_embedding_dim > 0) and (embedding_dim != old_embedding_dim):
        print("old embedding %d not match new embedding %d" % (old_embedding_dim, embedding_dim))
        return

    
    # 合并embedding并保存
    result_embedding_dict, update_num = combine_embedding(item_index_dict, index_embedding_dict, old_embedding_dict)
    print("result num: %d, update %d" % (len(result_embedding_dict), update_num))
    if not result_embedding_dict:
        return

    save_embedding(result_embedding_dict, out_file)

if __name__ == "__main__":
    main()
