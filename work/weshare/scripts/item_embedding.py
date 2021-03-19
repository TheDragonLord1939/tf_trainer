#!/usr/bin/python
# coding=utf-8
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser(description="save item embedding")
parser.add_argument("item_index_file", type=str, help='item index file')
parser.add_argument("embedding_file", type=str, help='index embedding file')
parser.add_argument("out_path", type=str, help='item embedding output path')
args = parser.parse_args()

item_index_file = args.item_index_file
embedding_file = args.embedding_file
out_file = os.path.join(args.out_path, '{}_vectors.txt')
langs = ['hi', 'te', 'kn', 'ta']


def read_embedding_file(embedding_file):
    embedding_dict = dict()
    langs_dict = dict()
    embedding_dim = 0
    try:
        with open(embedding_file) as fp:
            for line in fp:
                line_list = line.strip().split("\t")
                item_id = line_list[0]
                item_langs = line_list[2].split(',')
                item_vec = line_list[2:]

                vec_dim = len(line_list) - 1
                if embedding_dim == 0:  # 第一行决定维度
                    embedding_dim = vec_dim
                elif embedding_dim != vec_dim:  # 如果某一行的维度不正确，则pass
                    continue
                langs_dict[item_id] = item_langs
                embedding_dict[item_id] = item_vec
    except Exception as e:
        print("read_embedding_file error: %s" % str(e))
        embedding_dict = embedding_validation(embedding_dict)
    return embedding_dict, langs_dict, embedding_dim


def embedding_validation(embedding_dict):
    epsilon = 1e-15
    valid_dict = dict()
    for item_id, embedding in embedding_dict.items():
        valid = [float(i) > epsilon for i in embedding]
        if any(valid):
            valid_dict[item_id] = ''.join(embedding)
    return valid_dict


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


def parse_embeddings(item_index_dict, index_embedding_dict, langs_dict):
    item_embedding_dict = defaultdict(dict)
    for (item_id, item_index) in item_index_dict.items():
        item_embedding = index_embedding_dict.get(item_index, None)
        if not item_embedding:
            continue
        item_langs = langs_dict.get(item_index, None)
        item_langs = [i for i in item_langs if i in langs]
        if item_langs:
            for lang in item_langs:
                item_embedding_dict[lang].update({item_id: item_embedding})
        else:
            item_embedding_dict['other'].update({item_id: item_embedding})

    return item_embedding_dict


def save_embedding(item_embedding_dict, out_file):
    write_list = ["{} {}".format(item_id, item_vec) for item_id, item_vec in item_embedding_dict.items()]
    with open(out_file, "w") as fp:
        fp.write("\n".join(write_list))


def main():
    # 读入item_index文件
    item_index_dict = read_index_file(item_index_file)
    print("item num: %d" % len(item_index_dict))
    if not item_index_dict:
        return

    # 读入index的embedding 
    index_embedding_dict, langs_dict, embedding_dim = read_embedding_file(embedding_file)
    print("index num: {} embedding: {} invalid num: {}".format(len(index_embedding_dict), embedding_dim, len(langs_dict)-len(index_embedding_dict)))
    if not index_embedding_dict:
        return
    # 合并embedding并保存
    results = parse_embeddings(item_index_dict, index_embedding_dict, langs_dict)
    for lang, result in results.items():
        save_embedding(result, out_file.format(lang))
        print('write lang: {} num: {}'.format(lang, len(result)))


if __name__ == "__main__":
    main()
