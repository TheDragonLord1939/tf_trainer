#!/usr/bin/python
#coding=utf-8
import os
import sys
import time
import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("model_dir", None, "model directory")               # 模型保存目录
flags.DEFINE_string("checkpoint_dir", None, "checkpoint directory")     # 检查点保存目录
flags.DEFINE_string("variable", None, "variable name")                  # 待导出的变量名称
flags.mark_flag_as_required("variable")
flags.DEFINE_string("out_file", None, "out file")                       # embedding保存目录，必须指定
flags.mark_flag_as_required("out_file")
FLAGS = flags.FLAGS


def get_variable_embedding_from_model(model_dir, variable_name):
    export_name = "%s:0" % variable_name
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], model_dir)
        variables = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        variable_name_list = [variable.name for variable in variables]
        if export_name not in variable_name_list:
            print("%s not in trainable variables: %r" % (export_name, variable_name_list))
            return np.array([])

        for variable in variables:
            if variable.name == export_name:
                return sess.run(variable.value())                       # 返回的是ndarray


def get_variable_embedding_from_checkpoint(checkpoint_dir, variable_name):
    with tf.Session() as sess:
        # 从目录中读取ckp，如果有多个，则取最新
        reader    = tf.train.load_checkpoint(checkpoint_dir)
        variables = reader.get_variable_to_shape_map()
        if variable_name not in variables:
            print("%s not in trainable variables: %r" % (variable_name, variable_name_list))
            return np.array([])
        return reader.get_tensor(variable_name)
        

def save_embedding(embedding_matrix, out_file):
    write_list = []

    for index in range(embedding_matrix.shape[0]):
        vector     = embedding_matrix[index]
        vector_str = [str(each) for each in vector]
        write_list.append("%d %s" % (index, " ".join(vector_str)))

    with open(out_file, "w") as fp:
        fp.write("\n".join(write_list))


def main(argv):
    model_dir     = FLAGS.model_dir
    ckp_dir       = FLAGS.checkpoint_dir
    variable_name = FLAGS.variable
    out_file      = FLAGS.out_file

    # 获取变量的embedding
    if model_dir:
        embedding_matrix = get_variable_embedding_from_model(model_dir, variable_name)
    elif ckp_dir:
        embedding_matrix = get_variable_embedding_from_checkpoint(ckp_dir, variable_name)
    else:
        print("model_dir or checkpoint_dir needed!")
        return

    if embedding_matrix.size == 0:
        print("export %s fail!" % variable_name)
        return

    # 保存
    save_embedding(embedding_matrix, out_file)


if __name__ == "__main__":
    tf.app.run()

