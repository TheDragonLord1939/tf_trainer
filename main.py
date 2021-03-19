import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
# 设置日志显示级别，1表示所有，2表示只显示error和warning，3表示只显示error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
# 忽略一些warning
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings
warnings.filterwarnings('ignore')

from models import model


def main(_):
    mode = tf.flags.FLAGS.mode
    #print("is gpu avaliable:",tf.test.is_gpu_available())
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.test()
    elif mode == 'predict':
        model.predict()
    else:
        raise ValueError('--mode {} was not found.'.format(mode))


if __name__ == '__main__':
    tf.app.run(main)
