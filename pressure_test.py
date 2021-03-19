import argparse
import grpc
import tensorflow as tf
import time
import threading
from grpc.framework.interfaces.face.face import CancellationError, ExpirationError
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from dataset.rec_dataset import RecDataset


class _ResultCounter(object):
    '''Counter for the prediction results.'''

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._cancel = 0
        self._expire = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._mutex = threading.Lock()
        self._timecost = []

    def inc_cancel(self):
        self._mutex.acquire()
        try:
            self._cancel += 1
        finally:
            self._mutex.release()

    def inc_expire(self):
        self._mutex.acquire()
        try:
            self._expire += 1
        finally:
            self._mutex.release()

    def finish(self, tc):
        with self._condition:
            self._done += 1
            self._active -= 1
            self._timecost.append(tc)
            self._condition.notifyAll()

    def get_timecost_info(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            timecost_info = {}
            self._timecost = [tc*1000 for tc in self._timecost]
            self._timecost.sort()
            timecost_info['Max'] = self._timecost[len(self._timecost)-1]
            timecost_info['The99%'] = self._timecost[int(
                round(len(self._timecost)*.99))-1]
            timecost_info['The90%'] = self._timecost[int(
                round(len(self._timecost)*.9))-1]
            timecost_info['Average'] = sum(self._timecost)/len(self._timecost)
            timecost_info['Min'] = self._timecost[0]
            return timecost_info

    def get_cancel_rate(self):
        with self._condition:
            # Must check before main function exit
            while self._done != self._num_tests:
                self._condition.wait()
            return self._cancel / float(self._num_tests)

    def get_expire_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._expire / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(result_counter, start_time, show_result):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            if isinstance(exception, CancellationError):
                result_counter.inc_cancel()
            elif isinstance(exception, ExpirationError):
                result_counter.inc_expire()
            else:
                print(exception)
        elif show_result:
            for key, result in result_future.result().outputs.items():
                print('result is '+key)
                print(result.float_val)
        end_time = time.time()
        result_counter.finish(end_time-start_time)
    return _callback


def load_data(sess, num_tests, schema, dataset, batch_size, prebatch):
    dataset = RecDataset('label', schema, dataset, dataset, batch_size=batch_size, prebatch=prebatch)
    valid = dataset.eval_set()
    valid = valid.map(de_prebatch(dataset, prebatch))
    data = valid.make_one_shot_iterator()
    samples = list()
    try:
        for _ in range(num_tests):
            tensors = data.get_next()
            samples.append(sess.run(tensors))
        return samples
    except tf.errors.OutOfRangeError:
        return samples


def do_inference(hostport, model_name, num_tests, data, concurrency, show_results):
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    data_len = len(data)
    request_list = []
    for i in range(data_len):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        features = data[i]
        for key, feature in features.items():
            request.inputs[key].CopyFrom(tf.make_tensor_proto(feature, shape=feature.shape))
        request_list.append(request)
    tstart = time.time()
    for i in range(num_tests):
        result_counter.throttle()
        result_future = stub.Predict.future(request_list[i % data_len], 2)
        result_future.add_done_callback(
            _create_rpc_callback(result_counter, time.time(), show_results))
    expire_rate = result_counter.get_expire_rate()
    tend = time.time()
    print('Total predict time cost: {:.4f}s, QPS: {:.2f}'.format(tend-tstart, num_tests/(tend-tstart)))
    return expire_rate, result_counter.get_timecost_info()


def de_prebatch(dataset, prebatch):
    def mapper(features, _):
        for name in dataset.numerical_list + dataset.categorical_list:
            features[name] = _de_prebatch_dense(features[name], 1)
        for name in dataset.vector_list:
            features[name] = _de_prebatch_dense(features[name], dataset.vec_dim[name])
        for name in dataset.varlen_list:
            features[name] = _de_prebatch_sparse(features[name], dataset.voc_size[name], prebatch)
        features = _sparse2dense(features, dataset.varlen_list)
        return features
    return mapper


def _de_prebatch_dense(feature, dim):
    return tf.reshape(feature, [-1, dim])


def _de_prebatch_sparse(feature, dim, prebatch):
    new_indices = feature.values // dim + feature.indices[:, 0] * prebatch
    new_feature = tf.SparseTensor(
        indices=tf.stack([new_indices, feature.values % dim], axis=1),
        values=feature.values % dim,
        dense_shape=[-1, dim])
    return new_feature


def _sparse2dense(features, sparse_list):
    _features = dict()
    for name, tensor in features.items():
        if name in sparse_list:
            _features[name+'_indices'] = tensor.indices
            _features[name+'_values'] = tensor.values
            _features[name+'_denseShape'] = tensor.dense_shape
        else:
            _features[name] = tensor
    return _features


def main(args):
    with tf.Session() as sess:
        data = load_data(sess, args.num_tests, args.schema, args.dataset, args.batch_size, args.prebatch)
        expire_rate, timecost_info = do_inference(args.host, args.model_name,
                                                  args.num_tests, data, args.concurrency,
                                                  args.show_results)
    print('\nInference ExpirationError rate: {}'.format(expire_rate * 100))
    for key, value in sorted(timecost_info.items()):
        print('{}: {:.3f}'.format(key, value), end='\t')
    print('', end='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost:8500')
    parser.add_argument('-s', '--schema', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-n', '--model_name', type=str, default='test')
    parser.add_argument('--num_tests', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--prebatch', type=int, default=512)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('-r', '--show_results', type=bool, default=False)
    args = parser.parse_args()
    main(args)
