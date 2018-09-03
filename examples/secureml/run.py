import sys
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from convert import decode


if len(sys.argv) >= 2:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.config.load(config_file)
else:
    # default to using local config
    config = tfe.LocalConfig([
        'server0',
        'server1',
        'crypto-producer',
        'model-trainer',
        'prediction-client'
    ])


class ModelTrainer(tfe.io.InputProvider):

    BATCH_SIZE = 30
    ITERATIONS = 60000//BATCH_SIZE
    EPOCHS = 10

    def build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def build_training_graph(self, training_data) -> List[tf.Tensor]:

        # model parameters and initial values
        w0 = tf.Variable(tf.random_normal([28*28, 128]))
        b0 = tf.Variable(tf.zeros([128]))
        w1 = tf.Variable(tf.random_normal([128, 128]))
        b1 = tf.Variable(tf.zeros([128]))
        w2 = tf.Variable(tf.random_normal([128, 10]))
        b2 = tf.Variable(tf.zeros([10]))

        # optimizer and data pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        # training loop
        def loop_body(i):

            # get next batch
            x, y = training_data.get_next()

            # model construction
            layer0 = x
            layer1 = tf.nn.sigmoid(tf.matmul(layer0, w0) + b0)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, w1) + b1)
            layer3 = tf.matmul(layer2, w2) + b2
            predictions = layer3

            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))    
            with tf.control_dependencies([optimizer.minimize(loss)]):
                return i + 1

        loop = tf.while_loop(lambda i: i < self.ITERATIONS * self.EPOCHS, loop_body, (0,))

        # return model parameters after training
        loop = tf.Print(loop, [], message="Training complete")
        with tf.control_dependencies([loop]):
            return [
                var.read_value() 
                for var in [w0, b0, w1, b1, w2, b2]
            ]

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            training_data = self.build_data_pipeline()

        with tf.name_scope('training'):
            parameters = self.build_training_graph(training_data)

        return parameters


class PredictionClient(tfe.io.InputProvider, tfe.io.OutputReceiver):

    BATCH_SIZE = 20

    def build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/test.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            prediction_input, expected_result = self.build_data_pipeline().get_next()
            prediction_input = tf.Print(prediction_input, [expected_result], summarize=self.BATCH_SIZE, message="EXPECT ")

        with tf.name_scope('pre-processing'):
            prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28*28))

        return [prediction_input]

    def receive_output(self, tensors: List[tf.Tensor]) -> tf.Operation:
        likelihoods, = tensors
        with tf.name_scope('post-processing'):
            prediction = tf.argmax(likelihoods, axis=1)
            op = tf.Print([], [prediction], summarize=self.BATCH_SIZE, message="ACTUAL ")
            return op


model_trainer = ModelTrainer(config.get_player('model-trainer'))
prediction_client = PredictionClient(config.get_player('prediction-client'))

server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto-producer')

with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

    # get model parameters as private tensors from model owner
    params = prot.define_private_input(model_trainer, masked=True) # pylint: disable=E0632
    
    # we'll use the same parameters for each prediction so we cache them to avoid re-training each time
    params = prot.cache(params)

    # get prediction input from client
    x, = prot.define_private_input(prediction_client, masked=True) # pylint: disable=E0632

    # compute prediction
    w0, b0, w1, b1, w2, b2 = params
    layer0 = x
    layer1 = prot.sigmoid( (prot.dot(layer0, w0) + b0) ) # input normalized to avoid large values
    layer2 = prot.sigmoid( (prot.dot(layer1, w1) + b1) ) # input normalized to avoid large values
    layer3 = prot.dot(layer2, w2) + b2
    prediction = layer3

    # send prediction output back to client
    prediction_op = prot.define_output([prediction], prediction_client)


with config.session() as sess:
    print("Init")
    tfe.run(sess, tf.global_variables_initializer(), tag='init')
    
    print("Training")
    tfe.run(sess, tfe.global_caches_updator(), tag='training')

    for _ in range(5):
        print("Predicting")
        tfe.run(sess, prediction_op, tag='prediction')
