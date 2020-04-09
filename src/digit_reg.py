import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import utils
import argparse
import traceback
import sys
from logger import logger
from utils import TRAIN_MODE

X_SIZE, Y_SIZE = 28, 28
TRAIN_RATE = 0.9
batch_size = 50

images = tf.placeholder(tf.float32, shape=(None, X_SIZE, Y_SIZE))
labels = tf.placeholder(tf.int32, shape=(None, ))
tf.random.set_random_seed(2333)
debug_op = []

def model_fn(x):
    # x shape [batch size, x_size, y_size]
    x = tf.reshape(x, (-1, X_SIZE * Y_SIZE))
    # x shape [batch size, x_size * y_size]
    FC1 = tf.keras.layers.Dense(units=128, activation='relu')
    x = FC1(x)
    # x shape [batch size, 128]
    x = tf.nn.dropout(x, rate=0.2)
    # x shape [batch size, 128]
    FC2 = tf.keras.layers.Dense(units=10, activation=None)
    logits = FC2(x)
    # y shape [batch size, 10]
    return logits

def train(aug=False):
    # data augument
    if aug:
        imgs = tf.image.random_crop(images, [batch_size, int(X_SIZE * 0.9), int(Y_SIZE * 0.9)])
        imgs = tf.expand_dims(imgs, -1)
        imgs = tf.image.resize_image_with_pad(imgs, X_SIZE, Y_SIZE)
        imgs = tf.squeeze(imgs)
    else:
        imgs = images

    logits = model_fn(imgs)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=None)
    loss = tf.reduce_mean(entropy)
    tf.summary.scalar("loss", loss)
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)
    return train_op, loss

def infer():
    logits = model_fn(images)
    debug_op.append(logits)
    # print(logits.shape)
    predict = tf.math.argmax(logits, axis=1)
    # print(predict.shape)
    return predict

def train_main(inputdir, modeldir, train_mode):
    if train_mode == TRAIN_MODE.PRICE:
        paths, x_train, y_train = utils.load_train_data(["datasets/train"], [TRAIN_MODE.PRICE])
        model_name = "mnist.cpkt"
    elif train_mode == TRAIN_MODE.PRINT_DATASET:
        # TODO: data split, only print dataset are split
        paths, x_train, y_train = utils.load_train_data(
                                    ["datasets/train", "datasets/english"], [TRAIN_MODE.PRICE, TRAIN_MODE.PRINT_DATASET])
        model_name = "barcode.ckpt"
    elif train_mode == TRAIN_MODE.BARCODE:
        paths, x_train, y_train = utils.load_train_data(["datasets/print_figures"], [TRAIN_MODE.BARCODE])
        model_name = "barcode.ckpt"
    else:
        logger.error("UNKNOWN train mode")

    train_size = int(len(paths) * TRAIN_RATE)
    paths, x_train, y_train = paths[:train_size], x_train[:train_size], y_train[:train_size]

    # hyperparameters
    epochs = 500 # epochs round on all dataset

    # train process
    if train_mode == TRAIN_MODE.PRINT_DATASET:
        train_ops = train(aug=False)
    elif train_mode == TRAIN_MODE.BARCODE:
        train_ops = train(aug=False)
    else:
        train_ops = train(aug=False)
    saver = tf.train.Saver(max_to_keep=2)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        # summary_merge = tf.summary.merge_all()
        # f_summary = tf.summary.FileWriter(logdir="log", graph=sess.graph)
        for epoch in range(epochs):
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                if len(x_batch) < batch_size: continue
                _, train_loss = sess.run([train_ops[0], train_ops[1]], feed_dict={
                    images: x_batch,
                    labels: y_batch,
                })
                # summary_tmp, _, train_loss = sess.run([summary_merge, train_ops[0], train_ops[1]], feed_dict={
                #     images: x_train,
                #     labels: y_train,
                # })
                # f_summary.add_summary(summary=summary_tmp, global_step=epoch)
                batch_id = epoch * (x_train.shape[0] // batch_size * batch_size) + i
                if batch_id % 4000 == 0:
                    logger.info("epoch: %d, batch id: %d, loss: %f" %(epoch, batch_id, train_loss))

        saver.save(sess, os.path.join(modeldir, model_name))

        logger.info("train finished...")

def infer_main(inputdir, modeldir, outputdir, eval_mode):
    # select model
    if eval_mode == "price":
        train_mode = TRAIN_MODE.PRICE
        model_name = "mnist.cpkt"
    elif eval_mode == "barcode":
        train_mode = TRAIN_MODE.BARCODE
        model_name = "barcode.ckpt"
    else:
        raise ValueError("unk eval mode")

    # eval train data or infer data
    if outputdir is None or len(outputdir) <= 0:
        paths, x_test, y_test = utils.load_train_data([inputdir], [train_mode])
        # train_size = int(len(paths) * TRAIN_RATE)
        # paths, x_test, y_test = paths[train_size:], x_test[train_size:], y_test[train_size:]
        logger.info("x_test shape: {}".format(str(x_test.shape)))
        logger.info("y_test shape: {}".format(str(y_test.shape)))
    else:
        paths, x_test, _ = utils.load_infer_data(inputdir)
        logger.info("x_test shape: {}".format(str(x_test.shape)))

    # inference process
    predict = infer()
    saver = tf.train.Saver(max_to_keep=2)
    # saver = tf.train.import_meta_graph('model/mnist.cpkt-4.meta')
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(modeldir, model_name))
        y_predict, debug_op1 = sess.run([predict, debug_op[0]], feed_dict={
            images: x_test,
            # labels: y_test,
        })
    print("logits", debug_op1)
    if outputdir is None or len(outputdir) <= 0:
        logger.debug("y_predict, length: {:d}".format(len(y_predict)))
        logger.debug("true label, length: {:d}".format(len(y_test)))

        cnt, correct = 0, 0
        cnt_, co_ = 0, 0
        ind = -1
        for k1, k2 in zip(y_predict, y_test):
            ind += 1
            # print(k1, k2)
            if k2 != 0: cnt_ += 1
            cnt += 1
            if k1 == k2:
                correct += 1
                if k2 != 0: co_ += 1
            else:
                logger.info("fn: {}, predict: {:d}, true: {:d}".format(paths[ind], k1, k2))
                continue
                # print(p, k1, k2)
        logger.info("all data, length: {:d}, correct: {:d}, precision: {:f}".format(
            cnt, correct, correct / cnt))
        logger.info("all data except label 0, length: {:d}, corrent: {:d}, presicion: {:f}".format(
            cnt_, co_, co_ / cnt_))
    else:
        logger.debug("infer image length: {:d}, predict label length: {:d}".format(
            len(paths), len(y_predict)))
        with open(os.path.join(outputdir, "results"), "w") as out:
            for x, y in zip(paths, y_predict):
                out.write(x + "\t" + str(y) + "\n")



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--inputdir", required=True)
        parser.add_argument("-o", "--outputdir")
        parser.add_argument("-d", "--modeldir", required=True)
        parser.add_argument("-m", "--mode", required=True)
        args = parser.parse_args()
        logger.info("intputdir: {}".format(args.inputdir))
        logger.info("outputdir: {}".format(args.outputdir))
        logger.info("modeldir: {}".format(args.modeldir))
        logger.info("mode: {}".format(args.mode))
        if args.mode == "train":
            # train_main("datasets/train", args.modeldir, TRAIN_MODE.PRICE)
            train_main("datasets/print_figures", args.modeldir, TRAIN_MODE.BARCODE)
        elif args.mode == "price":
            infer_main(args.inputdir, args.modeldir, args.outputdir, args.mode)
        elif args.mode == "barcode":
            infer_main(args.inputdir, args.modeldir, args.outputdir, args.mode)
        else:
            logger.error("UNKNOWN MODE: {}".format(args.mode))
            sys.exit(-1)
    except:
        traceback.print_exc()
        sys.exit(-1)
