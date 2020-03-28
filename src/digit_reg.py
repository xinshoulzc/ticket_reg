import tensorflow as tf
import cv2 as cv
import utils
import argparse
import sys

X_SIZE, Y_SIZE = 28, 28
TRAIN_RATE = 0.9

images = tf.placeholder(tf.float32, shape=(None, X_SIZE, Y_SIZE))
labels = tf.placeholder(tf.int32, shape=(None, ))

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

def train():
    logits = model_fn(images)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=None)
    loss = tf.reduce_mean(entropy)
    tf.summary.scalar("loss", loss)
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)
    return train_op, loss

def infer():
    logits = model_fn(images)
    # print(logits.shape)
    predict = tf.math.argmax(logits, axis=1)
    # print(predict.shape)
    return predict

def train_main(inputdir, modeldir):
    paths, x_train, y_train = utils.load_data_once(inputdir)
    train_size = int(len(paths) * TRAIN_RATE)
    paths, x_train, y_train = paths[:train_size], x_train[:train_size], y_train[:train_size]

    # hyperparameters
    epochs = 20 # epochs round on all dataset
    batch_size = 200

    # train process
    train_ops = train()
    saver = tf.train.Saver(max_to_keep=2)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        # summary_merge = tf.summary.merge_all()
        # f_summary = tf.summary.FileWriter(logdir="log", graph=sess.graph)
        for epoch in range(epochs):
            print(epoch, x_train.shape[0])
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
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
                    print("epoch: %d, batch id: %d, loss: %f" %(epoch, batch_id, train_loss))

        saver.save(sess, os.path.join(modeldir, "mnist.cpkt"))

        print("train finished...")

def infer_main(inputdir, modeldir, outputdir):
    # load data
    if len(outputdir) <= 0:
        paths, x_test, y_test = utils.load_data_once(inputdir)
        train_size = int(len(paths) * TRAIN_RATE)
        paths, x_test, y_test = paths[train_size:], x_test[train_size:], y_test[train_size:]
        print(x_test.shape, y_test.shape, " mode: eval")
    else:
        paths, x_test, _ = utils.load_infer_data(inputdir)
        print(x_test.shape, " mode: infer")

    # inference process
    predict = infer()
    saver = tf.train.Saver(max_to_keep=2)
    # saver = tf.train.import_meta_graph('model/mnist.cpkt-4.meta')
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(modeldir, "mnist.cpkt"))
        y_predict = sess.run(predict, feed_dict={
            images: x_test,
            # labels: y_test,
        })
    if len(outputdir) <= 0:
        print("y_predict", y_predict)
        print("true label", y_test)

        cnt, correct = 0, 0
        cnt_, co_ = 0, 0
        for k1, k2 in zip(y_predict, y_test):
            # print(k1, k2)
            if k2 != 0: cnt_ += 1
            cnt += 1
            if k1 == k2:
                correct += 1
                if k2 != 0: co_ += 1
            else:
                continue
                # print(p, k1, k2)

        print("all data", cnt, correct, correct / cnt)
        print("not zero data", cnt_, co_, co_/cnt_)
    else:
        with open(outputdir, "w") as out:
            for paths, label in zip(imgs_paths, y_predict):
                out.write(path + "\t" + string(label) + "\n")



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--inputdir", required=True)
        parser.add_argument("-o", "--outputdir", required=True)
        parser.add_argument("-o", "--modeldir", required=True)
        parser.add_argument("-m", "--mode", required=True)
        args = parser.parse_args()
        print("intputdir: ", args.inputdir, "outputdir: ", args.outputdir, "modeldir: ", args.modeldir, "mode: ", args.mode)
        if args.mode == "train":
            train_main(args.inputdir, args.modeldir)
        elif args.mode == "eval":
            infer_main(args.inputdir, args.modeldir, args.outputdir)
        else:
            print("UNKNOWN MODE")
            sys.exit(-1)
    except:
        sys.exit(-1)
