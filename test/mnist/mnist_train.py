from __future__ import print_function
import os
import json
import mxnet as mx


def train(mnist):
    print("training...")
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)
    # The first fully-connected layer and the corresponding activation function
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")

    # The second fully-connected layer and the corresponding activation function
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")

    # MNIST has 10 classes
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    # create a trainable module on CPU
    model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='adam',  # use Adam to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback=mx.callback.do_checkpoint("data/model/mnist", 1),
              batch_end_callback=mx.callback.Speedometer(batch_size, 10),  # output progress for each 10 data batches
              num_epoch=10)  # train for at most 10 dataset passes

    return model


def extract_images(mnist):
    # save the first 10 test images as json
    print("extracting test images...")
    for i in range(10):
        with open("data/images/%02d.json" % i, 'w') as f:
            image = mnist['test_data'][i].tolist()
            json.dump(image, f)


if __name__ == '__main__':
    if not (os.path.exists('data')):
        print("building mnist model...")
        for d in ['data/model', 'data/images']:
            if not os.path.exists(d):
                os.makedirs(d)

        print("downloading dataset...")
        mnist = mx.test_utils.get_mnist()
        train(mnist)
        extract_images(mnist)

        os.remove("t10k-images-idx3-ubyte.gz")
        os.remove("t10k-labels-idx1-ubyte.gz")
        os.remove("train-images-idx3-ubyte.gz")
        os.remove("train-labels-idx1-ubyte.gz")
