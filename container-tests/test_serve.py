import json
import mxnet as mx
import numpy as np
import random

from mxnet_container.serve.transformer import ModuleTransformer


def get_graph(hidden=128):
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=hidden)
    graph = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    return graph


def get_training_data(num_rows, num_cols):
    train_data = [
        [random.uniform(-1, 1) for _ in range(num_cols)]
        for _ in range(num_rows)
    ]
    return train_data


def test_prepare_input_for_default_predict():
    model = mx.mod.Module(get_graph(), context=mx.cpu())

    # Make some training data
    train_data = get_training_data(500, 100)
    train_array = np.array(train_data).reshape((500, 1, 10, 10))
    labels = np.ones([500])
    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=5)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)
    iter_for_predict = ModuleTransformer._prepare_input_for_default_predict(model, train_array)

    # Assert that the batch size is read from the model correctly
    assert iter_for_predict.batch_size == train_iter.batch_size

    # Assert that the shape reflects the model batch size.
    assert iter_for_predict.provide_data[0].shape == (5, 1, 10, 10)


def test_prepare_input_for_default_predict_large_dimension():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(128, 2)

    # Reshape to 8 dimensions of size 2
    train_array = np.array(train_data)
    train_array = train_array.reshape(tuple([2 for _ in range(8)]))
    labels = np.ones([256])

    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)
    iter_for_predict = ModuleTransformer._prepare_input_for_default_predict(model, train_array)

    assert iter_for_predict.provide_data[0].shape == train_array.shape


def test_prepare_input_for_default_predict_single_dimension():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(1, 2)
    train_array = np.array(train_data)
    train_array = train_array.reshape((2))
    labels = np.ones([2])

    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)
    iter_for_predict = ModuleTransformer._prepare_input_for_default_predict(model, train_array)

    assert iter_for_predict.provide_data[0].shape == train_array.shape


def test_batch_size_truncating():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(100, 10)
    train_array = np.array(train_data)
    train_array = train_array.reshape((10, 10, 10))
    labels = np.ones([1000])

    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)
    iter_for_predict = ModuleTransformer._prepare_input_for_default_predict(model, train_array)

    assert iter_for_predict.provide_data[0].shape == (2, 10, 10)


def test_padding():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(100, 10)
    train_array = np.array(train_data)
    train_array = train_array.reshape((10, 10, 10))
    labels = np.ones([1000])

    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=10)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)

    # Take half the data, but keep batch size equal to ten. This will force padding
    # inside prepare_input_for_default_predict
    predict_array = mx.nd.array(train_array[5:])
    iter_for_predict = ModuleTransformer._prepare_input_for_default_predict(model, predict_array)
    padded_array = iter_for_predict.next().data[0]

    assert np.array_equal(padded_array[5:].asnumpy(), np.zeros((5, 10, 10)))
    assert iter_for_predict.getpad() == 5


def test_process_json_input():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(128, 2)

    # Reshape to 8 dimensions of size 2 and 8 dimensions of size 1
    train_array = np.array(train_data)
    train_array = train_array.reshape(tuple([2 for _ in range(8)] + [1 for _ in range(8)]))
    labels = np.ones([256])
    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)

    model.fit(train_iter, optimizer='sgd', num_epoch=1)

    # Create a JSON array with 8 dimensions of size 2, 8 dimensions of size 1
    json_array = json.dumps(train_array.tolist())
    iter_for_predict = ModuleTransformer._process_json_input(model, json_array)

    assert iter_for_predict.provide_data[0].shape == train_array.shape
    assert iter_for_predict.getpad() == 0


def test_process_csv_input():
    model = mx.mod.Module(get_graph(), context=mx.cpu())
    train_data = get_training_data(8, 2)
    labels = np.ones([8])
    train_array = np.array(train_data)
    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)

    model.fit(train_iter, optimizer='sgd', num_epoch=1)

    csv_data = "51.2810738,0.7694025\n47.6131742,-122.4824935"
    iter_for_predict = ModuleTransformer._process_csv_input(model, csv_data)

    assert iter_for_predict.provide_data[0].shape == (2, 2)
    assert iter_for_predict.getpad() == 0


def test_default_output():
    model = mx.mod.Module(get_graph(hidden=2), context=mx.cpu())

    # Make some training data, fit a model
    train_data = get_training_data(4, 2)
    train_array = np.array(train_data).reshape((4, 2))
    labels = np.ones([4])
    train_iter = mx.io.NDArrayIter(mx.nd.array(train_array), label=labels, data_name='data', batch_size=2)
    model.fit(train_iter, optimizer='sgd', num_epoch=1)

    # Use the training data to predict
    prediction = ModuleTransformer._default_predict_fn(model, train_iter)
    assert prediction.shape == (4, 2)

    # Test json formatting
    json_prediction, _ = ModuleTransformer._default_output_fn(prediction, "application/json")
    assert prediction.asnumpy().tolist() == json.loads(json_prediction)

    # Test CSV formatting
    # CSV formatting can cause loss of precision, so do an approximate 
    # comparison 
    csv_prediction, _ = ModuleTransformer._default_output_fn(prediction, "text/csv")
    csv_prediction = np.array([[float(x) for x in row.split(",")] for row in csv_prediction.split()])
    diff = csv_prediction - prediction.asnumpy()
    assert np.sum(diff) < 0.1
