import mxnet as mx


def model_fn():
    sym, arg_params, aux_params = mx.model.load_checkpoint('../src/mxnet_container/customer/model/mnist', 10)
    model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 28, 28))],
               label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model
