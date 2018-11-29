# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import os

from mock import call, Mock, mock_open, patch
import mxnet as mx
import pytest
from sagemaker_containers.beta.framework import content_types, errors, transformer, worker

from sagemaker_mxnet_container.serving import (_user_module_transformer, default_model_fn,
                                               GluonBlockTransformer, ModuleTransformer,
                                               MXNetTransformer)

MODEL_DIR = 'foo/model'


@patch('mxnet.cpu')
@patch('mxnet.mod.Module')
@patch('mxnet.model.load_checkpoint')
@patch('os.path.exists', return_value=True)
def test_default_model_fn(path_exists, mx_load_checkpoint, mx_module, mx_cpu):
    sym = Mock()
    args = Mock()
    aux = Mock()
    mx_load_checkpoint.return_value = [sym, args, aux]

    mx_context = Mock()
    mx_cpu.return_value = mx_context

    data_name = 'foo'
    data_shape = [1]
    signature = json.dumps([{'name': data_name, 'shape': data_shape}])

    with patch('six.moves.builtins.open', mock_open(read_data=signature)):
        default_model_fn(MODEL_DIR)

    mx_load_checkpoint.assert_called_with(os.path.join(MODEL_DIR, 'model'), 0)

    init_call = call(symbol=sym, context=mx_context, data_names=[data_name], label_names=None)
    assert init_call in mx_module.mock_calls

    model = mx_module.return_value
    model.bind.assert_called_with(for_training=False, data_shapes=[(data_name, data_shape)])
    model.set_params.assert_called_with(args, aux, allow_missing=True)


@patch('mxnet.eia', create=True)
@patch('mxnet.mod.Module')
@patch('mxnet.model.load_checkpoint')
@patch('os.path.exists', return_value=True)
@patch.dict(os.environ, {'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_default_model_accelerator_fn(path_exists, mx_load_checkpoint, mx_module, mx_eia):
    sym = Mock()
    args = Mock()
    aux = Mock()
    mx_load_checkpoint.return_value = [sym, args, aux]

    eia_context = Mock()
    mx_eia.return_value = eia_context

    data_name = 'foo'
    data_shape = [1]
    signature = json.dumps([{'name': data_name, 'shape': data_shape}])

    with patch('six.moves.builtins.open', mock_open(read_data=signature)):
        default_model_fn(MODEL_DIR)

    mx_load_checkpoint.assert_called_with(os.path.join(MODEL_DIR, 'model'), 0)

    init_call = call(symbol=sym, context=eia_context, data_names=[data_name], label_names=None)
    assert init_call in mx_module.mock_calls

    model = mx_module.return_value
    model.bind.assert_called_with(for_training=False, data_shapes=[(data_name, data_shape)])
    model.set_params.assert_called_with(args, aux, allow_missing=True)


@patch('sagemaker_containers.beta.framework.functions.error_wrapper', lambda x, y: x)
def test_mxnet_transformer_init():
    t = MXNetTransformer()

    assert t._model is None
    assert t._model_fn == transformer.default_model_fn
    assert t._input_fn == t.default_input_fn
    assert t._predict_fn == t.default_predict_fn
    assert t._output_fn == t.default_output_fn
    assert t.VALID_CONTENT_TYPES == (content_types.JSON,)


@patch('sagemaker_containers.beta.framework.functions.error_wrapper', lambda x, y: x)
def test_mxnet_transformer_init_with_args():
    model = Mock()
    model_fn = Mock()
    input_fn = Mock()
    predict_fn = Mock()
    output_fn = Mock()
    error_class = Mock()

    t = MXNetTransformer(model=model, model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn,
                         output_fn=output_fn, error_class=error_class)

    assert t._model == model
    assert t._model_fn == model_fn
    assert t._input_fn == input_fn
    assert t._predict_fn == predict_fn
    assert t._output_fn == output_fn
    assert t._error_class == error_class


@patch('sagemaker_containers.beta.framework.transformer.Transformer.initialize')
def test_mxnet_transformer_initialize_without_model(transformer_initialize):
    t = MXNetTransformer()
    t.initialize()

    transformer_initialize.assert_called_once()


@patch('sagemaker_containers.beta.framework.transformer.Transformer.initialize')
def test_mxnet_transformer_initialize_with_model(transformer_initialize):
    t = MXNetTransformer(model=Mock())
    t.initialize()

    transformer_initialize.assert_not_called()


@patch('sagemaker_containers.beta.framework.encoders.decode', return_value=[0])
def test_mxnet_transformer_default_input_fn(decode):
    input_data = Mock()
    content_type = 'application/json'

    t = MXNetTransformer()
    deserialized_data = t.default_input_fn(input_data, content_type)

    decode.assert_called_with(input_data, content_type)
    assert deserialized_data == mx.nd.array([0])


def test_mxnet_transformer_default_input_fn_invalid_content_type():
    t = MXNetTransformer()

    with pytest.raises(errors.UnsupportedFormatError) as e:
        t.default_input_fn(None, 'bad/content-type')
    assert 'Content type bad/content-type is not supported by this framework' in str(e)


@patch('sagemaker_containers.beta.framework.encoders.encode')
def test_mxnet_transformer_default_output_fn(encode):
    prediction = mx.ndarray.zeros(1)
    accept = 'application/json'

    t = MXNetTransformer()
    response = t.default_output_fn(prediction, accept)

    flattened_prediction = prediction.asnumpy().tolist()
    encode.assert_called_with(flattened_prediction, accept)

    assert isinstance(response, worker.Response)


def test_mxnet_transformer_default_output_fn_invalid_content_type():
    t = MXNetTransformer()

    with pytest.raises(errors.UnsupportedFormatError) as e:
        t.default_output_fn(None, 'bad/content-type')
    assert 'Content type bad/content-type is not supported by this framework' in str(e)


def test_module_transformer_init_valid_content_types():
    t = ModuleTransformer()
    assert content_types.JSON in t.VALID_CONTENT_TYPES
    assert content_types.CSV in t.VALID_CONTENT_TYPES


@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_containers.beta.framework.encoders.decode', return_value=[0])
def test_module_transformer_default_input_fn_with_json(decode, mx_ndarray_iter):
    model = Mock(data_shapes=[(1, (1,))])
    t = ModuleTransformer(model=model)

    input_data = Mock()
    content_type = 'application/json'
    t.default_input_fn(input_data, content_type)

    decode.assert_called_with(input_data, content_type)
    init_call = call(mx.nd.array([0]), batch_size=1, last_batch_handle='pad')
    assert init_call in mx_ndarray_iter.mock_calls


@patch('mxnet.nd.array')
@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_containers.beta.framework.encoders.decode', return_value=[0])
def test_module_transformer_default_input_fn_with_csv(decode, mx_ndarray_iter, mx_ndarray):
    ndarray = Mock(shape=(1, (1,)))
    ndarray.reshape.return_value = ndarray
    mx_ndarray.return_value = ndarray

    model = Mock(data_shapes=[(1, (1,))])
    t = ModuleTransformer(model=model)

    input_data = Mock()
    content_type = 'text/csv'
    t.default_input_fn(input_data, content_type)

    decode.assert_called_with(input_data, content_type)
    ndarray.reshape.assert_called_with((1,))
    init_call = call(mx.nd.array([0]), batch_size=1, last_batch_handle='pad')
    assert init_call in mx_ndarray_iter.mock_calls


def test_module_transformer_default_input_fn_invalid_content_type():
    t = ModuleTransformer()

    with pytest.raises(errors.UnsupportedFormatError) as e:
        t.default_input_fn(None, 'bad/content-type')
    assert 'Content type bad/content-type is not supported by this framework' in str(e)


def test_module_transformer_default_predict_fn():
    t = ModuleTransformer()
    module = Mock()
    data = Mock()

    t.default_predict_fn(data, module)
    module.predict.assert_called_with(data)


def test_gluon_transformer_default_predict_fn():
    data = [0]
    block = Mock()

    t = GluonBlockTransformer()
    t.default_predict_fn(data, block)

    block.assert_called_with(data)


@patch('sagemaker_containers.beta.framework.functions.error_wrapper', lambda x, y: x)
@patch('sagemaker_mxnet_container.serving.default_model_fn')
def test_user_module_transformer_with_transform_fn(model_fn):
    class UserModule:
        def __init__(self):
            self.transform_fn = Mock()

    user_module = UserModule()

    t = _user_module_transformer(user_module, MODEL_DIR)
    assert t._transform_fn == user_module.transform_fn


@patch('sagemaker_containers.beta.framework.functions.error_wrapper', lambda x, y: x)
@patch('sagemaker_mxnet_container.serving.default_model_fn')
def test_user_module_transformer_module_transformer_no_user_methods(model_fn):
    module = mx.module.BaseModule()
    model_fn.return_value = module

    user_module = None
    t = _user_module_transformer(user_module, MODEL_DIR)

    assert isinstance(t, ModuleTransformer)
    assert t._model == module
    assert t._model_fn == model_fn
    assert t._input_fn == t.default_input_fn
    assert t._predict_fn == t.default_predict_fn
    assert t._output_fn == t.default_output_fn


@patch('sagemaker_containers.beta.framework.functions.error_wrapper', lambda x, y: x)
def test_user_module_transformer_gluon_transformer_with_user_methods():
    gluon_block = mx.gluon.block.Block()

    class UserModule:
        def __init__(self):
            self.input_fn = Mock()
            self.predict_fn = Mock()
            self.output_fn = Mock()

        def model_fn(self, model_dir):
            return gluon_block

    user_module = UserModule()
    t = _user_module_transformer(user_module, MODEL_DIR)

    assert isinstance(t, GluonBlockTransformer)
    assert t._model == gluon_block
    assert t._model_fn == user_module.model_fn
    assert t._input_fn == user_module.input_fn
    assert t._predict_fn == user_module.predict_fn
    assert t._output_fn == user_module.output_fn


@patch('sagemaker_mxnet_container.serving.default_model_fn', return_value=Mock())
def test_user_module_transformer_unsupported_model_type(model_fn):
    user_module = None
    with pytest.raises(ValueError) as e:
        _user_module_transformer(user_module, MODEL_DIR)

    assert 'Unsupported model type' in str(e)
