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
import logging
import os

import mxnet as mx
from sagemaker_containers.beta.framework import (content_types, encoders, env, errors, modules,
                                                 transformer, worker)

logger = logging.getLogger(__name__)

PREFERRED_BATCH_SIZE_PARAM = 'SAGEMAKER_DEFAULT_MODEL_FIRST_DIMENSION_SIZE'
INFERENCE_ACCELERATOR_PRESENT_ENV = 'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT'
DEFAULT_ENV_VARS = {
    'MXNET_CPU_WORKER_NTHREADS': '1',
    'MXNET_CPU_PRIORITY_NTHREADS': '1',
    'MXNET_KVSTORE_REDUCTION_NTHREADS': '1',
    'MXNET_ENGINE_TYPE': 'NativeEngine',
    'OMP_NUM_THREADS': '1',
}

DEFAULT_MODEL_NAME = 'model'
DEFAULT_MODEL_FILENAMES = {
    'symbol': 'model-symbol.json',
    'params': 'model-0000.params',
    'shapes': 'model-shapes.json',
}


def default_model_fn(model_dir, preferred_batch_size=1):
    """Function responsible for loading the model. This implementation is designed to work with
    the default save function provided for MXNet training.

    Args:
        model_dir (str): The directory where model files are stored
        preferred_batch_size (int): The preferred batch size of the model's data shape (default: 1)

    Returns:
        mxnet.mod.Module: the loaded model.
    """
    for f in DEFAULT_MODEL_FILENAMES.values():
        path = os.path.join(model_dir, f)
        if not os.path.exists(path):
            raise ValueError('Failed to load model with default model_fn: missing file {}.'
                             'Expected files: {}'.format(f, [file_name for _, file_name
                                                             in DEFAULT_MODEL_FILENAMES.items()]))

    shapes_file = os.path.join(model_dir, DEFAULT_MODEL_FILENAMES['shapes'])
    preferred_batch_size = preferred_batch_size or os.environ.get(PREFERRED_BATCH_SIZE_PARAM)
    data_names, data_shapes = _read_data_shapes(shapes_file, preferred_batch_size)

    sym, args, aux = mx.model.load_checkpoint(os.path.join(model_dir, DEFAULT_MODEL_NAME), 0)

    # TODO mxnet ctx - better default, allow user control
    context = mx.cpu()

    if os.environ.get(INFERENCE_ACCELERATOR_PRESENT_ENV) == 'true':
        context = mx.eia()

    mod = mx.mod.Module(symbol=sym, context=context, data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(args, aux, allow_missing=True)

    return mod


def _read_data_shapes(path, preferred_batch_size=1):
    with open(path, 'r') as f:
        signature = json.load(f)

    data_names = []
    data_shapes = []

    for s in signature:
        name = s['name']
        data_names.append(name)

        shape = s['shape']

        if preferred_batch_size:
            shape[0] = preferred_batch_size

        data_shapes.append((name, shape))

    return data_names, data_shapes


class MXNetTransformer(transformer.Transformer):
    """Base class for creating Transformers with default methods for use with MXNet models.
    """

    VALID_CONTENT_TYPES = (content_types.JSON,)

    def __init__(self, model=None, model_fn=None, input_fn=None, predict_fn=None, output_fn=None,
                 error_class=None):
        """Initialize an ``MXNetTransformer``. For each function, if one is not specified,
        a default implementation is used.

        Args:
            model (obj): a loaded model object that is ready for to be used for prediction
            model_fn (fn): a function that loads a model
            input_fn (fn): a function that takes request data and deserializes it for prediction
            predict_fn (fn): a function that performs prediction with a model
            output_fn (fn): a function that serializes a prediction into a response
            error_class (Exception): the error class used to wrap functions that are not
                the default ones defined in SageMaker Containers.
        """
        input_fn = input_fn or self.default_input_fn
        predict_fn = predict_fn or self.default_predict_fn
        output_fn = output_fn or self.default_output_fn

        super(MXNetTransformer, self).__init__(model_fn=model_fn, input_fn=input_fn,
                                               predict_fn=predict_fn, output_fn=output_fn,
                                               error_class=error_class)
        self._model = model

    def initialize(self):
        """Execute any initialization necessary to start making predictions with the Transformer.
        This method will load a model if it hasn't been loaded already.
        """
        if self._model is None:
            super(MXNetTransformer, self).initialize()

    def default_input_fn(self, input_data, content_type):
        """Take request data and deserialize it into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request's content type, for example "application/json"
            - The request data

        The ``input_fn`` is responsible for preprocessing request data before prediction.

        Args:
            input_data (obj): the request data
            content_type (str): the request's content type

        Returns:
            mxnet.nd.array: an MXNet NDArray

        Raises:
            sagemaker_containers.beta.framework.errors.UnsupportedFormatError: if an unsupported
                content type is used.
        """
        if content_type in self.VALID_CONTENT_TYPES:
            np_array = encoders.decode(input_data, content_type)
            return mx.nd.array(np_array)
        else:
            raise errors.UnsupportedFormatError(content_type)

    def default_predict_fn(self, data, model):
        """Use the model to create a prediction for the data.

        Args:
            data (obj): input data for prediction
            model (obj): the loaded model

        Returns:
            obj: the prediction result
        """
        transformer.default_predict_fn(data, model)

    def default_output_fn(self, prediction, accept):
        """Serialize the prediction into a response.

        Args:
            prediction (mxnet.nd.array): an MXNet NDArray that is the result of a prediction
            accept (str): the accept content type expected by the client

        Returns:
            sagemaker_containers.beta.framework.worker.Response: a Flask response object

        Raises:
            sagemaker_containers.beta.framework.errors.UnsupportedFormatError: if an unsupported
                accept type is used.
        """
        if accept in self.VALID_CONTENT_TYPES:
            return worker.Response(response=encoders.encode(prediction.asnumpy().tolist(), accept),
                                   mimetype=accept)
        else:
            raise errors.UnsupportedFormatError(accept)


class ModuleTransformer(MXNetTransformer):

    VALID_CONTENT_TYPES = (content_types.JSON, content_types.CSV)

    def default_input_fn(self, input_data, content_type):
        """Take request data and deserialize it into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request's content type, for example "application/json"
            - The request data

        The ``input_fn`` is responsible for preprocessing request data before prediction.

        Args:
            input_data (obj): the request data
            content_type (str): the request's content type

        Returns:
            mxnet.io.NDArrayIter: data ready for prediction.

        Raises:
            sagemaker_containers.beta.framework.errors.UnsupportedFormatError: if an unsupported
                accept type is used.
        """
        if content_type not in self.VALID_CONTENT_TYPES:
            raise errors.UnsupportedFormatError(content_type)

        np_array = encoders.decode(input_data, content_type)
        ndarray = mx.nd.array(np_array)

        # We require model to only have one input
        [data_shape] = self._model.data_shapes

        # Reshape flattened CSV as specified by the model
        if content_type == content_types.CSV:
            _, target_shape = data_shape
            ndarray = ndarray.reshape(target_shape)

        # Batch size is first dimension of model input
        model_batch_size = data_shape[1][0]
        pad_rows = max(0, model_batch_size - ndarray.shape[0])

        # If ndarray has fewer rows than model_batch_size, then pad it with zeros.
        if pad_rows:
            num_pad_values = pad_rows
            for dimension in ndarray.shape[1:]:
                num_pad_values *= dimension
            padding_shape = tuple([pad_rows] + list(ndarray.shape[1:]))
            padding = mx.ndarray.zeros(shape=padding_shape)
            ndarray = mx.ndarray.concat(ndarray, padding, dim=0)

        model_input = mx.io.NDArrayIter(ndarray, batch_size=model_batch_size,
                                        last_batch_handle='pad')

        if pad_rows:
            # Update the getpad method on the model_input data iterator to return the amount of
            # padding. MXNet will ignore the last getpad() rows during Module predict.
            def _getpad():
                return pad_rows

            model_input.getpad = _getpad

        return model_input

    def default_predict_fn(self, data, module):
        """Use the model to create a prediction for the data.

        Args:
            data (mxnet.io.NDArrayIter): input data for prediction
            model (mxnet.module.BaseModule): an MXNet Module

        Returns:
            list: the prediction result. This will be either a list of ``NDArray`` or
                a list of lists of ``NDArray``
        """
        return module.predict(data)


class GluonBlockTransformer(MXNetTransformer):
    def default_predict_fn(self, data, block):
        """Use the model to create a prediction for the data.

        Args:
            data (mxnet.nd.array): input data for prediction (deserialized by ``input_fn``)
            block (mxnet.gluon.block.Block): a Gluon neural network

        Returns:
            mxnet.nd.array: the prediction result
        """
        return block(data)


def _update_mxnet_env_vars():
    for k, v in DEFAULT_ENV_VARS.items():
        if k not in os.environ:
            os.environ[k] = v


def _transformer_with_transform_fn(model_fn, transform_fn):
    user_transformer = transformer.Transformer(model_fn=model_fn, transform_fn=transform_fn)
    user_transformer.initialize()
    return user_transformer


def _user_module_transformer(user_module, model_dir):
    model_fn = getattr(user_module, 'model_fn', default_model_fn)

    if hasattr(user_module, 'transform_fn'):
        return _transformer_with_transform_fn(model_fn, getattr(user_module, 'transform_fn'))

    model = model_fn(model_dir)
    if isinstance(model, mx.module.BaseModule):
        transformer_cls = ModuleTransformer
    elif isinstance(model, mx.gluon.block.Block):
        transformer_cls = GluonBlockTransformer
    else:
        raise ValueError('Unsupported model type: {}'.format(model.__class__.__name__))

    input_fn = getattr(user_module, 'input_fn', None)
    predict_fn = getattr(user_module, 'predict_fn', None)
    output_fn = getattr(user_module, 'output_fn', None)

    return transformer_cls(model=model, model_fn=model_fn, input_fn=input_fn,
                           predict_fn=predict_fn, output_fn=output_fn)


app = None


def main(environ, start_response):
    global app
    if app is None:
        serving_env = env.ServingEnv()
        _update_mxnet_env_vars()

        user_module = modules.import_module(serving_env.module_dir, serving_env.module_name)
        user_transformer = _user_module_transformer(user_module, serving_env.model_dir)

        app = worker.Worker(transform_fn=user_transformer.transform,
                            module_name=serving_env.module_name)

    return app(environ, start_response)
