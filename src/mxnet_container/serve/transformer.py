#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import csv
import json
import os
from types import ModuleType

import mxnet as mx
from container_support.serving import UnsupportedContentTypeError, \
    UnsupportedAcceptTypeError, \
    UnsupportedInputShapeError, \
    JSON_CONTENT_TYPE, \
    CSV_CONTENT_TYPE
from six import StringIO

from mxnet_container.serve.environment import MXNetHostingEnvironment
from mxnet_container.train import DEFAULT_MODEL_FILENAMES, DEFAULT_MODEL_NAME


def transformer(user_module):
    return MXNetTransformer.from_module(user_module)


def load_dependencies():
    pass


class MXNetTransformer(object):
    """A ``Transformer`` encapsulates the function(s) responsible for parsing
    incoming request data, passing it through a model, and converting the
    result into something that can be returned as the body of and HTTP response.
    """

    def __init__(self, model, transform_fn):
        """Initialize a Transformer.

        :param model: a fully initialized model
        :param transform_fn: a transformer function
        """
        self.model = model
        self.transform_fn = transform_fn

    def transform(self, input_data, content_type, accept):
        """Transforms input data into a prediction result.
        The input data is expected to have the given ``input_content_type``.
        The output returned should have the given ``output_content_type``.

        :param input_data: input data
        :param content_type: content type from Content-Type headers
        :param accept: content type from Accept header
        :return: the transformed result
        """
        return self.transform_fn(self.model, input_data, content_type, accept)

    @classmethod
    def select_transformer_class(cls, model):
        if isinstance(model, mx.module.BaseModule):
            return ModuleTransformer

        if isinstance(model, mx.gluon.block.Block):
            return GluonBlockTransformer

        raise ValueError('Unsupported model type: {}'.format(model.__class__.__name__))

    @classmethod
    def from_module(cls, m):
        """Initialize a Transformer using functions supplied by the given module.

        The module may provide a ``model_fn`` that returns a fully initialized model of
        some kind. Generally this will be a Gluon ``Block`` or a Module API ``Module``, but
        it can be anything, as long as it is compatible with the ``transform_fn``.

        If the ``model_fn`` is not present, a default implementation will be used instead. The
        default implementation is compatible with the ``Module``s saved by the ``default_save``
        method in MXNetTrainingEnvironment.

        The ``model_fn`` (user-supplied or default) will be called once during
        each inference worker's startup process.

        The module may supply a ``transform_fn``. If it is present, it will be used to handle
        each inference request. If it is not present, then a ``transform_fn`` will be composed
        by chaining an ``input_fn``, ``predict_fn`` and ``output_fn``. If any of these are
        implemented in the given module, they will be used. Otherwise default implementations will
        be used instead.

        :param m: a python module
        :return: a configured Transformer object
        """

        if not isinstance(m, ModuleType):
            raise ValueError("not a module!")

        env = MXNetHostingEnvironment()

        # load model
        if hasattr(m, 'model_fn'):
            model = m.model_fn(env.model_dir)
        else:
            model = ModuleTransformer._default_model_fn(env.model_dir, env.preferred_batch_size)

        # if user has supplied a transform_fn, we can use base MXNetTransformer directly
        if hasattr(m, 'transform_fn'):
            return MXNetTransformer(model, m.transform_fn)

        # otherwise we need to create a Module- or Gluon-specific subclass
        transformer_class = cls.select_transformer_class(model)
        return transformer_class.from_module(m, model)


class GluonBlockTransformer(MXNetTransformer):
    def __init__(self, block, transform_fn):
        super(GluonBlockTransformer, self).__init__(block, transform_fn)

    @classmethod
    def from_module(cls, module, block):
        input_fn = GluonBlockTransformer._get_function(module, 'input_fn')
        predict_fn = GluonBlockTransformer._get_function(module, 'predict_fn')
        output_fn = GluonBlockTransformer._get_function(module, 'output_fn')

        def transform_fn(block, data, content_type, accept):
            i = input_fn(data, content_type)
            p = predict_fn(block, i)
            o, ct = output_fn(p, accept)
            return o, ct

        return cls(block, transform_fn)

    @classmethod
    def _get_function(cls, module, name):
        if hasattr(module, name):
            return getattr(module, name)
        else:
            return getattr(cls, '_default_' + name)

    @staticmethod
    def _default_input_fn(input, content_type):
        """A default input handler for Gluon ``Block``s.
        :param input: the request payload
        :param content_type: the request content_type (must equal JSON_CONTENT_TYPE)
        :return: NDArray to pass to ``predict_fn``
        """
        if JSON_CONTENT_TYPE == content_type:
            return mx.nd.array(json.loads(input))

        raise UnsupportedContentTypeError(content_type)

    @staticmethod
    def _default_predict_fn(block, ndarray):
        """A default prediction function for Gluon ``Block``s.
        :param block: a Gluon ``Block``
        :param ndarray: an NDArray (axis 1 = batch index)
        :return: an NDArray
        """
        return block(ndarray)

    @staticmethod
    def _default_output_fn(ndarray, accept):
        """A default output handler for Gluon ``Block``s.

        :param ndarray: an NDArray
        :param accept: content type from accept header (must equal JSON_CONTENT_TYPE)
        :return: a json string
        :raises: UnsupportedAcceptTypeError if accept != JSON_CONTENT_TYPE
        """
        if JSON_CONTENT_TYPE == accept:
            return json.dumps(ndarray.asnumpy().tolist()), JSON_CONTENT_TYPE

        raise UnsupportedAcceptTypeError(accept)


class ModuleTransformer(MXNetTransformer):
    def __init__(self, module, transform_fn):
        super(ModuleTransformer, self).__init__(module, transform_fn)

    @classmethod
    def from_module(cls, m, model):
        """Initialize a Transformer using functions supplied by the given module.

        If the module contains a ``transform_fn``, it will be used to handle incoming request
        data, execute the model prediction, and generation of response content.

        If the module does not contain a ``transform_fn``, then one will be assembled by:
        - chaining a ``process_request_fn`` and ``output_fn`` if ``process_request_fn`` is defined
        - otherwise: chaining an ``input_fn``, ``predict_fn``, and ``output_fn``
        Default handlers will be used for any of these that are not present in the supplied module.

        :param m: a python module
        :return: a configured Transformer object
        """
        # TODO remove process_request_fn?
        if hasattr(m, 'process_request_fn'):
            process_fn = m.process_request_fn
        else:
            input_fn = cls._default_input_fn if not hasattr(m, 'input_fn') else m.input_fn
            predict_fn = cls._default_predict_fn if not hasattr(m, 'predict_fn') else m.predict_fn
            process_fn = cls._process_request_fn(input_fn, predict_fn)

        if hasattr(m, 'output_fn'):
            output_fn = m.output_fn
        else:
            output_fn = cls._default_output_fn

        transform_fn = cls._build_transform_fn(process_fn, output_fn)

        return cls(model, transform_fn)

    @staticmethod
    def _process_request_fn(input_handler, prediction_handler):
        """Construct processing function from handlers.

        :param input_handler: handles input and transforms for predict call
        :param prediction_handler: consumes data from input handler and calls predict
        :return: processing function
        """

        def process(model, data, content_type):
            """Processing function for MXNet models.

            :param model: loaded MXNet model
            :param data: data from the request
            :param content_type: specified in the request
            :return: a list of NDArray
            """
            return prediction_handler(model, input_handler(model, data, content_type))

        return process

    @staticmethod
    def _default_input_fn(model, data, content_type):
        """A default input handler for MXNet models to support default input.

        :param model: loaded MXNet model
        :param data: data from the request
        :param content_type: specified in the request
        :return: NDArrayIter to feed to predict call
        """

        if content_type == JSON_CONTENT_TYPE:
            return ModuleTransformer._process_json_input(model, data)

        if content_type == CSV_CONTENT_TYPE:
            return ModuleTransformer._process_csv_input(model, data)

        raise UnsupportedContentTypeError(content_type)

    @staticmethod
    def _process_json_input(model, data):
        """A default inout handler for json input.

        'data' is deserialized from json into NDArray. This array is used to create
        iterator that is used to call 'predict' on the model.

        :param data: json data from the request
        :return: NDArrayIter to feed to predict call
        """

        parsed = json.loads(data)
        return ModuleTransformer._prepare_input_for_default_predict(model, mx.nd.array(parsed))

    @staticmethod
    def _process_csv_input(model, data):
        """A default prediction function for MXNet models that takes csv as input.

        :param model: loaded MXNet model
        :param data: data from the request
        :return: NDArrayIter to feed to predict call
        """

        # we can only support case when there is a single data input
        if len(model.data_shapes) != 1:
            raise UnsupportedInputShapeError(len(model.data_shapes))

        # model is already loaded with data shape bound,
        # ignore the first parameter that is batch_size
        model_data_shape = model.data_shapes[0]
        (shape_name, input_data_shape) = model_data_shape
        no_batch_data_shape = input_data_shape[1:]

        # let's read the csv into ndarray doing reshaping as
        # specified by the model since csv is arriving flattened
        csv_buff = StringIO(data)
        csv_to_parse = csv.reader(csv_buff, delimiter=',')
        full_array = []
        for row in csv_to_parse:
            casted_list = [float(i) for i in row]
            shaped_row = mx.nd.array(casted_list).reshape(no_batch_data_shape)
            full_array.append(shaped_row.asnumpy().tolist())
        return ModuleTransformer._prepare_input_for_default_predict(model, mx.nd.array(full_array))

    @staticmethod
    def _prepare_input_for_default_predict(model, ndarray):
        # We require model to only have one input
        [data_shape] = model.data_shapes

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

    @staticmethod
    def _build_transform_fn(process_request_fn, output_fn):
        """ Create a transformer function.
        :param process_request_fn: input processing function
        :param output_fn: an output handler function
        :return:
        """

        def f(model, data, input_content_type, requested_output_content_type):
            prediction_result = process_request_fn(model, data, input_content_type)
            o, ct = output_fn(prediction_result, requested_output_content_type)
            return o, ct

        return f

    @staticmethod
    def _default_predict_fn(module, data):
        """A default prediction function for MXNet models.
        :param module: an MXNet Module
        :param data: NDArrayIter
        :return: a list of NDArray or list of lists of NDArray
        """

        return module.predict(data)

    @staticmethod
    def _default_output_fn(data, content_type):
        """A default output handler for MXNet models.

        :param data: output of ``mxnet.Module.predict(...)``
        :param content_type: requested content type by the request to be returned
        :return: a json string
        """

        if content_type == JSON_CONTENT_TYPE:
            result_to_send = [arr.asnumpy().tolist() for arr in data]
            return json.dumps(result_to_send), JSON_CONTENT_TYPE

        if content_type == CSV_CONTENT_TYPE:
            result_to_send = [arr.asnumpy().flatten() for arr in data]
            str_io = StringIO()
            csv_writer = csv.writer(str_io, delimiter=',')
            for row in result_to_send:
                csv_writer.writerow(row)
            return str_io.getvalue(), CSV_CONTENT_TYPE

        raise UnsupportedAcceptTypeError(content_type)

    @staticmethod
    def _default_model_fn(model_dir, preferred_batch_size):
        for f in DEFAULT_MODEL_FILENAMES.values():
            path = os.path.join(model_dir, f)
            if not os.path.exists(path):
                raise ValueError('missing %s file' % f)

        shapes_file = os.path.join(model_dir, DEFAULT_MODEL_FILENAMES['shapes'])
        data_names, data_shapes = ModuleTransformer._read_data_shapes(shapes_file,
                                                                      preferred_batch_size)

        sym, args, aux = mx.model.load_checkpoint('%s/%s' % (model_dir, DEFAULT_MODEL_NAME), 0)

        # TODO mxnet ctx - better default, allow user control
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes)
        mod.set_params(args, aux, allow_missing=True)

        return mod

    @staticmethod
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
