#  Copyright <YEAR> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import pytest
from container_support.serving import UnsupportedContentTypeError, \
    UnsupportedAcceptTypeError, \
    JSON_CONTENT_TYPE
from mock import Mock
from mock import patch, MagicMock
from types import ModuleType


@pytest.fixture()
def mxc():
    os.environ['SAGEMAKER_CONTAINER_LOG_LEVEL'] = str(logging.INFO)
    os.environ['SAGEMAKER_REGION'] = 'us-west-2'

    mxnet_mock = MagicMock()
    modules = {
        'mxnet': mxnet_mock,
    }
    ndarray = Mock('ndarray')
    ndarray.asnumpy = Mock(name='asnumpy')
    ndarray.asnumpy().tolist = Mock(name='tolist', return_value=[1, 2, 3])
    ndarray.asnumpy().flatten = Mock(name='flatten', return_value=[1, 2, 3])
    ndarray.reshape = Mock(return_value=ndarray)
    ndarray.shape = [1, 1, 1]
    mxnet_mock.nd.array = Mock(name='array', return_value=ndarray)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    import mxnet_container
    yield mxnet_container
    patcher.stop()


generic_model = object()


def generic_model_fn(model_dir):
    return generic_model


def generic_transform_fn(model, input_data, content_type, accept):
    return input_data, accept


@pytest.fixture
def generic_module():
    m = ModuleType('mod')
    m.model_fn = generic_model_fn
    m.transform_fn = generic_transform_fn
    return m


class TestMXNetTransformer(object):
    def test_from_module_complete(self, mxc):
        from mxnet_container.serve.transformer import MXNetTransformer
        t = MXNetTransformer.from_module(generic_module())
        assert isinstance(t, MXNetTransformer)
        assert t.model == generic_model
        assert t.transform_fn == generic_transform_fn
        assert t.transform('x', 'content-type', 'accept') == ('x', 'accept')

    @patch('mxnet_container.serve.transformer.ModuleTransformer._default_model_fn')
    def test_from_module_with_default_model_fn(self, model_fn, mxc, generic_module):
        from mxnet_container.serve.transformer import MXNetTransformer
        model_fn.return_value = generic_model
        del generic_module.model_fn

        t = MXNetTransformer.from_module(generic_module)
        # expect MXNetTransformer with transform_fn from module, model from default_model_fn
        assert isinstance(t, MXNetTransformer)
        assert t.model == generic_model
        assert t.transform_fn == generic_transform_fn


@pytest.fixture
def gluon_module(mxc):
    mock = Mock('Block')
    m = ModuleType('mod')
    m.model_fn = lambda x: mock
    m.input_fn = lambda a, b: 'input({})'.format(str(a))
    m.predict_fn = lambda a, b: 'predict({})'.format(str(b))
    m.output_fn = lambda a, b: ('output({})'.format(str(a)), b)

    # extra attribute for test
    m._block = mock
    return m


class TestGluonBlockTransformer(object):
    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    def test_from_module(self, select, mxc, gluon_module):
        from mxnet_container.serve.transformer import MXNetTransformer, GluonBlockTransformer
        select.return_value = GluonBlockTransformer

        t = MXNetTransformer.from_module(gluon_module)
        assert isinstance(t, GluonBlockTransformer)
        assert t.model == gluon_module._block
        assert t.transform('x', JSON_CONTENT_TYPE, JSON_CONTENT_TYPE) == \
               ('output(predict(input(x)))', JSON_CONTENT_TYPE)

    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    @patch('mxnet_container.serve.transformer.GluonBlockTransformer._default_output_fn')
    @patch('mxnet_container.serve.transformer.GluonBlockTransformer._default_predict_fn')
    @patch('mxnet_container.serve.transformer.GluonBlockTransformer._default_input_fn')
    def test_from_module_with_defaults(self, input_fn, predict_fn, output_fn,
                                       select, mxc, gluon_module):
        from mxnet_container.serve.transformer import MXNetTransformer, GluonBlockTransformer
        select.return_value = GluonBlockTransformer

        # remove the handlers so we can test default handlers
        del gluon_module.input_fn
        del gluon_module.predict_fn
        del gluon_module.output_fn

        input_fn.return_value = 'default_input'
        predict_fn.return_value = 'default_predict'
        output_fn.return_value = 'default_output', 'accept'

        t = MXNetTransformer.from_module(gluon_module)
        assert isinstance(t, GluonBlockTransformer)
        assert t.model == gluon_module._block
        assert t.transform('x', JSON_CONTENT_TYPE, JSON_CONTENT_TYPE) == \
               ('default_output', 'accept')

        input_fn.assert_called_with('x', JSON_CONTENT_TYPE)
        predict_fn.assert_called_with(gluon_module._block, 'default_input')
        output_fn.assert_called_with('default_predict', JSON_CONTENT_TYPE)

    def test_default_input_fn(self, mxc):
        import mxnet
        from mxnet_container.serve.transformer import GluonBlockTransformer
        _ = GluonBlockTransformer._default_input_fn('[[1,2,3,4]]', JSON_CONTENT_TYPE)
        mxnet.nd.array.assert_called_with([[1, 2, 3, 4]])

    def test_default_input_fn_unsupported_content_type(self, mxc):
        from mxnet_container.serve.transformer import GluonBlockTransformer

        with pytest.raises(UnsupportedContentTypeError):
            GluonBlockTransformer._default_input_fn('whatever', 'wrong content type')

    def test_default_predict_fn(self, mxc):
        from mxnet_container.serve.transformer import GluonBlockTransformer

        # block, ndarray could be any compatible callable/arg pair
        block = list
        ndarray = [1, 2, 3]

        result = GluonBlockTransformer._default_predict_fn(block, ndarray)

        assert [1, 2, 3] == result

    def test_default_output_fn(self, mxc):
        import mxnet
        from mxnet_container.serve.transformer import GluonBlockTransformer
        mock_ndarray = mxnet.nd.array()
        output, accept = GluonBlockTransformer._default_output_fn(mock_ndarray, JSON_CONTENT_TYPE)
        assert accept == JSON_CONTENT_TYPE
        assert output == '[1, 2, 3]'

    def test_default_output_fn_unsupported_content_type(self, mxc):
        from mxnet_container.serve.transformer import GluonBlockTransformer

        with pytest.raises(UnsupportedAcceptTypeError):
            GluonBlockTransformer._default_output_fn('whatever', 'wrong content type')
