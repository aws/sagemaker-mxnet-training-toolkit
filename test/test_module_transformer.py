import json
import logging
import os
import pytest
import tempfile
from container_support.serving import UnsupportedContentTypeError, \
    UnsupportedAcceptTypeError, \
    UnsupportedInputShapeError, \
    JSON_CONTENT_TYPE, \
    CSV_CONTENT_TYPE
from mock import Mock
from mock import patch, MagicMock
from types import ModuleType

JSON_DATA = json.dumps({'k1': 'v1', 'k2': [1, 2, 3]})
CSV_INPUT = "1,2,3\r\n"


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


@pytest.fixture()
def user_module():
    m = ModuleType('mod')
    m.model_fn = _model_fn
    yield m


@pytest.fixture()
def transformer(mxc, user_module):
    import mxnet_container
    from mxnet_container.serve.transformer import ModuleTransformer
    with patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class') as select:
        select.return_value = ModuleTransformer
        yield mxnet_container.serve.transformer.transformer(user_module)


@pytest.fixture
def module_module(mxc):
    mock = Mock('BaseModule')
    m = ModuleType('mod')
    m.model_fn = lambda x: mock
    m.input_fn = lambda a, b, c: 'input({})'.format(str(b))
    m.predict_fn = lambda a, b: 'predict({})'.format(str(b))
    m.output_fn = lambda a, b: ('output({})'.format(str(a)), b)

    # extras for test
    m._module = mock
    return m


class TestModuleTransformer(object):
    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    def test_from_module(self, select, mxc, module_module):
        from mxnet_container.serve.transformer import MXNetTransformer, ModuleTransformer
        select.return_value = ModuleTransformer

        t = MXNetTransformer.from_module(module_module)
        assert isinstance(t, ModuleTransformer)
        assert t.model == module_module._module
        assert t.transform('x', JSON_CONTENT_TYPE, JSON_CONTENT_TYPE) == \
               ('output(predict(input(x)))', JSON_CONTENT_TYPE)

    def test_transformer_from_module_transform_fn(self, mxc, user_module):
        import mxnet_container
        user_module.transform_fn = _transform_fn
        t = mxnet_container.serve.transformer.transformer(user_module)
        assert t.transform("data", JSON_CONTENT_TYPE, JSON_CONTENT_TYPE) == \
               ("transform_fn data", JSON_CONTENT_TYPE)

    def test_transformer_from_module_separate_fn(self, mxc, user_module):
        user_module.process_request_fn = _process_request_fn
        user_module.output_fn = _output_fn
        t = next(transformer(mxc, user_module))
        assert t.transform("data", JSON_CONTENT_TYPE, JSON_CONTENT_TYPE) == \
               ("output_fn predict_fn input_fn data", JSON_CONTENT_TYPE)

    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    @patch('mxnet_container.serve.transformer.ModuleTransformer._default_model_fn')
    def test_transformer_from_module_default_fns(self, model_fn, select, mxc):
        import mxnet_container
        model_fn.return_value = DummyModel()
        select.return_value = mxnet_container.serve.transformer.ModuleTransformer

        m = ModuleType('mod')  # an empty module
        t = mxnet_container.serve.transformer.transformer(m)
        assert hasattr(t, 'model')
        assert hasattr(t, 'transform_fn')

    def test_transformer_default_handler_json(self, mxc, transformer):
        with patch('json.dumps') as patched:
            patched.return_value = JSON_DATA
            response, response_content_type = transformer.transform(JSON_DATA, JSON_CONTENT_TYPE, JSON_CONTENT_TYPE)

        assert JSON_DATA == response
        assert JSON_CONTENT_TYPE == response_content_type

    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    def test_transformer_default_handler_csv(self, select, mxc):
        import mxnet_container

        m = ModuleType('mod')
        m.model_fn = _model_fn_csv
        select.return_value = mxnet_container.serve.transformer.ModuleTransformer

        csv_transformer = mxnet_container.serve.transformer.transformer(m)

        response, response_content_type = csv_transformer.transform(CSV_INPUT, CSV_CONTENT_TYPE, CSV_CONTENT_TYPE)

        assert CSV_INPUT == response
        assert CSV_CONTENT_TYPE == response_content_type

    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    def test_transformer_default_handler_csv_empty(self, select, mxc):
        import mxnet_container
        select.return_value = mxnet_container.serve.transformer.ModuleTransformer

        m = ModuleType('mod')
        m.model_fn = _model_fn_csv

        csv_transformer = mxnet_container.serve.transformer.transformer(m)

        response, response_content_type = csv_transformer.transform("", CSV_CONTENT_TYPE, CSV_CONTENT_TYPE)

        assert CSV_INPUT == response
        assert CSV_CONTENT_TYPE == response_content_type

    @patch('mxnet_container.serve.transformer.MXNetTransformer.select_transformer_class')
    def test_transformer_default_handler_csv_wrong_shape(self, select, mxc):
        import mxnet_container
        select.return_value = mxnet_container.serve.transformer.ModuleTransformer

        m = ModuleType('mod')
        m.model_fn = _model_fn_csv_wrong_shape

        csv_transformer = mxnet_container.serve.transformer.transformer(m)

        with pytest.raises(UnsupportedInputShapeError):
            csv_transformer.transform(CSV_INPUT, CSV_CONTENT_TYPE, CSV_CONTENT_TYPE)

    def test_transformer_default_handler_unsupported_content_type(self, transformer):
        with pytest.raises(UnsupportedContentTypeError):
            transformer.transform(JSON_DATA, "application/bad", JSON_CONTENT_TYPE)

    def test_transformer_default_handler_unsupported_accept_type(self, transformer):
        with pytest.raises(UnsupportedAcceptTypeError):
            transformer.transform(JSON_DATA, JSON_CONTENT_TYPE, "application/bad")

    def test_transformer_read_data_shapes(self, mxc, user_module):
        from mxnet_container.serve.transformer import ModuleTransformer
        data_shapes = [
            {"name": "data1", "shape": [10, 2, 3, 4]},
            {"name": "data2", "shape": [13, 4, 5, 6]}
        ]

        fname = tempfile.mkstemp()[1]
        try:
            with open(fname, 'w') as f:
                json.dump(data_shapes, f)

            names, shapes = ModuleTransformer._read_data_shapes(f.name)
            assert 2 == len(shapes)
            assert ('data1', [1, 2, 3, 4]) in shapes
            assert ('data2', [1, 4, 5, 6]) in shapes

        finally:
            os.remove(fname)


class DummyModel(object):
    def predict(self, data):
        nd_array = Mock(name='ndarray')
        nd_array.asnumpy = Mock(name='asnumpy')
        nd_array.asnumpy().tolist = Mock(name='tolist', return_value=[1, 2, 3])
        nd_array.asnumpy().flatten = Mock(name='flatten', return_value=[1, 2, 3])
        nd_array.shape = [1, 1, 1]
        return [nd_array]

    @property
    def data_shapes(self):
        return [["DataDesc1", (1, 1, 3)]]


class DummyModelForCsv(object):
    def __init__(self, wrong_shape=False):
        self.make_wrong_shape = wrong_shape

    def predict(self, data):
        nd_array = Mock(name='ndarray')
        nd_array.asnumpy = Mock(name='asnumpy')
        nd_array.asnumpy().tolist = Mock(name='tolist', return_value=[1, 2, 3])
        nd_array.asnumpy().flatten = Mock(name='flatten', return_value=[1, 2, 3])
        nd_array.shape = [1, 1, 1]
        return [nd_array]

    @property
    def data_shapes(self):
        if self.make_wrong_shape:
            return []
        return [["DataDesc1", (1, 1, 3)]]


def _model_fn(model_dir):
    return DummyModel()


def _model_fn_csv(model_dir):
    return DummyModelForCsv()


def _model_fn_csv_wrong_shape(model_dir):
    return DummyModelForCsv(wrong_shape=True)


def _process_request_fn(model, data, content_type):
    return "predict_fn input_fn " + data


def _output_fn(data, content_type):
    return "output_fn " + data, JSON_CONTENT_TYPE


def _transform_fn(model, data, input_content_type, output_content_type):
    return "transform_fn " + str(data), JSON_CONTENT_TYPE
