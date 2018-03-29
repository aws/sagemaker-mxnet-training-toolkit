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

import json
import os
import pytest
import shutil
import tempfile

from container_support import ContainerEnvironment
from inspect import getargspec
from mock import patch, MagicMock, create_autospec

INPUT_DATA_CONFIG = {
    "train": {"ContentType": "trainingContentType"},
    "evaluation": {"ContentType": "evalContentType"},
    "Validation": {}
}

HYPERPARAMETERS = {
    ContainerEnvironment.USER_SCRIPT_NAME_PARAM: 'myscript.py',
    ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM: 's3://mybucket/code.tar.gz',
    "sagemaker_s3_uri_training": "blah/blah",
    "sagemaker_s3_uri_validation": "xxx/yyy",
    'sagemaker_region': "us-west-2"
}

class NoKWArgsModule:
    def train(self, hyperparameters):
        pass

class KWArgsModule:
    def train(self, **kwargs):
        pass

getargspec_orig = getargspec


def train_no_kwargs_mock():
    return create_autospec(NoKWArgsModule)


def train_kwargs_mock():
    return create_autospec(KWArgsModule)


@pytest.fixture()
def mxc():
    mxnet_mock = MagicMock()
    modules = {
        'mxnet': mxnet_mock
    }

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    import mxnet_container
    yield mxnet_container
    patcher.stop()


@pytest.fixture()
def training():
    d = optml()
    yield d
    shutil.rmtree(d)


def optml():
    tmp = tempfile.mkdtemp()
    for d in ['input/data/training', 'input/config', 'model', 'output/data']:
        os.makedirs(os.path.join(tmp, d))

    with open(os.path.join(tmp, 'input/data/training/data.csv'), 'w') as f:
        f.write('dummy data file')

    _write_resource_config(tmp, 'a', ['a', 'b'])
    _write_config_file(tmp, 'inputdataconfig.json', INPUT_DATA_CONFIG)
    _write_config_file(tmp, 'hyperparameters.json', _serialize_hyperparameters(HYPERPARAMETERS))

    return tmp


def test_mxnet_env_is_distributed(mxc, training):
    from mxnet_container.train import MXNetTrainingEnvironment

    with patch('socket.gethostbyname') as patched:
        mxnet_env = MXNetTrainingEnvironment(training)
        assert mxnet_env.distributed


def test_mxnet_env_is_not_distributed(mxc, training):
    from mxnet_container.train import MXNetTrainingEnvironment

    _write_resource_config(training, 'a', ['a'])

    with patch('socket.gethostbyname') as patched:
        mxnet_env = MXNetTrainingEnvironment(training)
        assert not mxnet_env.distributed


def test_mnxet_env_env_vars(mxc, training):
    from mxnet_container.train import MXNetTrainingEnvironment

    with patch('socket.gethostbyname') as patched:
        patched.return_value = '0.0.0.0'
        mxnet_env = MXNetTrainingEnvironment(training)
        assert mxnet_env.env_vars_for_role('worker') == {
            'DMLC_NUM_WORKER': "2",
            'DMLC_NUM_SERVER': "2",
            'DMLC_ROLE': 'worker',
            'DMLC_PS_ROOT_URI': '0.0.0.0',
            'DMLC_PS_ROOT_PORT': "8000",
            'PS_VERBOSE': "0"
        }


def test_mxnet_env_is_current_host_scheduler(mxc, training):
    from mxnet_container.train import MXNetTrainingEnvironment

    with patch('socket.gethostbyname') as patched:
        mxnet_env = MXNetTrainingEnvironment(training)
        assert mxnet_env.current_host_scheduler


def test_mxnet_env_not_is_current_host_scheduler(mxc, training):
    from mxnet_container.train import MXNetTrainingEnvironment

    _write_resource_config(training, 'b', ['a', 'b'])

    with patch('socket.gethostbyname') as patched:
        mxnet_env = MXNetTrainingEnvironment(training)
        assert not mxnet_env.current_host_scheduler


def test_train_with_no_kwargs_in_user_module(mxc):
    from mxnet_container import train
    with patch('container_support.download_s3_resource') as patched_download_s3_resource, \
            patch('container_support.untar_directory') as patched_untar_directory, \
            patch('socket.gethostbyname') as patched_gethostbyname, \
            patch('inspect.getargspec') as patched_getargspec, \
            patch('importlib.import_module', new_callable=train_no_kwargs_mock) as patched_import_module:
        patched_getargspec.return_value = getargspec_orig(NoKWArgsModule.train)

        train(optml())
        assert patched_import_module.return_value.train.called



def test_train_failing_script(mxc):
    from mxnet_container import train

    def raise_error(*args, **kwargs):
        raise ValueError("I failed")

    with patch('container_support.download_s3_resource') as patched_download_s3_resource, \
            patch('container_support.untar_directory') as patched_untar_directory, \
            patch('socket.gethostbyname') as patched_gethostbyname, \
            patch('inspect.getargspec') as patched_getargspec, \
            patch('importlib.import_module', new_callable=train_kwargs_mock) as patched_import_module:
        patched_getargspec.return_value = getargspec_orig(KWArgsModule.train)
        patched_import_module.return_value.train.side_effect = raise_error

        with pytest.raises(ValueError):
            train(optml())
        assert patched_import_module.return_value.train.called


def test_train(mxc):
    from mxnet_container import train

    with patch('container_support.download_s3_resource') as patched_download_s3_resource, \
            patch('container_support.untar_directory') as patched_untar_directory, \
            patch('subprocess.Popen') as patched_Popen, \
            patch('socket.gethostbyname'),\
            patch('inspect.getargspec') as patched_getargspec,\
            patch('importlib.import_module', new_callable=train_kwargs_mock) as patched_import_module:
        patched_getargspec.return_value = getargspec_orig(KWArgsModule.train)

        train(optml())
        assert patched_Popen.call_count == 3
        assert patched_import_module.return_value.train.called


def test_train_save_shape(mxc, training):
    with patch('socket.gethostbyname') as patched_gethostbyname:
        from mxnet_container.train import MXNetTrainingEnvironment
        env = MXNetTrainingEnvironment(training)
        mock_module = MagicMock()
        data_desc = MagicMock()
        data_desc.name = "elizabeth"
        data_desc.shape = [100, 200, 300, 400]
        mock_module.data_shapes = [data_desc]
        env.default_save(mock_module)
        with open(os.path.join(env.model_dir, 'model-shapes.json')) as f:
            read_data_shape = json.load(f)
            expected_data_shape = [{'shape': [100, 200, 300, 400], 'name': 'elizabeth'}]
            assert expected_data_shape == read_data_shape


def _write_config_file(path, filename, data):
    path = os.path.join(path, "input/config/%s" % filename)
    with open(path, 'w') as f:
        json.dump(data, f)


def _write_resource_config(path, current_host, hosts):
    _write_config_file(path, 'resourceconfig.json', {'current_host': current_host, 'hosts': hosts})


def _serialize_hyperparameters(hp):
    return {str(k): json.dumps(v) for (k, v) in hp.items()}
