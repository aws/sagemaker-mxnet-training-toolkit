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
from __future__ import absolute_import

import logging
import os
import platform
import shutil
import tempfile

import boto3
import pytest
from sagemaker import LocalSession, Session
from sagemaker.mxnet import MXNet

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption('--docker-base-name', default='preprod-mxnet')
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default=MXNet.LATEST_VERSION)
    parser.addoption('--py-version', default='3', choices=['2', '3'])
    parser.addoption('--processor', default='cpu', choices=['gpu', 'cpu'])
    parser.addoption('--aws-id', default=None)
    parser.addoption('--instance-type', default=None)
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption('--tag', default=None)


@pytest.fixture(scope='session')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session')
def py_version(request):
    return int(request.config.getoption('--py-version'))


@pytest.fixture(scope='session')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session')
def aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(scope='session')
def tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-py{}'.format(framework_version, processor, py_version)
    return provided_tag if provided_tag is not None else default_tag


@pytest.fixture(scope='session')
def instance_type(request, processor):
    return request.config.getoption('--instance-type') or \
        'ml.c4.xlarge' if processor == 'cpu' else 'ml.p2.xlarge'


@pytest.fixture(scope='session')
def docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture(scope='session')
def ecr_image(aws_id, docker_base_name, tag, region):
    return '{}.dkr.ecr.{}.amazonaws.com/{}:{}'.format(aws_id, region, docker_base_name, tag)


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def sagemaker_local_session(region):
    return LocalSession(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def local_instance_type(processor):
    return 'local' if processor == 'cpu' else 'local_gpu'


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)
