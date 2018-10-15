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

import os

from sagemaker.mxnet import MXNet

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'resources', 'mnist')
SCRIPT_PATH = os.path.join(RESOURCE_PATH, 'mnist_script_mode.py')

TRAIN_INPUT = 'file://{}'.format(os.path.join(RESOURCE_PATH, 'train'))
TEST_INPUT = 'file://{}'.format(os.path.join(RESOURCE_PATH, 'test'))


def test_mnist_training(docker_image, sagemaker_local_session):
    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=1,
               train_instance_type='local', sagemaker_session=sagemaker_local_session,
               image_name=docker_image)

    _train_and_assert_success(mx)


def test_distributed_mnist_training(docker_image, sagemaker_local_session):
    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=2,
               train_instance_type='local', sagemaker_session=sagemaker_local_session,
               image_name=docker_image)

    _train_and_assert_success(mx)


def _train_and_assert_success(estimator):
    estimator.fit({'train': TRAIN_INPUT, 'test': TEST_INPUT})

    output_path = os.path.dirname(estimator.create_model().model_data)
    for f in ['output/success', 'model/model-symbol.json', 'model/model-0000.params',
              'model/model-shapes.json']:
        assert os.path.exists(os.path.join(output_path, f)), 'expected file not found: {}'.format(f)
