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

import numpy
from sagemaker.mxnet import MXNet

import local_mode
from test.integration import MODEL_SUCCESS_FILES, RESOURCE_PATH

MNIST_PATH = os.path.join(RESOURCE_PATH, 'mnist')
SCRIPT_PATH = os.path.join(MNIST_PATH, 'mnist.py')

TRAIN_INPUT = 'file://{}'.format(os.path.join(MNIST_PATH, 'train'))
TEST_INPUT = 'file://{}'.format(os.path.join(MNIST_PATH, 'test'))


def test_mnist_training_and_serving(docker_image, sagemaker_local_session, local_instance_type):
    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=1,
               train_instance_type=local_instance_type, sagemaker_session=sagemaker_local_session,
               image_name=docker_image)

    _train_and_assert_success(mx)

    with local_mode.lock():
        try:
            predictor = mx.deploy(1, local_instance_type)
            data = numpy.zeros(shape=(1, 1, 28, 28))
            predictor.predict(data)
        finally:
            mx.delete_endpoint()


def test_distributed_mnist_training(docker_image, sagemaker_local_session, local_instance_type):
    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=2,
               train_instance_type=local_instance_type, sagemaker_session=sagemaker_local_session,
               image_name=docker_image)

    _train_and_assert_success(mx)


def _train_and_assert_success(estimator):
    estimator.fit({'train': TRAIN_INPUT, 'test': TEST_INPUT})

    output_path = os.path.dirname(estimator.create_model().model_data)
    for f in MODEL_SUCCESS_FILES:
        assert os.path.exists(os.path.join(output_path, f)), 'expected file not found: {}'.format(f)
