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


def test_mnist_script_mode(docker_image, sagemaker_local_session):
    resource_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'mnist')
    script_path = os.path.join(resource_path, 'mnist_script_mode.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type='local', sagemaker_session=sagemaker_local_session,
               image_name=docker_image)

    train = 'file://{}'.format(os.path.join(resource_path, 'train'))
    test = 'file://{}'.format(os.path.join(resource_path, 'test'))
    mx.fit({'train': train, 'test': test})

    output_path = os.path.dirname(mx.create_model().model_data)
    for f in ['output/success', 'model/model-symbol.json', 'model/model-0000.params',
              'model/model-shapes.json']:
        assert os.path.exists(os.path.join(output_path, f)), 'expected file not found: {}'.format(f)
