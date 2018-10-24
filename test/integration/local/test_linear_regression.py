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

from test.integration import MODEL_SUCCESS_FILES, RESOURCE_PATH


def test_linear_regression(docker_image, sagemaker_local_session, local_instance_type):
    lr_path = os.path.join(RESOURCE_PATH, 'linear_regression')

    mx = MXNet(entry_point=os.path.join(lr_path, 'linear_regression.py'), role='SageMakerRole',
               train_instance_count=1, train_instance_type=local_instance_type,
               sagemaker_session=sagemaker_local_session, image_name=docker_image)

    data_path = os.path.join(lr_path, 'data')
    s3_prefix = 'integ-test-data/mxnet-linear-regression'
    train_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'training'),
                                                      key_prefix=s3_prefix)
    eval_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'evaluation'),
                                                     key_prefix=s3_prefix)

    mx.fit({'training': train_input, 'evaluation': eval_input})

    output_path = os.path.dirname(mx.create_model().model_data)
    for f in MODEL_SUCCESS_FILES:
        assert os.path.exists(os.path.join(output_path, f)), 'expected file not found: {}'.format(f)
