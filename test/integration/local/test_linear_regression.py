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

import pytest
from sagemaker.fw_utils import tar_and_upload_dir
from sagemaker.mxnet import MXNet
from sagemaker.utils import sagemaker_timestamp

from test.integration import MODEL_SUCCESS_FILES, RESOURCE_PATH


@pytest.mark.skip(reason='waiting for default save before converting script')
def test_linear_regression(docker_image, sagemaker_local_session, local_instance_type):
    data_path = os.path.join(RESOURCE_PATH, 'linear_regression')

    s3_source_archive = tar_and_upload_dir(session=sagemaker_local_session.boto_session,
                                           bucket=sagemaker_local_session.default_bucket(),
                                           s3_key_prefix=sagemaker_timestamp(),
                                           script='linear_regression.py',
                                           directory=data_path)

    mx = MXNet(entry_point=os.path.join(data_path, 'linear_regression.py'), role='SageMakerRole',
               train_instance_count=1, train_instance_type=local_instance_type,
               sagemaker_session=sagemaker_local_session, image_name=docker_image)

    mx.fit(s3_source_archive.s3_prefix)

    output_path = os.path.dirname(mx.create_model().model_data)
    for f in MODEL_SUCCESS_FILES:
        assert os.path.exists(os.path.join(output_path, f)), 'expected file not found: {}'.format(f)
