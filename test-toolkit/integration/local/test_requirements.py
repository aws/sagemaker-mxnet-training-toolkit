# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

from sagemaker.mxnet import MXNet

from integration import MODEL_SUCCESS_FILES, RESOURCE_PATH
from utils import local_mode_utils

SOURCE_PATH = os.path.join(RESOURCE_PATH, 'requirements')


def test_requirements_file(
    image_uri, sagemaker_local_session, local_instance_type, framework_version, tmpdir
):
    mx = MXNet(
        entry_point='entry.py',
        source_dir=SOURCE_PATH,
        role='SageMakerRole',
        train_instance_count=1,
        train_instance_type=local_instance_type,
        image_name=image_uri,
        framework_version=framework_version,
        output_path='file://{}'.format(tmpdir),
        sagemaker_session=sagemaker_local_session,
    )

    mx.fit()
    local_mode_utils.assert_output_files_exist(str(tmpdir), 'output', MODEL_SUCCESS_FILES['output'])
