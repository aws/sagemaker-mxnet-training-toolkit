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

import json
import os

from sagemaker.mxnet.model import MXNetModel

import local_mode


# The image should support serving Gluon-created models.
def test_gluon_hosting(docker_image, sagemaker_local_session):
    resource_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'gluon_hosting')
    m = MXNetModel(os.path.join('file://', resource_path, 'model'), 'SageMakerRole',
                   os.path.join(resource_path, 'code', 'gluon.py'), image=docker_image)

    with open('test/resources/mnist_images/04.json', 'r') as f:
        input = json.load(f)

    with local_mode.lock():
        try:
            predictor = m.deploy(1, 'local')
            output = predictor.predict(input)
            assert [4.0] == output
        finally:
            predictor.delete_endpoint()
