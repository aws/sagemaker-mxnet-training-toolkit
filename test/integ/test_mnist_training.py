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
from __future__ import print_function

import os

import local_mode


def test_mnist_script_mode(docker_image, sagemaker_session, opt_ml, processor):
    resource_path = 'test/resources/mnist'
    script_path = os.path.join(resource_path, 'mnist_script_mode.py')

    local_mode.train(script_path, resource_path, docker_image, opt_ml)

    for f in ['output/success', 'model/model-symbol.json', 'model/model-0000.params',
              'model/model-shapes.json']:
        assert local_mode.file_exists(opt_ml, f), 'expected file not found: {}'.format(f)
