# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))

MODEL_SUCCESS_FILES = {
    'output': ['success'],
    'model': ['model-symbol.json', 'model-shapes.json', 'model-0000.params'],
}

# Workaround for the intermittent worker timeout errors
# TODO: find and solve the root cause of this issue
NUM_MODEL_SERVER_WORKERS = 2

# EI is currently only supported in the following regions
# regions were derived from https://aws.amazon.com/machine-learning/elastic-inference/pricing/
EI_SUPPORTED_REGIONS = [
    'us-east-1',
    'us-east-2',
    'us-west-2',
    'eu-west-1',
    'ap-northeast-1',
    'ap-northeast-2',
]
