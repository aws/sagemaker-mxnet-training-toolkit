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

MODEL_SUCCESS_FILES = [
    os.path.join('output', 'success'),
    os.path.join('model', 'model-symbol.json'),
    os.path.join('model', 'model-shapes.json'),
    os.path.join('model', 'model-0000.params'),
]

# Workaround for the intermittent worker timeout errors
NUM_MODEL_SERVER_WORKERS = 4
