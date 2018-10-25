# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

from mock import Mock, mock_open, patch

from sagemaker_mxnet_container import default_save


@patch('json.dump')
def test_save(json_dump):
    model_dir = 'foo/model'
    model = Mock()
    model.data_shapes = []

    with patch('six.moves.builtins.open', mock_open(read_data=Mock())):
        default_save.save(model_dir, model)

    model.symbol.save.assert_called_with(os.path.join(model_dir, 'model-symbol.json'))
    model.save_params.assert_called_with(os.path.join(model_dir, 'model-0000.params'))
    json_dump.assert_called_once
