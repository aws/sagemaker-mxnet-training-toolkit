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

import uuid

import mxnet as mx
import numpy as np


def train(hosts, **kwargs):
    """
    Tests that hosts can communicate via a Parameter Server KV Store.

    Each host pushes a matrix of all 1s to the kv-store. 

    Each host then pulls this matrix and asserts that the kv-store has aggregated
    the matrices received from each host.
    """
    shared_var = str(uuid.uuid4())

    kv = mx.kv.create('dist_sync')
    shape = (3, 3)

    # Init to -1
    kv.init(shared_var, mx.nd.ones(shape) * -1)

    # Push to contain a single value repeated across array equal
    # to this host's index in the host array
    kv.push(shared_var, mx.nd.ones(shape))

    # Pull aggregated matrix
    arr_out = mx.nd.zeros(shape)
    kv.pull(shared_var, out=arr_out)

    expected = mx.nd.full(shape, len(hosts)).asnumpy()
    received = arr_out.asnumpy()

    assert np.array_equal(expected, received), \
        "Received: {}, Expected: {}".format(received, expected)
