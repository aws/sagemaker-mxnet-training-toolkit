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

import json

import docker_utils
import utils


# The image should support serving Gluon-created models.
def test_gluon_hosting(docker_image, opt_ml, processor):
    resource_path = 'test/resources/gluon_hosting'
    for path in ['code', 'model']:
        utils.copy_resource(resource_path, opt_ml, path)

    with open('test/resources/mnist_images/04.json', 'r') as f:
        input = json.load(f)

    with docker_utils.HostingContainer(image=docker_image, processor=processor,
                                       opt_ml=opt_ml, script_name='gluon.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)
        assert '[4.0]' == output
