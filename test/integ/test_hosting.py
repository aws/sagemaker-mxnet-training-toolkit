#  Copyright <YEAR> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import docker_utils
import utils
import json


# The image should use the model_fn and transform_fn defined in the user-provided script when serving.
def test_hosting(docker_image, opt_ml, processor):
    resource_path = 'test/resources/dummy_hosting'
    utils.copy_resource(resource_path, opt_ml, 'code')

    input = json.dumps({'some': 'json'})

    with docker_utils.HostingContainer(image=docker_image, processor=processor,
                                       opt_ml=opt_ml, script_name='dummy_hosting_module.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)
        assert input == output

