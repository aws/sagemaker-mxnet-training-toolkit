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

import docker_utils
import utils


# The image should serve a MXNet model saved in the default format, even without a user-provided script.
def test_default_model_fn(docker_image, opt_ml, processor):
    resource_path = 'test/resources/default_handlers'
    for path in ['code', 'model']:
        utils.copy_resource(resource_path, opt_ml, path)

    input = [[1, 2]]

    with docker_utils.HostingContainer(image=docker_image, processor=processor,
                                       opt_ml=opt_ml, script_name='empty_module.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)
        assert '[[4.9999918937683105]]' == output
