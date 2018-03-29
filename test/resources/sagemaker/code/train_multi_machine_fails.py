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

import logging
logger = logging.getLogger(__name__)


def train(current_host, hosts, **kwargs):
    """All hosts except one succeed."""
    hosts = sorted(hosts)
    my_index = hosts.index(current_host)
    if my_index == 0:
        raise Exception("Host zero is failing")
    logger.info("Not host zero, not failing")
