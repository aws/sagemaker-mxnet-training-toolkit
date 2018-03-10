from __future__ import print_function

from sagemaker import fw_utils
from sagemaker.utils import sagemaker_timestamp
import docker_utils
import utils
import os
import json

def test_train_py_version(docker_image, sagemaker_session, py_version, opt_ml):
    resource_path = 'test/resources/py_version/code'

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                bucket=sagemaker_session.default_bucket(),
                                s3_key_prefix=sagemaker_timestamp(),
                                script='usermodule.py',
                                directory=resource_path)

    hp = _py_version_dict(py_version)

    utils.create_config_files('usermodule.py', s3_source_archive.s3_prefix, opt_ml, additional_hp=hp)
    os.makedirs(os.path.join(opt_ml, 'model'))
    docker_utils.train(docker_image, opt_ml)

    success_file = 'output/success'
    assert os.path.exists(os.path.join(opt_ml, success_file)), 'expected file not found: {}'.format(success_file)



# The image should use the model_fn and transform_fn defined in the user-provided script when serving.
def test_hosting_py_version(docker_image, py_version, opt_ml):
    resource_path = 'test/resources/py_version'
    utils.copy_resource(resource_path, opt_ml, 'code')

    input = json.dumps(_py_version_dict(py_version))

    with docker_utils.HostingContainer(image=docker_image,
                                       opt_ml=opt_ml, script_name='usermodule.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)


def _py_version_dict(py_version):
    maj_to_minor = {2: 7, # Need Python 2.7 for Python 2
                    3: 4} # Need Python 3.4 or above for Python 3

    return {'py_major_version': str(py_version),
            'py_minimum_minor_version': str(maj_to_minor[py_version])}

