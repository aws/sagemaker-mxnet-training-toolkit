import docker_utils
import utils
import json


# The image should use the model_fn and transform_fn defined in the user-provided script when serving.
def test_hosting(docker_image, opt_ml):
    resource_path = 'test/resources/dummy_hosting'
    utils.copy_resource(resource_path, opt_ml, 'code')

    input = json.dumps({'some': 'json'})

    with docker_utils.HostingContainer(image=docker_image,
                                       opt_ml=opt_ml, script_name='dummy_hosting_module.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)
        assert input == output

