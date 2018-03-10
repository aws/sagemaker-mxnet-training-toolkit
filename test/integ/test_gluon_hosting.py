import docker_utils
import utils
import json


# The image should support serving Gluon-created models.
def test_gluon_hosting(docker_image, opt_ml):
    resource_path = 'test/resources/gluon_hosting'
    for path in ['code', 'model']:
        utils.copy_resource(resource_path, opt_ml, path)

    with open('test/resources/mnist_images/04.json', 'r') as f:
        input = json.load(f)

    with docker_utils.HostingContainer(image=docker_image,
                                       opt_ml=opt_ml, script_name='gluon.py') as c:
        c.ping()
        output = c.invoke_endpoint(input)
        assert '[4.0]' == output
