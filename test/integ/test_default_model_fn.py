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
