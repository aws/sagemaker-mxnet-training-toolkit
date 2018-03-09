import docker_utils


# The image should have an appropriate version of python installed.
def test_py2(py_version, docker_image):

    maj_to_minor = {2: 7, # Need Python 2.7 for Python 2
                    3: 4} # Need Python 3.4 or above for Python 3
    expected_maj = int(py_version)
    expected_min_minor = maj_to_minor[expected_maj]

    with docker_utils.Container(docker_image) as c:
        c.execute_command([
            'python', '-c', '; '.join([
                'import sys',
                'maj, min = sys.version_info[0:2]',
                'print \'running python {}.{}\'.format(maj, min)',
                'assert maj == {} and min >= {}'.format(expected_maj, expected_min_minor)
            ])]
        )
