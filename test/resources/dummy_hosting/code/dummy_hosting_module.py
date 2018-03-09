

class DummyModel(object):
    def predict(self, data):
        return data


def model_fn(model_dir):
    return DummyModel()


def transform_fn(model, data, input_content_type, output_content_type):
    return data, "application/json"

