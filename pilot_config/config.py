import os

def get_config_dir():
    return os.path.dirname(os.path.realpath(__file__))

# def _get_default_config(filename):
#     return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

# def get_digit_config_path():
#     return _get_default_config("config_digit.yml")

# def get_digit_shadow_config_path():
#     return _get_default_config("config_digit_shadow.yml")

# def get_omnitact_config_path():
#     return _get_default_config("config_omnitact.yml")

if __name__ == "__main__":
    a=1