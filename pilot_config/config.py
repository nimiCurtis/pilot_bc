import os
import json
import yaml

def get_config_dir():
    return os.path.dirname(os.path.realpath(__file__))

def _get_default_config(file_path: str):
    """
    Load a configuration file in either JSON or YAML format.

    Args:
    file_path (str): The path to the configuration file.

    Returns:
    dict: The configuration loaded into a dictionary.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file format is not supported.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No configuration file found at {file_path}")

    # Determine the file format from the extension
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in ['.json', '.jsn']:
        # Load JSON file
        with open(file_path, 'r') as file:
            return json.load(file)
    elif file_extension.lower() in ['.yaml', '.yml']:
        # Load YAML file
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        raise ValueError("Unsupported file format: Please provide a .json or .yaml file")


def get_robot_config(robot_name: str):
    """
    Retrieve the configuration for a specific robot based on its name.

    Args:
    robot_name (str): The name of the robot for which to retrieve the configuration.

    Returns:
    dict: The configuration loaded into a dictionary, sourced from a YAML file named after the robot.

    Raises:
    FileNotFoundError: If the configuration file does not exist.
    ValueError: If the configuration file is not in a supported format.
    """
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "robot",
                               f"{robot_name}.yaml")
    return _get_default_config(file_path=config_path)

def get_recording_config(data_folder: str, trajectory_name: str):
    """
    Retrieve the configuration for a specific recording based on the trajectory name.

    Args:
    data_folder (str): The folder where the trajectory data is stored.
    trajectory_name (str): The name of the trajectory for which to retrieve the metadata.

    Returns:
    dict: The configuration loaded into a dictionary, sourced from a JSON file associated with the trajectory.

    Raises:
    FileNotFoundError: If the configuration file does not exist.
    ValueError: If the configuration file is not in a supported format.
    """
    config_path = os.path.join(data_folder,
                               trajectory_name,
                               "metadata.json")
    return _get_default_config(file_path=config_path)


# def _get_default_config(filename):
#     return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

# def get_digit_config_path():
#     return _get_default_config("config_digit.yml")

# def get_digit_shadow_config_path():
#     return _get_default_config("config_digit_shadow.yml")

# def get_omnitact_config_path():
#     return _get_default_config("config_omnitact.yml")
