import argparse
import os
import shutil
import random

from pilot_config.config import get_dataset_config

PATH = os.path.dirname(__file__)

def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def main(args: argparse.Namespace):
    
    # Get the names of the folders in the data directory that contain the file 'traj_robot_data.json'
    # TODO: change the collecting method
    folder_names = [
        f
        for f in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, f))
        and "traj_robot_data.json" in os.listdir(os.path.join(args.data_dir, f))
    ]

    # Assert that there is at least one folder
    assert len(folder_names) > 0, "No valid folders found in the specified data directory."
    
    # Create directories for the train and test sets
    train_dir = os.path.join(PATH,
                            args.data_splits_dir,
                            args.dataset_name, "train")
    test_dir = os.path.join(PATH,
                            args.data_splits_dir,
                            args.dataset_name, "test")

    dataset_config = get_dataset_config(args.dataset_name)
    robots_names = dataset_config.get('robots')

    assert len(robots_names) > 0, f"No robots names were given. Check {args.dataset_name} config"

    # Map folder name for robot name
    folders_robots_mapping = {}
    for robot_name in robots_names :
        folders_robots_mapping[robot_name] = []
    for folder_name in folder_names:
        robot_name = folder_name.split('_')[0]
        if robot_name in folders_robots_mapping:
            folders_robots_mapping[robot_name].append(folder_name)

    # For each robot in the dataset create dir and throw the pathes to the traj_names in the folders of the robot
    for robot_name in folders_robots_mapping:
        robot_folder_names = folders_robots_mapping.get(robot_name)

        # Randomly shuffle the names of the folders
        random.shuffle(robot_folder_names)

        # Split the names of the folders into train and test sets or use the same folder if only one exists
        if len(folder_names) == 1:
            train_folder_names = folder_names
            test_folder_names = folder_names
        else:
            split_index = int(args.split * len(robot_folder_names))
            train_folder_names = robot_folder_names[:split_index]
            test_folder_names = robot_folder_names[split_index:]

        robot_train_dir = os.path.join(train_dir,robot_name)
        robot_test_dir = os.path.join(test_dir,robot_name)
        
        for dir_path in [robot_train_dir, robot_test_dir]:
            if os.path.exists(dir_path):
                print(f"Clearing files from {dir_path} for new data split")
                remove_files_in_dir(dir_path)
            else:
                print(f"Creating {dir_path}")
                os.makedirs(dir_path)

        # Write the names of the train and test folders to files
        with open(os.path.join(robot_train_dir, "traj_names.txt"), "w") as f:
            for folder_name in train_folder_names:
                f.write(folder_name + "\n")

        with open(os.path.join(robot_test_dir, "traj_names.txt"), "w") as f:
            for folder_name in test_folder_names:
                f.write(folder_name + "\n")
    
    # TODO:
    # Take dataset stats and push to dataset config
    


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir", "-i", help="Directory containing the data", default="/home/roblab20/dev/pilot/pilot_bc/pilot_dataset/follow_in_broadcom_lab", required=False
    )
    
    parser.add_argument(
        "--dataset-name", "-r", help="Name of the dataset", default="follow_in_broadcom_lab",  required=False, 
    )
    
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="../dataset", help="Data splits directory"
    )
    args = parser.parse_args()
    main(args)
    print("DONE!")
