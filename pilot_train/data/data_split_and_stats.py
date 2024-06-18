import argparse
import os
import shutil
import random
import json
import numpy as np

from pilot_config.config import get_dataset_config, set_dataset_config, recursive_update

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
    robot_demo_time = 0
    robot_number_of_bags = 0
    for robot_name in folders_robots_mapping:
        robot_folder_names = folders_robots_mapping.get(robot_name)

        if len(robot_folder_names)>0:
            # Randomly shuffle the names of the folders
            random.shuffle(robot_folder_names)
            robot_number_of_bags = len(folder_names)
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
            max_lin_vel = []
            min_lin_vel = []
            mean_lin_vel = []
            std_lin_vel = []
            
            max_ang_vel = []
            min_ang_vel = []
            mean_ang_vel = []
            std_ang_vel = []

            time = []
            for folder_name in test_folder_names + train_folder_names:
                
                with open(os.path.join(args.data_dir,folder_name, "metadata.json"), 'r') as file:
                    metadata =  json.load(file)
                    stats = metadata["stats"]
                    ## Think about combine the dy velocity
                    max_lin_vel.append(stats["dx"]["max"])
                    min_lin_vel.append(stats["dx"]["min"])
                    mean_lin_vel.append(stats["dx"]["mean"])
                    std_lin_vel.append(stats["dx"]["std"])

                    max_ang_vel.append(stats["dyaw"]["max"])
                    min_ang_vel.append(stats["dyaw"]["min"])
                    mean_ang_vel.append(stats["dyaw"]["mean"])
                    std_ang_vel.append(stats["dyaw"]["std"])

                    # accumulate time
                    time.append(metadata["time"]) # in seconds

            tot_max_lin_vel = np.max(max_lin_vel)
            tot_min_lin_vel = np.min(min_lin_vel)
            tot_mean_lin_vel = np.mean(mean_lin_vel)
            tot_std_lin_vel = np.mean(std_lin_vel)
            
            tot_max_ang_vel = np.max(max_ang_vel)
            tot_min_ang_vel = np.min(min_ang_vel)
            tot_mean_ang_vel = np.mean(mean_ang_vel)
            tot_std_ang_vel = np.mean(std_ang_vel)
            
            robot_demo_time = np.sum(time) / 60 # in minutes
            dataset_config = recursive_update(d=dataset_config, u = {robot_name:   
                                            {"stats" : {"max_lin_vel": float(np.round(tot_max_lin_vel,4)),
                                            "min_lin_vel": float(np.round(tot_min_lin_vel,4)),
                                            "mean_lin_vel": float(np.round(tot_mean_lin_vel,4)),
                                            "std_lin_vel": float(np.round(tot_std_lin_vel,4)),
                                            
                                            "max_ang_vel": float(np.round(tot_max_ang_vel,4)),
                                            "min_ang_vel": float(np.round(tot_min_ang_vel,4)),
                                            "mean_ang_vel": float(np.round(tot_mean_ang_vel,4)),
                                            "std_ang_vel": float(np.round(tot_std_ang_vel,4))},
                                            
                                            "time": float(np.round(robot_demo_time,4)),
                                            "number_of_bags": robot_number_of_bags,
                                            }})
            
            set_dataset_config(dataset_name=args.dataset_name, config=dataset_config)
            print(f"Update {args.dataset_name} : {robot_name} stats")

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir", "-i", help="Directory containing the data", required=True
    )
    
    parser.add_argument(
        "--dataset-name", "-r", help="Name of the dataset",  required=True, 
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
