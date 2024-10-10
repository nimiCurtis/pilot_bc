<div align="center">
<h2>Pilot Behavior Cloning</h2>
<h3>A Method For Learning Tracking Skills From Human Demos</h3>


<img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="https://opensource.org/license/apache-2-0" />
<img src="https://img.shields.io/github/last-commit/nimiCurtis/pilot_bc?style&color=5D6D7E" alt="git-last-commit" />

</div>

---

- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
- [Usage](#usage)
    - [Data](#data)
    - [Train](#train)
    - [Deploy](#deploy)
- [License](#license)


---


## Overview

This repository contains the code and resources related to an ongoing research of learning visual navigation policy in autonomous person-following task based on supervised Imitation Learning framework.

This framework involves collecting visual data from human demonstrations and labeling it with the odometry of
the traversed trajectory. The resulting agent uses this visual context to generate action trajectories,
employing generative models, including diffusion policies and transformers in a goal-condition fashion. The system enables a mobile
robot to track a subject and to navigate
around obstacles. This approach has the potential to simplifies and scale data collection and facilitates deployment in new environments (no need in mapping process) and robots (data is robot agnostic),
by non-experts.



<p align="center">
  <img src="web/Pilot_project.gif" alt="animated"
  width="800px"
  height="400px"

</p>




This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.




---

## Getting Started

#### Dependencies


Project was tested on:
- ubuntu >=20
- python >= 3.8
- cuda >=11.7
- built on ROS noetic
  
Deploy:
  
  - platforms: unitree go2, turtlebot3
  - edge devices: Jetson orin dev-kit, Jetson orin nano

Additionally, the whole project-cycle (data collection till deploy) depends on following software:

- [zion_ros_interface](https://github.com/nimiCurtis/zion_zed_ros_interface) (iterface wrapper for zed cameras and bag recording system)
- [bagtool](https://github.com/nimiCurtis/bagtool) (bag processing and pilot_bc dataset structure creator)
- [waypoints_follower_control](https://github.com/nimiCurtis/waypoints_follower_control) (ROS wrapper for a simple waypoints reaching controller)


#### Installation

1. Clone repo:
```sh
git clone https://github.com/nimiCurtis/pilot_bc
```

2. Install the project:
```sh
cd pilot_bc && pip install -e .
```

---
## Usage

#### Data

In this project, data collection is performed by recording ROS bag files using a robot or manually with the appropriate system.   

If you have a folder containing ROS bag files, you can process and extract them into a desired dataset structure by adjusting the relevant settings in the ```pilot_train/config/process_bag_config.yaml``` file.
Once configured, you can run the following command to convert the bags to a dataset:

```bash
python pilot_train/bag2dataset.py
```

Dataset folder should be looked as follow:

```
├── <dataset folder name>
│   ├── <name_of_bag_batch1>
|   |    ├── raw_data
│   │       ├── raw_<topic1>.h5
│   │       ├── ...
│   │       └── raw_<topicN>.h5
|   |    ├── visual_data
|   |       ├── depth            
│   │           ├── 0.jpg
│   │           ├── ...
│   │           └── T.jpg
|   |       └── rgb          
│   │           ├── ...
│   │           └── T.jpg
│   │    ├── metadata.json
│   │    ├── robot_traj_data.json
│   │    ├── target_traj_data.json (when using the object detection topic)
│   │    └── traj_sample.mp4
│   ...
└── └── <name_of_bag_batchN>
         ├── raw_data
            ├── raw_<topic1>.h5
            ├── ...
            └── raw_<topicN>.h5
         ├── visual_data
             ├── depth            
                 ├── 0.jpg
                 ├── ...
                 └── T.jpg
             └── rgb          
                 ├── ...
                 └── T.jpg
         ├── metadata.json
         ├── robot_traj_data.json
         ├── target_traj_data.json (when using the object detection topic)
         └── traj_sample.mp4
```  

Once the dataset is prepared,split and calculate stats by:

```bash
python pilot_train/data/data_split_and_stats.py -r <dataset folder name> -i <dataset folder path>
```

Stats will be saved at ```pilot_config``` relevant files which will be used in trianing phase.


#### Train




#### Deploy

---


## License


---


---



[↑ Return](#Top)

---
