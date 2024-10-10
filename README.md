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

![](web/Pilot_project.gif)

This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.





---

## Getting Started

#### Dependencies


Project was tested on:
- Ubuntu >=20
- python >= 3.8
- cuda >=11.7
  
Deploy:
  - built on ROS noetic
  - platforms: Jetson orin dev-kit, Jetson orin nano

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

#### Train

#### Deploy

---


## License


---


---



[↑ Return](#Top)

---
