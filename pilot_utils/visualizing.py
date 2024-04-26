import numpy as np
from PIL import Image
import torch
import os
import wandb
from typing import List, Optional, Tuple
from pilot_config.config import get_robot_config
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List
import wandb
import yaml
import torch
import torch.nn as nn
from pilot_utils.utils import to_numpy


# # load data_config.yaml
# with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
#     data_config = yaml.safe_load(f)


VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


class Visualizer:

    def __init__(self,datasets_cfg, log_cfg) -> None:
        self.datasets_cfg = datasets_cfg
        self.use_wandb = log_cfg.wandb.run.enable
        log_folder_path = os.path.join(log_cfg.project_folder,
                                    log_cfg.run_name)
        self.log_folder = log_folder_path
        
    def visualize_dist_pred(
        self,
        batch_obs_images: np.ndarray,
        batch_goal_images: np.ndarray,
        batch_dist_preds: np.ndarray,
        batch_dist_labels: np.ndarray,
        eval_type: str,
        epoch: int,
        num_images_preds: int = 8,
        display: bool = False,
        rounding: int = 4,
        dist_error_threshold: float = 3.0,
    ):
        """
        Visualize the distance classification predictions and labels for an observation-goal image pair.

        Args:
            batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
            batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
            batch_dist_preds (np.ndarray): batch of distance predictions [batch_size]
            batch_dist_labels (np.ndarray): batch of distance labels [batch_size]
            eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
            epoch (int): current epoch number
            num_images_preds (int): number of images to visualize
            display (bool): whether to display the images
            rounding (int): number of decimal places to round the distance predictions and labels
            dist_error_threshold (float): distance error threshold for classifying the distance prediction as correct or incorrect (only used for visualization purposes)
        """
        visualize_path = os.path.join(
            self.log_folder,
            "visualize",
            eval_type,
            f"epoch{epoch}",
            "dist_classification",
        )
        if not os.path.isdir(visualize_path):
            os.makedirs(visualize_path)
        assert (
            len(batch_obs_images)
            == len(batch_goal_images)
            == len(batch_dist_preds)
            == len(batch_dist_labels)
        )
        batch_size = batch_obs_images.shape[0]
        wandb_list = []
        for i in range(min(batch_size, num_images_preds)):
            dist_pred = np.round(batch_dist_preds[i], rounding)
            dist_label = np.round(batch_dist_labels[i], rounding)
            obs_image = numpy_to_img(batch_obs_images[i])
            goal_image = numpy_to_img(batch_goal_images[i])

            save_path = None
            if self.log_folder is not None:
                save_path = os.path.join(visualize_path, f"{i}.png")
            text_color = "black"
            if abs(dist_pred - dist_label) > dist_error_threshold:
                text_color = "red"

            display_distance_pred(
                [obs_image, goal_image],
                ["Observation", "Goal"],
                dist_pred,
                dist_label,
                text_color,
                save_path,
                display,
            )
        #     if use_wandb:
        #         wandb_list.append(wandb.Image(save_path))
        # if use_wandb:
        #     wandb.log({f"{eval_type}_dist_prediction": wandb_list}, commit=False)
    
    
    def visualize_traj_pred(
            self,
            batch_obs_images: np.ndarray,
            batch_goal_images: np.ndarray,
            dataset_indices: np.ndarray,
            batch_goals: np.ndarray,
            batch_pred_waypoints: np.ndarray,
            batch_label_waypoints: np.ndarray,
            eval_type: str,
            normalized: bool,
            epoch: int,
            num_images_preds: int = 8,
            display: bool = False,
            
    ):
        """
        Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.

        Args:
            batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
            batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
            dataset_names: indices corresponding to the dataset name
            batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
            batch_pred_waypoints (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
            batch_label_waypoints (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
            eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
            normalized (bool): whether the waypoints are normalized
            epoch (int): current epoch number
            num_images_preds (int): number of images to visualize
            display (bool): whether to display the images
        """
        visualize_path = None
        if self.log_folder is not None:
            visualize_path = os.path.join(
                self.log_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
            )

        if not os.path.exists(visualize_path):
            os.makedirs(visualize_path)

        assert (
                len(batch_obs_images)
                == len(batch_goal_images)
                == len(batch_goals)
                == len(batch_pred_waypoints)
                == len(batch_label_waypoints)
        )

        dataset_names = self.datasets_cfg.robots
        dataset_names.sort()

        batch_size = batch_obs_images.shape[0]
        wandb_list = []
        for i in range(min(batch_size, num_images_preds)):
            obs_img = numpy_to_img(batch_obs_images[i])
            goal_img = numpy_to_img(batch_goal_images[i])
            dataset_name = dataset_names[int(dataset_indices[i])]
            robot_config = get_robot_config(robot_name=dataset_name)
            goal_pos = batch_goals[i]
            pred_waypoints = batch_pred_waypoints[i]
            label_waypoints = batch_label_waypoints[i]

            if normalized:
                pred_waypoints *= robot_config[dataset_name]["metric_waypoint_spacing"]
                label_waypoints *= robot_config[dataset_name]["metric_waypoint_spacing"]
                goal_pos *= robot_config[dataset_name]["metric_waypoint_spacing"]

            save_path = None
            if visualize_path is not None:
                save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

            self.compare_waypoints_pred_to_label(
                obs_img,
                goal_img,
                dataset_name,
                goal_pos,
                pred_waypoints,
                label_waypoints,
                save_path,
                display,
            )
            if self.use_wandb:
                wandb_list.append(wandb.Image(save_path))
        if self.use_wandb:
            wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)


    def compare_waypoints_pred_to_label(
            self,
            obs_img,
            goal_img,
            dataset_name: str,
            goal_pos: np.ndarray,
            pred_waypoints: np.ndarray,
            label_waypoints: np.ndarray,
            save_path: Optional[str] = None,
            display: Optional[bool] = False,
    ):
        """
        Compare predicted path with the gt path of waypoints using egocentric visualization.

        Args:
            obs_img: image of the observation
            goal_img: image of the goal
            dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
            goal_pos: goal position in the image
            pred_waypoints: predicted waypoints in the image
            label_waypoints: label waypoints in the image
            save_path: path to save the figure
            display: whether to display the figure
        """

        fig, ax = plt.subplots(1, 3)
        start_pos = np.array([0, 0])
        if len(pred_waypoints.shape) > 2:
            trajs = [*pred_waypoints, label_waypoints]
        else:
            trajs = [pred_waypoints, label_waypoints]
        plot_trajs_and_points(
            ax[0],
            trajs,
            [start_pos, goal_pos],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        self.plot_trajs_and_points_on_image(
            ax[1],
            obs_img,
            dataset_name,
            trajs,
            [start_pos, goal_pos],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        ax[2].imshow(goal_img)

        fig.set_size_inches(18.5, 10.5)
        ax[0].set_title(f"Action Prediction")
        ax[1].set_title(f"Observation")
        ax[2].set_title(f"Goal")

        if save_path is not None:
            fig.savefig(
                save_path,
                bbox_inches="tight",
            )

        if not display:
            plt.close(fig)


    def plot_trajs_and_points_on_image(
            self,
            ax: plt.Axes,
            img: np.ndarray,
            dataset_name: str,
            list_trajs: list,
            list_points: list,
            traj_colors: list = [CYAN, MAGENTA],
            point_colors: list = [RED, GREEN],
    ):
        """
        Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
        Args:
            ax: matplotlib axis
            img: image to plot
            dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
            list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
            list_points: list of points, each point is a numpy array of shape (2,)
            traj_colors: list of colors for trajectories
            point_colors: list of colors for points
        """
        assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
        assert len(list_points) <= len(point_colors), "Not enough colors for points"
        assert (
                dataset_name in self.datasets_cfg
        ), f"Dataset {dataset_name} not found in datasets_cfg"

        robot_config = get_robot_config(robot_name=dataset_name)
        
        ax.imshow(img)
        if (
                "camera_metrics" in robot_config[dataset_name]
                and "camera_height" in robot_config[dataset_name]["camera_metrics"]
                and "camera_matrix" in robot_config[dataset_name]["camera_metrics"]
                and "dist_coeffs" in robot_config[dataset_name]["camera_metrics"]
        ):
            camera_height = robot_config[dataset_name]["camera_metrics"]["camera_height"]
            camera_x_offset = robot_config[dataset_name]["camera_metrics"]["camera_x_offset"]

            fx = robot_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
            fy = robot_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
            cx = robot_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
            cy = robot_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
            camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

            k1 = robot_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
            k2 = robot_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
            p1 = robot_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
            p2 = robot_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
            k3 = robot_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
            dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

            for i, traj in enumerate(list_trajs):
                xy_coords = traj[:, :2]  # (horizon, 2)
                traj_pixels = get_pos_pixels(
                    xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
                )
                if len(traj_pixels.shape) == 2:
                    ax.plot(
                        traj_pixels[:250, 0],
                        traj_pixels[:250, 1],
                        color=traj_colors[i],
                        lw=2.5,
                    )

            for i, point in enumerate(list_points):
                if len(point.shape) == 1:
                    # add a dimension to the front of point
                    point = point[None, :2]
                else:
                    point = point[:, :2]
                pt_pixels = get_pos_pixels(
                    point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
                )
                ax.plot(
                    pt_pixels[:250, 0],
                    pt_pixels[:250, 1],
                    color=point_colors[i],
                    marker="o",
                    markersize=10.0,
                )
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
            ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


def plot_trajs_and_points(
        ax: plt.Axes,
        list_trajs: list,
        list_points: list,
        traj_colors: list = [CYAN, MAGENTA],
        point_colors: list = [RED, GREEN],
        traj_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot", "goal"],
        traj_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
            len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
            traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")

def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def gen_bearings_from_waypoints(
        waypoints: np.ndarray,
        mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
        xy: np.ndarray,
        camera_height: float,
        camera_x_offset: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
        points: np.ndarray,
        camera_height: float,
        camera_x_offset: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate
    Returns:
        camera matrix
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])



def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_close_images (np.ndarray): batch of close goal images [batch_size, height, width, channels]
        batch_far_images (np.ndarray): batch of far goal images [batch_size, height, width, channels]
        batch_close_preds (np.ndarray): batch of close predictions [batch_size]
        batch_far_preds (np.ndarray): batch of far predictions [batch_size]
        batch_close_labels (np.ndarray): batch of close labels [batch_size]
        batch_far_labels (np.ndarray): batch of far labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        close_image = numpy_to_img(batch_close_images[i])
        far_image = numpy_to_img(batch_far_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")

        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_pairwise_classification": wandb_list}, commit=False)


def display_distance_pred(
    imgs: list,
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)


def numpy_to_img(arr: np.ndarray) -> Image:
    if arr.shape[0] == 1:  # Check if the array has one channel a.k.a depth image
        arr = np.repeat(arr, 3, axis=0)  # Repeat the channel 3 times
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img
