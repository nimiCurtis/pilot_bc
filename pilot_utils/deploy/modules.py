import numba as nb
import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter
from pilot_utils.utils import clip_angle
from typing import List
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
import logging

spec = [
    ("_window_size", nb.int64),
    ("_data_dim", nb.int64),
    ("_value_deque", nb.types.Array(nb.float64, 2, "C")),
    ("_sum", nb.float64[:]),
    ("_correction", nb.float64[:]),
]

@jitclass(spec=spec)  # type: ignore
class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.
    We implement the Neumaier's algorithm to calculate the moving window average,
    which is numerically stable.
    """

    def __init__(self, window_size: int, data_dim: int):
        """Initializes the class.

        Args:
        window_size: The moving window size.
        """
        assert window_size > 0
        self._window_size: int = window_size
        self._data_dim = data_dim
        # self._value_deque = collections.deque(maxlen=window_size)
        # Use numpy array to simulate deque so that it can be compiled by numba
        self._value_deque = np.zeros((self._data_dim, window_size), dtype=np.float64)
        # The moving window sum.
        self._sum = np.zeros((self._data_dim,), dtype=np.float64)
        # The correction term to compensate numerical precision loss during
        # calculation.
        self._correction = np.zeros((self._data_dim,), dtype=np.float64)

    def _neumaier_sum(self, value: npt.NDArray[np.float64]):
        """Update the moving window sum using Neumaier's algorithm.

        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

        Args:
        value: The new value to be added to the window.
        """
        assert value.shape == (self._data_dim,)
        new_sum = self._sum + value
        for k in range(self._data_dim):
            if abs(self._sum[k]) >= abs(value[k]):
                # If self._sum is bigger, low-order digits of value are lost.
                self._correction[k] += (self._sum[k] - new_sum[k]) + value[k]
            else:
                # low-order digits of sum are lost
                self._correction[k] += (value[k] - new_sum[k]) + self._sum[k]

        self._sum = new_sum

    def calculate_average(
        self, new_value: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes the moving window average in O(1) time.

        Args:
          new_value: The new value to enter the moving window.

        Returns:
          The average of the values in the window.

        """
        assert new_value.shape == (self._data_dim,)

        self._neumaier_sum(-self._value_deque[:, 0])
        self._neumaier_sum(new_value)

        # self._value_deque.append(new_value)
        for i in range(self._data_dim):
            self._value_deque[i, :] = np.roll(self._value_deque[i, :], -1)
        self._value_deque[:, -1] = new_value

        return (self._sum + self._correction) / self._window_size

@nb.jit(nopython=True, cache=True, parallel=True)
def inv_with_jit(M: npt.NDArray[np.float64]):
    return np.linalg.inv(M)

class GoalPositionEstimator:
    """Estimates base velocity of A1 robot.

    The velocity estimator consists of 2 parts:
    1) A state estimator for CoM velocity.

    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.

    2) A moving average filter to smooth out velocity readings
    """

    def __init__(
        self,
        sensor_variance=np.array([1e-2,1e-2,1e-1]),
        initial_variance=np.array([0.1,0.1,0.1]),
        moving_window_filter_size=15,   
        k=0.05
    ):
        """Initiates the velocity estimator.

        See filterpy documentation in the link below for more details.
        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

        Args:

        """

        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        # Motion Model
        self.filter.x = np.zeros(3)
        self._initial_variance = initial_variance
        self.filter.P = np.eye(3) * self._initial_variance  # State covariance

        self._prediction_variance = np.zeros(6)
        self.filter.F = np.eye(3)  # state transition matrix
        self.k = k # np.array([k,k,k]) # coefficient
        self.filter.B = np.eye(3) * self.k # type: ignore
        self.filter.inv = inv_with_jit  # type: ignore     To accelerate inverse calculation (~3x faster)

        # Sensor Model
        self._sensor_variance = sensor_variance
        self.filter.R = np.eye(3) * sensor_variance
        self.filter.H = np.eye(3)  # measurement function (y=H*x)

        self._window_size = moving_window_filter_size
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )
        self._estimated_goal = np.zeros(3)
        self.filter.inv(np.eye(3))

    def reset(self):
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )

    def update(
        self,
        state_error: npt.NDArray[np.float64],
        sensor_prediction:npt.NDArray[np.float64],
        prediction_variance: npt.NDArray[np.float64],
        prediction_mag: np.float64,
        correction_mag: np.float64,
    ):
        """Propagate current state estimate with new accelerometer reading."""

        # Get rotation matrix from quaternion
        self._prediction_variance = prediction_variance
        # Reconstructing the covariance matrix
        pred_cov_matrix = np.diag([self._prediction_variance[0],
                                self._prediction_variance[1],
                                self._prediction_variance[2]])
        
        self.filter.Q = pred_cov_matrix.dot(prediction_mag)
        self.filter.B = np.eye(3) * self.k * correction_mag
        self.filter.predict(u=state_error)
        # self.filter.x[-1] = clip_angle(self.filter.x[-1])
        if self.observed_model_prediction(sensor_prediction):
            self.filter.R = np.eye(3).dot(correction_mag) * self._sensor_variance 
            self.filter.update(sensor_prediction)
            # self.filter.x[-1] = clip_angle(self.filter.x[-1])


        self._estimated_goal = self.moving_window_filter.calculate_average(
            self.filter.x
        )
        
    def observed_model_prediction(self,model_prediction):
        return model_prediction is not None
    
    @property
    def estimated_goal(self):
        return self._estimated_goal.copy()



def slerp_wxyz(quat1, quat2, alpha):
    x, y, z, w = quat1
    start_rot = Rotation.from_quat([x, y, z, w])
    x, y, z, w = quat2
    end_rot = Rotation.from_quat([x, y, z, w])
    orientation_slerp = Slerp(
        times=[0, 1], rotations=Rotation.concatenate([start_rot, end_rot])
    )
    x, y, z, w = orientation_slerp([alpha])[0].as_quat()
    return np.array([x, y, z, w])

class RealtimeTraj:
    def __init__(self):
        self.translations = np.zeros((0, 3), dtype=np.float64)
        self.quaternions_xyzw = np.zeros((0, 4), dtype=np.float64)
        self.timestamps = np.zeros((0,), dtype=np.float64)  # in seconds

    def update(
        self,
        translations: npt.NDArray[np.float64],
        quaternions_xyzw: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        current_timestamp: float,
        adaptive_latency_matching: bool = False,
        smoothen_time: float = 0.0,
    ):
        assert (
            translations.shape[1] == 3
        ), f"Invalid shape {translations.shape[1]} for translations!"
        assert (
            quaternions_xyzw.shape[1] == 4
        ), f"Invalid shape {quaternions_xyzw.shape[1]} for quaternions_xyzw!"
        assert (
            len(timestamps.shape) == 1
        ), f"Invalid shape {timestamps.shape} for timestamps!"
        assert (
            translations.shape[0]
            == quaternions_xyzw.shape[0]
            == timestamps.shape[0]
        ), f"Input number inconsistent!"
        if len(timestamps) > 1 and np.any(timestamps[1:] - timestamps[:-1] <= 0):
            logging.warning(f"Input timestamps are not monotonically increasing!")

        if self.translations.shape[0] == 0:
            self.translations = np.array(translations)
            self.quaternions_xyzw = np.array(quaternions_xyzw)
            self.timestamps = np.array(timestamps)
        else:
            input_traj = RealtimeTraj()
            input_traj.update(
                translations=translations,
                quaternions_xyzw=quaternions_xyzw,
                timestamps=timestamps,
                current_timestamp=timestamps[0],
            )
            if adaptive_latency_matching:
                latency_precision = 0.02
                max_latency = 1.5
                min_latency = -0.0
                matching_dt = 0.05

                pose_samples = np.zeros((3, 3))
                for i in range(3):
                    t = self.interpolate_translation(
                        current_timestamp + (i - 1) * matching_dt
                    )
                    pose_samples[i, :3] = t
                errors = []
                error_weights = np.array(
                    [1, 1, 1]
                )  # x, y, z, qw, qx, qy, qz

                for latency in np.arange(min_latency, max_latency, latency_precision):
                    input_pose_samples = np.zeros((3, 3))
                    for i in range(3):
                        t = input_traj.interpolate_translation(
                            current_timestamp + latency + (i - 1) * matching_dt
                        )
                        input_pose_samples[i, :3] = t

                    error = np.sum(
                        np.abs(input_pose_samples - pose_samples) * error_weights
                    )
                    errors.append(error)
                errors = np.array(errors)
                best_latency = np.arange(min_latency, max_latency, latency_precision)[
                    np.argmin(errors)
                ]
                print(f"{best_latency=}")
                # input_traj.timestamps -= latency

            if smoothen_time > 0.0:
                for i in range(len(input_traj.timestamps)):
                    if input_traj.timestamps[i] <= current_timestamp:
                        t, q = self.interpolate(input_traj.timestamps[i])
                        input_traj.translations[i] = t
                        input_traj.quaternions_xyzw[i] = q
                    elif input_traj.timestamps[i] <= current_timestamp + smoothen_time:
                        alpha = (
                            input_traj.timestamps[i] - current_timestamp
                        ) / smoothen_time
                        t, q = self.interpolate(input_traj.timestamps[i])
                        input_traj.translations[i] = (
                            alpha * input_traj.translations[i] + (1 - alpha) * t
                        )
                        input_traj.quaternions_xyzw[i] = (
                            alpha * input_traj.quaternions_xyzw[i] + (1 - alpha) * q
                        )
                    else:
                        break

            # Find the last timestamp prior to the first timestamp of the input data
            idx = np.searchsorted(self.timestamps, input_traj.timestamps[0])
            # print(idx)
            # Remove all data after this timestamp
            self.translations = self.translations[:idx]
            self.quaternions_xyzw = self.quaternions_xyzw[:idx]
            self.timestamps = self.timestamps[:idx]

            self.translations = np.concatenate(
                [self.translations, input_traj.translations]
            )
            self.quaternions_xyzw = np.concatenate(
                [self.quaternions_xyzw, input_traj.quaternions_xyzw]
            )

            self.timestamps = np.concatenate([self.timestamps, input_traj.timestamps])

            assert np.all(
                self.timestamps[1:] - self.timestamps[:-1] > 0
            ), f"Timestamps are not monotonically increasing!"

        current_idx = np.searchsorted(self.timestamps, current_timestamp)
        # Only keep one data point before the current timestamp (for interpolation)
        if current_idx >= 2:
            self.translations = self.translations[current_idx - 1 :]
            self.quaternions_xyzw = self.quaternions_xyzw[current_idx - 1 :]
            self.timestamps = self.timestamps[current_idx - 1 :]
    
    def interpolate_translation(self, timestamp: float):
        if len(self.timestamps) == 0:
            raise ValueError("Trajectory not initialized")
        if timestamp <= self.timestamps[0]:
            return self.translations[0].copy()
        if timestamp >= self.timestamps[-1]:
            return self.translations[-1].copy()

        idx = np.searchsorted(self.timestamps, timestamp)
        alpha = (timestamp - self.timestamps[idx - 1]) / (
            self.timestamps[idx] - self.timestamps[idx - 1]
        )
        translation = (1 - alpha) * self.translations[
            idx - 1
        ] + alpha * self.translations[idx]
        
        return translation

    def interpolate(self, timestamp: float):
        if len(self.timestamps) == 0:
            raise ValueError("Trajectory not initialized")
        if timestamp <= self.timestamps[0]:
            return (
                self.translations[0].copy(),
                self.quaternions_xyzw[0].copy(),
            )
        if timestamp >= self.timestamps[-1]:
            return (
                self.translations[-1].copy(),
                self.quaternions_xyzw[-1].copy(),
            )

        idx = np.searchsorted(self.timestamps, timestamp)
        alpha = (timestamp - self.timestamps[idx - 1]) / (
            self.timestamps[idx] - self.timestamps[idx - 1]
        )
        translation = (1 - alpha) * self.translations[
            idx - 1
        ] + alpha * self.translations[idx]
        
        quaternion_wxyz = slerp_wxyz(self.quaternions_xyzw[idx-1], self.quaternions_xyzw[idx], alpha)


        return (
            translation,
            quaternion_wxyz,
        )

    def interpolate_traj(self, timestamps: List[float]):
        
        assert len(timestamps) >= 1, "Not enough timestamps"
        
        translations = []
        quaternions_xyzw = []

        for timestamp in timestamps:
            t, q = self.interpolate(timestamp)
            translations.append(t)
            quaternions_xyzw.append(q)
        
        return (
            np.stack(translations), # (N, 3)
            np.stack(quaternions_xyzw), # (N, 4)
        )












if __name__ == "__main__":
    
    # Example usage

    # Instantiate the RealtimeTraj class
    traj = RealtimeTraj()

    # Example: Initial trajectory data (simulating data at t=0 to t=2 seconds)
    translations1 = np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5], [1.8, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]])
    quaternions_xyzw1 = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Quaternion representing no rotation
        [0.707, 0.707, 0.0, 0.0],  # Quaternion representing 90 degrees rotation around X-axis
        [0.707, 0.0, 0.707, 0.0],  # Quaternion representing 90 degrees rotation around Y-axis
        [0.5, 0.5, 0.5, 0.5],  # Quaternion representing 90 degrees rotation around an arbitrary axis
        [0.0, 0.0, 0.0, 1.0],  # Quaternion representing 180 degrees rotation around Z-axis
    ])
    timestamps1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    # Update the trajectory with the initial data
    traj.update(translations=translations1, quaternions_xyzw=quaternions_xyzw1, timestamps=timestamps1, current_timestamp=2.0)

    # New trajectory data (simulating data at t=2 to t=3 seconds)
    translations2 = np.array([[1.0, 2.0, 2.0], [2.5, 2.5, 2.5], [3.0, 3.0, 3.0]])
    quaternions_xyzw2 = np.array([
        [0.0, 0.0, 0.0, 1.0],  # Continuation of 180 degrees rotation around Z-axis
        [0.707, 0.0, 0.707, 0.0],  # Another 90 degrees rotation around Y-axis
        [1.0, 0.0, 0.0, 0.0],  # Quaternion representing no rotation
    ])
    timestamps2 = np.array([1.0, 2.0, 3.0])

    # Update the trajectory with the new data, enabling smoothing
    traj.update(
        translations=translations2,
        quaternions_xyzw=quaternions_xyzw2,
        timestamps=timestamps2,
        current_timestamp=1.1,
        smoothen_time=1.5  # Smooth transition over 1 second
    )

    plt.plot(timestamps1, translations1[:,0],'x-', label="Translation 1")
    plt.plot(timestamps2, translations2[:,0],'x-', label="Translation 2")

    # Plot the updated trajectory
    plt.plot(traj.timestamps, traj.translations[:, 0], 'x-', label="Updated X Translation")

    # Interpolate the trajectory at specific timestamps (e.g., for t=0.1, 0.9, 2.5 seconds)
    interpolated_timestamps = [0.1, 0.9, 2.5]
    interpolated_translations, _= traj.interpolate_traj(interpolated_timestamps)

    # Print interpolated X translations
    for t, x in zip(interpolated_timestamps, interpolated_translations[:, 0]):
        print(f"Interpolated X Translation at t={t}: {x}")

    # Plot the interpolated points
    plt.plot(interpolated_timestamps, interpolated_translations[:, 0], 's', label="Interpolated X Translation")

    # Set plot labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("X Translation")
    plt.title("X Translation Over Time with Updates and Interpolations")
    plt.legend()

    # Display the plot
    plt.show()