import numba as nb
import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter
from pilot_utils.utils import clip_angle
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
    
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.interpolate import interp1d

    # # Set up the figure and axis
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(0, 10)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("X Position")
    # ax.set_title("Trajectory Alignment Animation")

    # # Initialize the plot lines
    # line_last, = ax.plot([], [], 'o-', label="Last Trajectory x")
    # line_new, = ax.plot([], [], 'x-', label="New Trajectory x (Original)")
    # line_interp, = ax.plot([], [], 'd-', label="Interpolated New Trajectory x")
    # line_aligned, = ax.plot([], [], 's-', label="Aligned Trajectory x")

    # # Set up the legend
    # ax.legend()

    # # Parameters
    # time_first_initial = np.array([0, 1/7, 2/7, 3/7, 4/7])
    # first_trajectory_initial = np.array([
    #     [0.0, 0.0, 0.0],  # Example [x, y, yaw] values
    #     [1.0, 1.0, 0.1],
    #     [2.0, 2.0, 0.2],
    #     [3.0, 3.0, 0.3],
    #     [4.0, 4.0, 0.4]
    # ])

    # def generate_new_trajectory(t_start):
    #     """Generate a new trajectory starting at time t_start"""
    #     time_new = np.array([t_start + i/7 for i in range(5)])
    #     new_trajectory = np.array([
    #         [t_start * 10 + 0.5 + i, t_start * 10 + 0.5 + i, 0.45 + i * 0.1] for i in range(5)
    #     ])
    #     return time_new, new_trajectory

    # def interpolate_trajectory(time_first, time_second, second_trajectory):
    #     """Interpolate the second trajectory to the first trajectory's time points"""
    #     interpolated_second = np.zeros_like(first_trajectory_initial)

    #     for i in range(3):  # Iterate over x, y, yaw
    #         interp_func = interp1d(time_second, second_trajectory[:, i], kind='linear', fill_value="extrapolate")
    #         interpolated_second[:, i] = interp_func(time_first)

    #     return interpolated_second

    # def weighted_average_trajectory(first_trajectory, interpolated_second):
    #     """Compute the weighted average of the first and interpolated second trajectories"""
    #     weights_first = np.linspace(1, 0, len(first_trajectory))  # Decreasing weights for the first trajectory
    #     weights_second = np.linspace(0, 1, len(first_trajectory))  # Increasing weights for the second trajectory

    #     # Normalize weights
    #     weights_first = weights_first / (weights_first + weights_second)
    #     weights_second = weights_second / (weights_first + weights_second)

    #     aligned_trajectory = (weights_first[:, np.newaxis] * first_trajectory + 
    #                         weights_second[:, np.newaxis] * interpolated_second)
    #     return aligned_trajectory

    # # Initialization function for the animation
    # def init():
    #     line_last.set_data([], [])
    #     line_new.set_data([], [])
    #     line_interp.set_data([], [])
    #     line_aligned.set_data([], [])
    #     return line_last, line_new, line_interp, line_aligned

    # # Animation update function
    # def update(frame):
    #     global first_trajectory_initial, time_first_initial

    #     # Generate new trajectory
    #     t_start = frame * 1/5
    #     time_second, second_trajectory = generate_new_trajectory(t_start)

    #     # Interpolate new trajectory to first trajectory's time points
    #     interpolated_second = interpolate_trajectory(time_first_initial, time_second, second_trajectory)

    #     # Compute the aligned trajectory
    #     aligned_trajectory = weighted_average_trajectory(first_trajectory_initial, interpolated_second)

    #     # Update the plot lines with the new data
    #     line_last.set_data(time_first_initial, first_trajectory_initial[:, 0])
    #     line_new.set_data(time_second, second_trajectory[:, 0])
    #     line_interp.set_data(time_first_initial, interpolated_second[:, 0])
    #     line_aligned.set_data(time_first_initial, aligned_trajectory[:, 0])

    #     # Update the last trajectory to the aligned trajectory for the next iteration
    #     first_trajectory_initial = aligned_trajectory

    #     return line_last, line_new, line_interp, line_aligned

    # # Create the animation
    # ani = FuncAnimation(fig, update, frames=range(10), init_func=init, blit=True, interval=1000, repeat=False)

    # # Show the animation
    # plt.show()
    

    # Assume t0 = 0 for simplicity
    # First trajectory prediction (spanning ~0.71 seconds)
    # t0 = 0
    # time_first = np.array([t0, t0 + 1/7, t0 + 2/7, t0 + 3/7, t0 + 4/7])
    # first_trajectory = np.array([
    #     [0.0, 0.0, 0.0],  # Example [x, y, yaw] values
    #     [1.0, 1.0, 0.1],
    #     [2.0, 2.0, 0.2],
    #     [3.0, 3.0, 0.3],
    #     [4.0, 4.0, 0.4]
    # ])

    # # Second trajectory prediction (starting after 1/5 seconds)
    # time_second = np.array([t0 + 1/5, t0 + 1/5 + 1/7, t0 + 1/5 + 2/7, t0 + 1/5 + 3/7, t0 + 1/5 + 4/7])
    # second_trajectory = np.array([
    #     [2.5, 4.5, 0.45],  # Example [x, y, yaw] values
    #     [3.5, 6.5, 0.55],
    #     [2.5, 6.5, 0.65],
    #     [7.5, 7.5, 0.75],
    #     [10.5, 8.5, 0.85]
    # ])

    # # Interpolation of the second trajectory onto the first time steps
    # interpolated_second = np.zeros_like(first_trajectory)

    # for i in range(3):  # Iterate over x, y, yaw
    #     interp_func = interp1d(time_second, second_trajectory[:, i], kind='linear', fill_value="extrapolate")
    #     interpolated_second[:, i] = interp_func(time_first)

    # # Define weights for the first and second trajectories
    # weights_first = np.linspace(1, 0, len(time_first))  # Decreasing weights for the first trajectory
    # weights_second = np.linspace(0, 1, len(time_first))  # Increasing weights for the second trajectory

    # # Normalize weights so that their sum equals 1
    # weights_first = weights_first / (weights_first + weights_second)
    # weights_second = weights_second / (weights_first + weights_second)

    # # Apply the weighted average to combine the trajectories
    # aligned_trajectory = (weights_first[:, np.newaxis] * first_trajectory + 
    #                     weights_second[:, np.newaxis] * interpolated_second)

    # # Print the aligned trajectory
    # print("Aligned Trajectory:\n", aligned_trajectory)

    # # Plot for visualization
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_first, first_trajectory[:, 0], 'o-', label="First Trajectory x")
    # plt.plot(time_second, second_trajectory[:, 0], 'x-', label="Second Trajectory x (Original)")
    # plt.plot(time_first, interpolated_second[:, 0], 'd-', label="Interpolated Second Trajectory x")
    # plt.plot(time_first, aligned_trajectory[:, 0], 's-', label="Aligned Trajectory x")
    # plt.xlabel("Time (s)")
    # plt.ylabel("X Position")
    # plt.legend()
    # plt.title("Trajectory Alignment with Weighted Averaging")
    # plt.show()
    


    # Example data: Assuming t0 = 0 for simplicity
    time_first = np.array([0, 1/7, 2/7, 3/7, 4/7])
    first_trajectory = np.array([
        [0.0, 0.0, 0.0],  # Example [x, y, yaw] values
        [1.0, 1.0, 0.1],
        [2.0, 2.0, 0.2],
        [3.0, 3.0, 0.3],
        [4.0, 4.0, 0.4]
    ])

    time_second = np.array([1/5, 1/5 + 1/7, 1/5 + 2/7, 1/5 + 3/7, 1/5 + 4/7])
    second_trajectory = np.array([
        [5.5, 2.5, 0.45],  # Example [x, y, yaw] values
        [5.5, 5.5, 0.55],
        [8.5, 6.5, 0.65],
        [7.5, 7.5, 0.75],
        [8.5, 8.5, 0.85]
    ])

    # Interpolate the second trajectory to the first trajectory's time points
    interpolated_second = np.zeros_like(first_trajectory)

    for i in range(3):  # Iterate over x, y, yaw
        interp_func = interp1d(time_second, second_trajectory[:, i], kind='linear', fill_value="extrapolate")
        interpolated_second[:, i] = interp_func(time_first)

    # Define weights for the first and second trajectories
    weights_first = np.linspace(1, 0, len(time_first))  # Decreasing weights for the first trajectory
    weights_second = np.linspace(0, 1, len(time_first))  # Increasing weights for the second trajectory

    # Normalize weights so that their sum equals 1
    weights_first = weights_first / (weights_first + weights_second)
    weights_second = weights_second / (weights_first + weights_second)

    # Smoothen time factor
    smoothen_time = 1.  # You can adjust this value
    t_start = 0 #time_second[0]  # Assuming the transition starts at t0
    t = np.linspace(t_start, t_start + len(time_second) / 7, len(time_second))  # Time vector

    # Transition factor as a function of time, applying the smoothen time
    transition_factor = np.clip((t - t_start) / smoothen_time, 0, 1)

    # Adjust the weights using the smoothen time factor
    adjusted_weights_first = (1 - transition_factor)  #* weights_first
    adjusted_weights_second = transition_factor  #* weights_second

    # Apply the weighted average with smoothen time to combine the trajectories
    aligned_trajectory = (adjusted_weights_first[:, np.newaxis] * first_trajectory + 
                        adjusted_weights_second[:, np.newaxis] * interpolated_second)

    total_aligned_trajectory = np.concatenate([aligned_trajectory[time_first>t_start],second_trajectory[time_second>[time_first[-1]]]])
    total_time = np.concatenate([time_first[time_first>t_start],time_second[time_second>[time_first[-1]]]])
    
    
    # Plot for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(time_first, first_trajectory[:, 0], 'o-', label="First Trajectory x")
    plt.plot(time_second, second_trajectory[:, 0], 'x-', label="Second Trajectory x (Original)")
    plt.plot(time_first, interpolated_second[:, 0], '*', label="Interpolated Second Trajectory x")
    plt.plot(time_first, aligned_trajectory[:, 0], 's-', label="Aligned Trajectory x")
    plt.plot(total_time, total_aligned_trajectory[:, 0], 'b--', label="Final Trajectory x")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position")
    plt.title(f"Trajectory Smoothing with Smoothen Time = {smoothen_time}")
    plt.legend()
    plt.show()

