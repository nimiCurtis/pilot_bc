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
        # self.filter.Q = np.eye(3) * target_position_variance
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