import numpy as np

from tracking.kalman_filter import KalmanFilter
from tracking.optical_flow import lucas_kanade_flow


def test_kalman_filter_predict_update():
    kf = KalmanFilter()
    # initial prediction should be origin
    assert np.allclose(kf.predict(), [0.0, 0.0])
    kf.update(np.array([1.0, 1.0]))
    # after update and a predict step, position should be close to the measurement
    pos = kf.predict()
    assert np.allclose(pos, [1.0, 1.0], atol=1e-6)


def test_lucas_kanade_flow_translation():
    prev = np.zeros((10, 10))
    prev[4:6, 4:6] = 1
    curr = np.zeros((10, 10))
    curr[4:6, 5:7] = 1  # shift right by one pixel

    points = [np.array([4.5, 4.5]), np.array([5.5, 4.5])]
    flows = lucas_kanade_flow(prev, curr, points)

    expected = [np.array([-0.26666667, 0.06666667]), np.array([-0.26666667, 0.06666667])]
    for f, e in zip(flows, expected):
        assert np.allclose(f, e)
