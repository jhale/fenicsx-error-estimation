import numpy as np


def loglog_marker_pos(xs, y1s, y2s):
    """Calculate the position of a marker in a log-log plot"""
    marker_x = np.power(xs[0], 0.73)*np.power(xs[-1], 0.27)
    y1_y2_mid = [np.power(y1s[0]*y2s[0], 0.5), np.power(y1s[-1]*y2s[-1], 0.5)]
    marker_y = np.power(y1_y2_mid[0], 0.7)*np.power(y1_y2_mid[-1], 0.3)

    return marker_x, marker_y
