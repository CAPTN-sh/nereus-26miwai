import numpy as np


def get_abs_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the absolute bearing from (lat1, lon1) to (lat2, lon2).
    Returns bearing in degrees from North (0° to 360°).
    """
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

    bearing_rad = np.arctan2(x, y)
    bearing_deg = (np.rad2deg(bearing_rad) + 360) % 360

    return bearing_deg


def get_rel_bearing(lat1, lon1, heading1, lat2, lon2, heading2=None):
    """Get the relative bearing between two ships with heading in degreen

    Args:
        lat1: Geographical latitude in degree
        lon1: Geographical longitude in degree
        heading1: Heading of ship 1 in degrees
        lat2: Geographical latitude in degree
        lon2: Geographical longitude in degree
        heading2: Heading of ship 2 in degrees

    Returns:
        rel_bearing: Bearing angle in degrees from ship 1 to 2
    """
    abs_bearing = get_abs_bearing(lat1, lon1, lat2, lon2)
    rel_bearing = (abs_bearing - heading1 + 360) % 360

    return rel_bearing


def get_rel_speed(speed1, course1, speed2, course2):
    """Calculate the relative speed between two ships considering their heading

    Args:
        speed1 (float): Speed of ship one
        course1 (float): Course over ground of ship 1
        speed2 (float): Speed of ship 2
        course2 (float): Course over ground of ship 1

    Returns:
        float: relative speed between ship one and two
    """
    x1, y1 = _to_vector(speed1, course1)
    x2, y2 = _to_vector(speed2, course2)

    dx = x1 - x2
    dy = y1 - y2
    rel_speed = np.sqrt(dx**2 + dy**2)

    return rel_speed


def _to_vector(speed, course):
    course_rad = np.deg2rad(90 - course)
    x = speed * np.cos(course_rad)
    y = speed * np.sin(course_rad)
    return x, y
