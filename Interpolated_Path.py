import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def generate_reference_track():
    raw_waypoints = np.array([
        [0, 0], [10, 0], [20, 5], [30, 10], [40, 15], 
        [50, 15], [60, 10], [70, 0], [80, 5], [90, 8], [105, 8]
    ])
    x, y = raw_waypoints[:, 0], raw_waypoints[:, 1]
    tck, u = splprep([x, y], s=0)
    u_fine = np.linspace(0, 1, 200)
    x_smooth, y_smooth = splev(u_fine, tck)
    return np.column_stack((x_smooth, y_smooth)), raw_waypoints

# Generate tracks
smooth_track, raw_waypoints = generate_reference_track()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(smooth_track[:, 0], smooth_track[:, 1], 'r-', label='Interpolated Path')
plt.plot(raw_waypoints[:, 0], raw_waypoints[:, 1], 'bo--', label='Raw Waypoints')
plt.scatter(raw_waypoints[:, 0], raw_waypoints[:, 1], color='blue', s=50)
plt.title("Smooth Reference Track using B-Spline Interpolation")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
