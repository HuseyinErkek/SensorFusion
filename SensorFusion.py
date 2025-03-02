import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

def simulate_vehicle_motion(v, yaw_rate, dt, steps):
    x, y, psi = 0, 0, 0
    x_true, y_true = [x], [y]
    for _ in range(steps):
        x += v * np.cos(psi) * dt
        y += v * np.sin(psi) * dt
        psi += yaw_rate * dt
        x_true.append(x)
        y_true.append(y)
    return np.array(x_true), np.array(y_true)

def add_sensor_noise(x, y, imu_std, gps_std):
    return (x + np.random.normal(0, imu_std, x.shape),
            y + np.random.normal(0, imu_std, y.shape),
            x + np.random.normal(0, gps_std, x.shape),
            y + np.random.normal(0, gps_std, y.shape))

def add_sensor_bias(imu_x, imu_y, gps_x, gps_y, imu_bias, gps_bias):
    return (imu_x + imu_bias, imu_y + imu_bias,
            gps_x + gps_bias, gps_y + gps_bias)

def add_scale_factor(imu_x, imu_y, gps_x, gps_y, imu_scale_factor, gps_scale_factor):
    return (imu_x * imu_scale_factor, imu_y * imu_scale_factor,
            gps_x * gps_scale_factor, gps_y * gps_scale_factor)

def calibrate_bias(data):
    return data - np.mean(data)

def low_pass_filter(x_imu, y_imu, x_gps, y_gps, alpha=0.5):
    x_fused, y_fused = np.zeros_like(x_imu), np.zeros_like(y_imu)
    x_fused[0], y_fused[0] = (x_imu[0] + x_gps[0]) / 2, (y_imu[0] + y_gps[0]) / 2
    for i in range(1, len(x_imu)):
        x_fused[i] = alpha * x_fused[i - 1] + (1 - alpha) * ((x_imu[i] + x_gps[i]) / 2)
        y_fused[i] = alpha * y_fused[i - 1] + (1 - alpha) * ((y_imu[i] + y_gps[i]) / 2)
    return x_fused, y_fused

def apply_kalman_filter(x_fused, y_fused, dt):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x_fused[0], y_fused[0], 0, 0])
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P = np.diag([1, 1, 0.1, 0.1])
    kf.R = np.diag([0.05, 0.05])
    kf.Q = np.diag([0.001, 0.001, 0.01, 0.01])
    x_filtered, y_filtered = [], []
    for i in range(len(x_fused)):
        kf.predict()
        kf.update([x_fused[i], y_fused[i]])
        x_filtered.append(kf.x[0])
        y_filtered.append(kf.x[1])
    return np.array(x_filtered), np.array(y_filtered)

# Parametreler
dt, steps, v, yaw_rate = 0.3, 350, 2, 0.2
x, y = simulate_vehicle_motion(v, yaw_rate, dt, steps)
x_imu, y_imu, x_gps, y_gps = add_sensor_noise(x, y, 0.4, 0.2)
x_imu, y_imu, x_gps, y_gps = add_sensor_bias(x_imu, y_imu, x_gps, y_gps, 0.4, 0)
x_imu, y_imu, x_gps, y_gps = add_scale_factor(x_imu, y_imu, x_gps, y_gps, 1.2, 0.8)
x_imu, y_imu, x_gps, y_gps = calibrate_bias(x_imu), calibrate_bias(y_imu), calibrate_bias(x_gps), calibrate_bias(y_gps)
x_fused, y_fused = low_pass_filter(x_imu, y_imu, x_gps, y_gps, 0.5)
x_filtered, y_filtered = apply_kalman_filter(x_fused, y_fused, 0.5)

plt.figure(figsize=(12, 8))
plt.plot(x, y, 'r-', linewidth=2.5, label='Gerçek Rota')
plt.scatter(x_imu, y_imu, color='yellow', edgecolors='black', label='IMU Verisi')
plt.scatter(x_gps, y_gps, color='green', edgecolors='black', label='GPS Verisi')
plt.plot(x_filtered, y_filtered, 'b-', linewidth=2, label='Kalman Filtresi Sonucu')
plt.xlabel("X Konumu (metre)")
plt.ylabel("Y Konumu (metre)")
plt.title("Araç Takibi: Sensör Füzyonu ve Kalman Filtresi Sonucu")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()