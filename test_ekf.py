import numpy as np

imu = np.loadtxt("data/mav0_diff/imu0/data.csv", delimiter=',', skiprows=1)

for offset_sec in [0, 10, 20, 30, 40, 50]:
    start = offset_sec * 200   # 200 Hz
    end   = start + 6000       # 30 seconds
    if end > len(imu): break
    
    gyro  = imu[start:end, 1:4]
    accel = imu[start:end, 4:7]
    gyro_norm  = np.linalg.norm(gyro, axis=1)
    accel_norm = np.linalg.norm(accel, axis=1)
    dynamic = np.mean(np.abs(accel_norm - 9.81) > 0.5) * 100
    
    print(f"offset={offset_sec:3d}s | gyro mean={np.mean(gyro_norm):.3f} max={np.max(gyro_norm):.3f} | dynamic={dynamic:.1f}%")