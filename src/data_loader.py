import numpy as np
import jax.numpy as jnp

def load_imu(imu_path):
    """
    Load IMU data from EuRoC CSV.
    Returns:
        timestamps: (N,) array in seconds
        gyro: (N, 3) array of angular velocities in rad/s
    """
    data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
    timestamps = data[:, 0] * 1e-9  # nanoseconds -> seconds
    gyro = data[:, 1:4]             # wx, wy, wz
    return jnp.array(timestamps), jnp.array(gyro)


def load_groundtruth(gt_path):
    """
    Load ground truth data from EuRoC CSV.
    Returns:
        timestamps: (M,) array in seconds
        quaternions: (M, 4) array as [qw, qx, qy, qz]
    """
    data = np.loadtxt(gt_path, delimiter=',', skiprows=1)
    timestamps = data[:, 0] * 1e-9  # nanoseconds -> seconds
    # EuRoC stores quaternion as [qw, qx, qy, qz] in columns 4:8
    quaternions = data[:, 4:8]
    return jnp.array(timestamps), jnp.array(quaternions)


def quat_to_rot(q):
    """
    Convert a quaternion to a rotation matrix.
    q: (4,) array as [qw, qx, qy, qz]
    returns: (3, 3) rotation matrix
    """
    qw, qx, qy, qz = q
    R = jnp.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def align_timestamps(imu_times, gt_times, gt_quats):
    """
    For each IMU timestamp find the nearest ground truth timestamp.
    Returns:
        R_gt: (N, 3, 3) rotation matrices aligned to IMU timestamps
    """
    # For each IMU time find index of closest GT time
    indices = np.searchsorted(np.array(gt_times), np.array(imu_times))
    indices = np.clip(indices, 0, len(gt_times) - 1)
    
    # Convert all matched quaternions to rotation matrices
    matched_quats = gt_quats[indices]  # (N, 4)
    R_gt = jnp.array([quat_to_rot(q) for q in matched_quats])  # (N, 3, 3)
    return R_gt


def load_euroc(data_dir):
    """
    Main loader. Pass in the path to the mav0 folder.
    Returns:
        timestamps: (N,) IMU timestamps in seconds
        gyro: (N, 3) gyroscope readings in rad/s
        R_gt: (N, 3, 3) ground truth rotation matrices
        dt: scalar timestep in seconds
    """
    imu_path = f"{data_dir}/imu0/data.csv"
    gt_path  = f"{data_dir}/state_groundtruth_estimate0/data.csv"

    timestamps, gyro = load_imu(imu_path)
    gt_times, gt_quats = load_groundtruth(gt_path)
    R_gt = align_timestamps(timestamps, gt_times, gt_quats)

    dt = float(jnp.mean(jnp.diff(timestamps)))
    print(f"Loaded {len(timestamps)} IMU samples at ~{1/dt:.1f} Hz")
    print(f"Duration: {float(timestamps[-1] - timestamps[0]):.1f} seconds")
    print(f"dt: {dt*1000:.3f} ms")

    return timestamps, gyro, R_gt, dt