import numpy as np
import jax.numpy as jnp
from jaxlie import SO3


def _load_ts_ns(path):
    return np.loadtxt(path, delimiter=',', skiprows=1,
                      usecols=(0,), dtype=np.int64)


def load_imu(imu_path):
    raw   = np.loadtxt(imu_path, delimiter=',', skiprows=1)
    ts_ns = _load_ts_ns(imu_path)
    gyro  = raw[:, 1:4].astype(np.float32)
    return ts_ns, jnp.array(gyro)


def load_groundtruth(gt_path):
    raw   = np.loadtxt(gt_path, delimiter=',', skiprows=1)
    ts_ns = _load_ts_ns(gt_path)
    quaternions = raw[:, 4:8].astype(np.float32)
    return ts_ns, jnp.array(quaternions)


def quat_to_rot(q):
    qw, qx, qy, qz = q
    return SO3(jnp.array([qw, qx, qy, qz])).as_matrix()


def align_timestamps(imu_ts_ns, gt_ts_ns, gt_quats):
    indices = np.searchsorted(gt_ts_ns, imu_ts_ns)
    indices = np.clip(indices, 0, len(gt_ts_ns) - 1)
    matched_quats = gt_quats[indices]
    R_gt = jnp.array([quat_to_rot(q) for q in matched_quats])
    return R_gt


def truncate(timestamps, gyro, R_gt, seconds=30, offset=0):
    cutoff_low  = timestamps[0] + offset
    cutoff_high = timestamps[0] + offset + seconds
    mask = (timestamps >= cutoff_low) & (timestamps <= cutoff_high)
    return timestamps[mask], gyro[mask], R_gt[mask]


def load_euroc(data_dir):
    imu_path = f"{data_dir}/imu0/data.csv"
    gt_path  = f"{data_dir}/state_groundtruth_estimate0/data.csv"

    imu_ts_ns, gyro    = load_imu(imu_path)
    gt_ts_ns, gt_quats = load_groundtruth(gt_path)

    R_gt = align_timestamps(imu_ts_ns, gt_ts_ns, gt_quats)

    timestamps = (imu_ts_ns - imu_ts_ns[0]).astype(np.float64) * 1e-9
    dt = float(np.mean(np.diff(imu_ts_ns))) * 1e-9

    print(f"Loaded {len(imu_ts_ns)} IMU samples at ~{1/dt:.1f} Hz")
    print(f"Duration: {timestamps[-1]:.1f} seconds")
    print(f"dt: {dt*1000:.3f} ms")

    return timestamps, gyro, R_gt, dt