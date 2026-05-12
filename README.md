# EKF vs. Shampoo-Powered Batch Smoother for Attitude Estimation
Attitude estimation from IMU data using EKF and batch optimization on SO(3). The batch method solves a nonlinear least squares problem over gyroscope data with the Shampoo optimizer. Performance is evaluated on the EuRoC MAV dataset using geodesic attitude error.

**Course:** CS5757 Optimization Methods for Robotics, Cornell University, Spring 2026  


📄 [View Full Report (PDF)](CS_5757_State_Estimation_Project__Final.pdf)

## Results

### Geodesic Attitude Error
![Fig 1](results/error_grid.png)

### EKF Covariance Consistency (easy)
![Fig 2](results/sigma_easy.png)

### EKF Covariance Consistency (difficult)
![Fig 3](results/sigma_diff.png)

### Convergence and runtime
![Fig 4](results/cost_comparison.png)

### Runtime Comparison
![Fig 5](results/runtime_bars.png)

