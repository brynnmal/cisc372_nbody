void compute();

__global__ void fillAccel(vector3* d_accel_sum);
__global__ void computeAccels(vector3** d_accel, vector3* d_hPos, double* d_mass);
__global__ void sumCols(vector3** d_accel, vector3* d_accel_sum);
__global__ void updatePos(vector3* d_accel_sum, vector3* d_hPos, vector3* d_hVel);