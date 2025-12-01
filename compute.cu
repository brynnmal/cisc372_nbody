#include <cuda_runtime.h>
#include <math.h>
#include "vector.h"
#include "config.h"

// GPU pointers (declared in vector.h as extern)
extern vector3 *d_hPos;
extern vector3 *d_hVel;
extern double *mass;

// Host pointers (already exist in nbody.c)
extern vector3 *hPos;
extern vector3 *hVel;

// Device copy of mass (must be allocated once)
double *d_mass;

// Device acceleration matrix
vector3 *d_accels;

// ---------------------------------------------
// Kernel 1: Compute pairwise accelerations
// ---------------------------------------------
__global__ void computeAccelsKernel(vector3 *pos,
                                   double *mass,
                                   vector3 *accels,
                                   int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    int idx = i * N + j;

    if (i == j) {
        accels[idx][0] = 0;
        accels[idx][1] = 0;
        accels[idx][2] = 0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double distSq = dx*dx + dy*dy + dz*dz;
    double dist   = sqrt(distSq);

    double accelMag = -GRAV_CONSTANT * mass[j] / distSq;

    accels[idx][0] = accelMag * dx / dist;
    accels[idx][1] = accelMag * dy / dist;
    accels[idx][2] = accelMag * dz / dist;
}

// ---------------------------------------------
// Kernel 2: Sum accelerations and update motion
// ---------------------------------------------
__global__ void updateMotionKernel(vector3 *accels,
                                   vector3 *vel,
                                   vector3 *pos,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    vector3 accelSum = {0,0,0};

    // Sum row i
    for (int j = 0; j < N; j++) {
        int idx = i * N + j;
        accelSum[0] += accels[idx][0];
        accelSum[1] += accels[idx][1];
        accelSum[2] += accels[idx][2];
    }

    // Update velocity + position
    vel[i][0] += accelSum[0] * INTERVAL;
    vel[i][1] += accelSum[1] * INTERVAL;
    vel[i][2] += accelSum[2] * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

// ---------------------------------------------
// compute() called every timestep by nbody.c
// ---------------------------------------------
void compute()
{
    int N = NUMENTITIES;

    // Allocate GPU memory on first call only
    static int firstCall = 1;
    if (firstCall) {
        cudaMalloc((void**)&d_hPos, sizeof(vector3)*N);
        cudaMalloc((void**)&d_hVel, sizeof(vector3)*N);
        cudaMalloc((void**)&d_mass, sizeof(double)*N);
        cudaMalloc((void**)&d_accels, sizeof(vector3)*N*N);

        cudaMemcpy(d_hPos, hPos, sizeof(vector3)*N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hVel, hVel, sizeof(vector3)*N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, mass, sizeof(double)*N, cudaMemcpyHostToDevice);

        firstCall = 0;
    }

    // ---- Launch Kernel 1 ----
    dim3 blockSize(16,16);
    dim3 gridSize( (N+15)/16, (N+15)/16 );
    computeAccelsKernel<<<gridSize, blockSize>>>(d_hPos, d_mass, d_accels, N);

    // ---- Launch Kernel 2 ----
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    updateMotionKernel<<<blocks, threads>>>(d_accels, d_hVel, d_hPos, N);

    // Copy updated positions and velocities back to host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
}

