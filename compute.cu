#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

vector3 *d_hPos = NULL;
vector3 *d_hVel = NULL;
double *d_mass = NULL;
vector3 *d_accels = NULL;

/*__global__ void computeAccelsKernel(vector3 *pos,
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
*/
__global__ void computeAccelsSharedKernel(vector3 *pos, double *mass, vector3 *accels, int N)
{
    extern __shared__ double sharedData[];
    vector3 *sharedPos = (vector3*)sharedData;
    double *sharedMass = (double*)&sharedPos[blockDim.x];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    for (int tile = 0; tile < gridDim.x; tile++) {
        int loadIdx = tile * blockDim.x + tx;
        if (loadIdx < N) {
            sharedPos[tx][0] = pos[loadIdx][0];
            sharedPos[tx][1] = pos[loadIdx][1];
            sharedPos[tx][2] = pos[loadIdx][2];
            sharedMass[tx] = mass[loadIdx];
        }
        __syncthreads();
        
        int i = by * blockDim.y + ty;
        int j = tile * blockDim.x + tx;
        
        if (i < N && j < N && tx < blockDim.x) {
            int idx = i * N + j;
            
            if (i == j) {
                accels[idx][0] = accels[idx][1] = accels[idx][2] = 0;
            } else {
                double dx = pos[i][0] - sharedPos[tx][0];
                double dy = pos[i][1] - sharedPos[tx][1];
                double dz = pos[i][2] - sharedPos[tx][2];
                
                double distSq = dx*dx + dy*dy + dz*dz;
                double dist = sqrt(distSq);
                
                double accelMag = -GRAV_CONSTANT * sharedMass[tx] / distSq;
                
                accels[idx][0] = accelMag * dx / dist;
                accels[idx][1] = accelMag * dy / dist;
                accels[idx][2] = accelMag * dz / dist;
            }
        }
        __syncthreads();
    }
}

__global__ void updateMotionKernel(vector3 *accels,
                                   vector3 *vel,
                                   vector3 *pos,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    vector3 accelSum = {0,0,0};

    for (int j = 0; j < N; j++) {
        int idx = i * N + j;
        accelSum[0] += accels[idx][0];
        accelSum[1] += accels[idx][1];
        accelSum[2] += accels[idx][2];
    }

    vel[i][0] += accelSum[0] * INTERVAL;
    vel[i][1] += accelSum[1] * INTERVAL;
    vel[i][2] += accelSum[2] * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

void compute()
{
    int N = NUMENTITIES;

    static int firstCall = 1;
    if (firstCall) {
        CUDA_CHECK(cudaMalloc((void**)&d_hPos, sizeof(vector3)*N));
        CUDA_CHECK(cudaMalloc((void**)&d_hVel, sizeof(vector3)*N));
        CUDA_CHECK(cudaMalloc((void**)&d_mass, sizeof(double)*N));
        CUDA_CHECK(cudaMalloc((void**)&d_accels, sizeof(vector3)*N*N));

        CUDA_CHECK(cudaMemcpy(d_hPos, hPos, sizeof(vector3)*N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hVel, hVel, sizeof(vector3)*N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass, sizeof(double)*N, cudaMemcpyHostToDevice));

        firstCall = 0;
    }

    dim3 blockSize(16,16);
    dim3 gridSize( (N+15)/16, (N+15)/16 );
    //computeAccelsKernel<<<gridSize, blockSize>>>(d_hPos, d_mass, d_accels, N);
    size_t sharedMemSize = (sizeof(vector3) * blockSize.x + sizeof(double) * blockSize.x);
    computeAccelsSharedKernel<<<gridSize, blockSize, sharedMemSize>>>(d_hPos, d_mass, d_accels, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); 

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    updateMotionKernel<<<blocks, threads>>>(d_accels, d_hVel, d_hPos, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hPos, d_hPos, sizeof(vector3)*N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel, d_hVel, sizeof(vector3)*N, cudaMemcpyDeviceToHost));
}

#ifdef __cplusplus
}
#endif