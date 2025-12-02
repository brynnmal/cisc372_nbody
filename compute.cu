#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"
#include <stdio.h>

extern vector3 **d_accel, *d_hPos, *d_hVel, *d_accel_sum;
extern double *d_mass;

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {
	int numBlocksPerDim = (NUMENTITIES + 7) / 8;
	dim3 computeAccelsTPB(8, 8, 3);
	dim3 computeAccelsBlocks(numBlocksPerDim, numBlocksPerDim);

	computeAccels<<<computeAccelsBlocks, computeAccelsTPB>>>(d_accel, d_hPos, d_mass);

	int sumColsTPB = 32;
	dim3 sumColsBlocks(NUMENTITIES, 3);
	int sumColsSharedMem = sumColsTPB * sizeof(double) * 2;

	sumCols<<<sumColsBlocks, sumColsTPB, sumColsSharedMem>>>(d_accel, d_accel_sum);
    
    int updatePosBlockDim = (NUMENTITIES + 7) / 8;
	dim3 updatePosTPB(8, 3);

	updatePos<<<updatePosBlockDim, updatePosTPB>>>(d_accel_sum, d_hPos, d_hVel);

}

__global__ void computeAccels(vector3** d_accel, vector3* d_hPos, double* d_mass){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	__shared__ vector3 shDistance[8][8];
	if (i >= NUMENTITIES || j >= NUMENTITIES) return;
    if (i == j) {
		d_accel[i][j][k] = 0;
	} else {
		shDistance[threadIdx.x][threadIdx.y][k] = d_hPos[i][k] - d_hPos[j][k];

		__syncthreads();

		double magnitude_sq = shDistance[threadIdx.x][threadIdx.y][0] * shDistance[threadIdx.x][threadIdx.y][0] + shDistance[threadIdx.x][threadIdx.y][1] * shDistance[threadIdx.x][threadIdx.y][1] + shDistance[threadIdx.x][threadIdx.y][2] * shDistance[threadIdx.x][threadIdx.y][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;

		d_accel[i][j][k] = accelmag * shDistance[threadIdx.x][threadIdx.y][k] / magnitude;
	}
}

__global__ void sumCols(vector3** d_accel, vector3* d_accel_sum){
	int row = threadIdx.x;
	int col = blockIdx.x; 
	int dim = blockIdx.y;
	extern __shared__ double shArr[];
	__shared__ int offset;
	int blocksize = blockDim.x;
	int arrSize = NUMENTITIES;
	shArr[row] = row < arrSize ? d_accel[col][row][dim] : 0;
	if (row == 0) {
		offset = blocksize;
	}
	__syncthreads();
	while (offset < arrSize) {
		shArr[row+blocksize] = row+blocksize < arrSize ? d_accel[col][row+offset][dim] : 0;
		__syncthreads();
		if (row == 0) {
			offset += blocksize;
		}
		double sum = shArr[2*row] + shArr[2*row+1];
		__syncthreads();
		shArr[row] = sum;
	}
	__syncthreads();
	for (int stride = 1; stride < blocksize; stride *= 2) {
		int arrIdx = row*stride*2;
		if (arrIdx + stride < blocksize) {
			shArr[arrIdx] += shArr[arrIdx + stride];
		}
		__syncthreads();
	}
	if (row == 0) {
		d_accel_sum[col][dim] = shArr[0];
	}
}

__global__ void updatePos(vector3* accel_sum, vector3* hPos, vector3* hVel) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y;

	if (i >= NUMENTITIES) return;

	hVel[i][j] += accel_sum[i][j] * INTERVAL;
	hPos[i][j] += hVel[i][j] * INTERVAL;
}