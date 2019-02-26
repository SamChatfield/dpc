//
// Sam Chatfield
// 1559986
//
// Intel Core i5-6500
// GeForce GTX 960
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 1024

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR(err, msg) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

void h_block_scan(int n, int numBlocks, int blockSize, int *in, int *out)
{
	for (int b = 0; b < numBlocks; b++)
	{
		out[b * blockSize] = 0;
		for (int i = 1; i < blockSize; i++)
		{

			int elem = b * blockSize + i;
			if (elem < n)
			{
				out[elem] = in[elem-1] + out[elem-1];
			}
		}
	}
}

__global__ void d_block_scan(int n, int *in, int *out)
{
	__shared__ int temp[BLOCK_SIZE * 2];

	int thid = threadIdx.x;
	int offset = 1;

	// Load input into shared memory
	temp[2*thid] = in[2*thid];
	temp[2*thid+1] = in[2*thid+1];

	// Build sum in place up the tree
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * (2*thid+1) - 1;
			int bi = offset * (2*thid+2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Zero the last element
	if (thid == 0) { temp[n-1] = 0; };

	// Traverse down tree and build scan
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * (2*thid+1) - 1;
			int bi = offset * (2*thid+2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// Write results to global memory
	out[2*thid] = temp[2*thid];
	out[2*thid+1] = temp[2*thid+1];
}

void h_full_scan(int numElements, int *in, int *out)
{
	out[0] = 0;
	for (int i = 1; i < numElements; i++)
	{
		out[i] = in[i-1] + out[i-1];
	}
}

bool correct_results_block(int length, int blockSize, int *result, int *expected)
{
	return false;
}

bool correct_results_full(int length, int *result, int *expected)
{
	for (int i = 0; i < length; i++)
	{
		if (result[i] != expected[i])
		{
			printf("TEST FAILED at element %d: %d received, %d expected\n", i, result[i], expected[i]);
			return false;
		}
	}
	return true;
}

int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Create host timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// Create Device timer event objects
	cudaEvent_t start, stop;
	float d_msecs;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blockSize = 1024;
	int numElements = 2048;
	size_t size = numElements * sizeof(int);
	int numBlocks = 1 + ((numElements - 1) / blockSize);
	int numSegments = 1 + ((numElements - 1) / (blockSize * 2));

	// Create and initialise host input vector
	int *h_A = (int*) malloc(size);
	printf("Sum Scan of %d elements\n", numElements);

	for (int i = 0; i < numElements; i++)
	{
		h_A[i] = rand() % 10;
	}

	// Copy host input vector to device
	int *d_A = NULL;
	err = cudaMalloc((void**) &d_A, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector h_A to d_A");

	//
	// H_BLOCK_SCAN
	//

	// TEST CASE

	int hbsTestSize = 10;
	int h_T_hbs_in[] = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
	int h_T_hbs_exp[] = { 0, 1, 3, 6, 10, 0, 1, 3, 6, 10 };
	int testBlockSize = 5;
	int testBlocks = 1 + ((hbsTestSize - 1) / testBlockSize);
	int *h_T_hbs = (int*) malloc(sizeof(int) * hbsTestSize);

	sdkStartTimer(&timer);
	h_block_scan(hbsTestSize, testBlocks, testBlockSize, h_T_hbs_in, h_T_hbs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_BLOCK_SCAN] for test case in %.5fmSecs\n", h_msecs);

	bool resultCorrectHBS = correct_results_full(hbsTestSize, h_T_hbs, h_T_hbs_exp);
	if (resultCorrectHBS)
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);

	free(h_T_hbs);

	// ACTUAL CALCULATION

	int *h_B_hbs = (int*) malloc(size);

	sdkStartTimer(&timer);
	h_block_scan(numElements, numSegments, blockSize * 2, h_A, h_B_hbs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_BLOCK_SCAN] for %d elements in %.5fmSecs\n", numElements, h_msecs);

	//
	// D_BLOCK_SCAN (listing 2)
	//
	int *d_B = NULL;
	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	cudaEventRecord(start, 0);
	d_block_scan<<<numSegments, blockSize>>>(numElements, d_A, d_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[D_BLOCK_SCAN] Executed block scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

	// Verify result against result of h_block_scan
	int *h_B_dbs = (int*) malloc(size);
	err = cudaMemcpy(h_B_dbs, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dbs");
	if (correct_results_full(numElements, h_B_dbs, h_B_hbs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dbs);

	err = cudaFree(d_B);
	CUDA_ERROR(err, "Failed to free device vector d_B");

	//
	// H_FULL_SCAN
	//

	// TEST CASE (from paper)

	int hfsTestSize = 8;
	int h_T_hfs_in[] = { 3, 1, 7, 0, 4, 1, 6, 3 };
	int h_T_hfs_exp[] = { 0, 3, 4, 11, 11, 15, 16, 22 };
	int *h_T_hfs = (int*) malloc(sizeof(int) * hfsTestSize);

	sdkStartTimer(&timer);
	h_full_scan(hfsTestSize, h_T_hfs_in, h_T_hfs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_FULL_SCAN] for test case in %.5fmSecs\n", h_msecs);

	bool resultCorrectHFS = correct_results_full(hfsTestSize, h_T_hfs, h_T_hfs_exp);
	if (resultCorrectHFS)
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);

	free(h_T_hfs);

	// ACTUAL CALCULATION

	int *h_B_hfs = (int*) malloc(size);

	sdkStartTimer(&timer);
	h_full_scan(numElements, h_A, h_B_hfs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_FULL_SCAN] for %d elements in %.5fmSecs\n", numElements, h_msecs);

	//
	// TEARDOWN
	//

	// Free host memory
	free(h_B_hbs);
	free(h_B_hfs);
	free(h_A);

	// Clean up the Host timer
	sdkDeleteTimer(&timer);

	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	printf("Done\n");
	return EXIT_SUCCESS;
}
