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
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

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

__global__ void d_block_scan(int n, int *in, int *out, int *sums)
{
	__shared__ int temp[BLOCK_SIZE * 2];

	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;
	int offset = 1;

	// Load input into shared memory
	if (segStartIdx + 2*thid < n)
	{
		temp[2*thid] = in[segStartIdx + 2*thid];
		temp[2*thid+1] = in[segStartIdx + 2*thid+1];
	}
	else
	{
		temp[2*thid] = 0;
		temp[2*thid+1] = 0;
	}

	// Build sum in place up the tree
	for (int d = segSize >> 1; d > 0; d >>= 1)
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
	if (thid == 0) {
		// Move total sum to sums array
		sums[blockIdx.x] = temp[segSize-1];
		// Zero the last element
		temp[segSize-1] = 0;
	};

	// Traverse down tree and build scan
	for (int d = 1; d < segSize; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * (2*thid+1) - 1;
			int bi = offset * (2*thid+2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// Write results to global memory
	if (segStartIdx + 2*thid < n)
	{
		out[segStartIdx + 2*thid] = temp[2*thid];
		out[segStartIdx + 2*thid+1] = temp[2*thid+1];
	}
}

__global__ void d_block_scan_bcao(int n, int *in, int *out, int *sums)
{
	// TODO: Verify that +64 is the min amount we can add for it to work
	__shared__ int temp[BLOCK_SIZE * 2 + 64];

	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + blockDim.x;

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	/*
	// Load input into shared memory
	if (segStartIdx + 2*thid < n)
	{
		// TODO: BCAO change
//		temp[2*thid] = in[segStartIdx + 2*thid];
//		temp[2*thid+1] = in[segStartIdx + 2*thid+1];
		temp[ai + bankOffsetA] = in[segStartIdx + ai];
		temp[bi + bankOffsetB] = in[segStartIdx + bi];
	}
	else
	{
		// TODO: BCAO change
//		temp[2*thid] = 0;
//		temp[2*thid+1] = 0;
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}
	*/

	if (segStartIdx + ai < n) temp[ai + bankOffsetA] = in[segStartIdx + ai];
	else temp[ai + bankOffsetA] = 0;

	if (segStartIdx + bi < n) temp[bi + bankOffsetB] = in[segStartIdx + bi];
	else temp[bi + bankOffsetB] = 0;

	// Build sum in place up the tree
	for (int d = segSize >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * (2*thid+1) - 1;
			int bi = offset * (2*thid+2) - 1;
			// TODO: BCAO addition
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Zero the last element
	if (thid == 0) {
		// Move total sum to sums array
		// TODO: BCAO change
//		sums[blockIdx.x] = temp[segSize-1];
		sums[blockIdx.x] = temp[segSize-1 + CONFLICT_FREE_OFFSET(segSize-1)];
		// Zero the last element
		// TODO: BCAO change
//		temp[segSize-1] = 0;
		temp[segSize-1 + CONFLICT_FREE_OFFSET(segSize-1)] = 0;
	};

	// Traverse down tree and build scan
	for (int d = 1; d < segSize; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * (2*thid+1) - 1;
			int bi = offset * (2*thid+2) - 1;
			// TODO: BCAO addition
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	/*
	// Write results to global memory
	if (segStartIdx + 2*thid < n)
	{
		// TODO: BCAO change
//		out[segStartIdx + 2*thid] = temp[2*thid];
//		out[segStartIdx + 2*thid+1] = temp[2*thid+1];
		out[segStartIdx + ai] = temp[ai + bankOffsetA];
		out[segStartIdx + bi] = temp[bi + bankOffsetB];
	}
	*/

	if (segStartIdx + ai < n) out[segStartIdx + ai] = temp[ai + bankOffsetA];
	if (segStartIdx + bi < n) out[segStartIdx + bi] = temp[bi + bankOffsetB];
}

void h_full_scan(int numElements, int *in, int *out)
{
	out[0] = 0;
	for (int i = 1; i < numElements; i++)
	{
		out[i] = in[i-1] + out[i-1];
	}
}

__global__ void d_uniform_add(int n, int *incr, int *out)
{
	// TODO: Try storing one value of incr in shared mem

	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;

	if (segStartIdx + 2*thid < n)
	{
		out[segStartIdx + 2*thid] += incr[blockIdx.x];
		out[segStartIdx + 2*thid+1] += incr[blockIdx.x];
	}
}

void one_level_scan(int n, int numSegments, int segSize, int *in, int *out)
{
	cudaError_t err = cudaSuccess;

	size_t sumsSize = numSegments * sizeof(int);
	int *sums = NULL;
	err = cudaMalloc((void**) &sums, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sums");

	d_block_scan<<<numSegments, segSize/2>>>(n, in, out, sums);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	err = cudaFree(sums);
	CUDA_ERROR(err, "Failed to free device vector sums");
}

void two_level_scan(int n, int numSegments, int segSize, int *in, int *out)
{
	cudaError_t err = cudaSuccess;

	size_t sumsSize = numSegments * sizeof(int);
	int *sums = NULL;
	err = cudaMalloc((void**) &sums, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sums");

	d_block_scan<<<numSegments, segSize/2>>>(n, in, out, sums);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	// Debug sums output
//	int *h_SUMS_dbs = (int*) malloc(sumsSize);
//	err = cudaMemcpy(h_SUMS_dbs, sums, sumsSize, cudaMemcpyDeviceToHost);
//	CUDA_ERROR(err, "Failed to copy vector sums to h_SUMS_dbs");
//	for (int i = 0; i < numSegments; i++)
//	{
//		printf("sums[%d] = %d\n", i, h_SUMS_dbs[i]);
//	}
//	free(h_SUMS_dbs);

	int *incr = NULL;
	err = cudaMalloc((void**) &incr, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector incr");

	// Unused
	int *sums2 = NULL;
	err = cudaMalloc((void**) &sums2, sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector sums2");

	d_block_scan<<<1, segSize/2>>>(numSegments, sums, incr, sums2);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	// Debug incr output
//	int *h_INCR_dbs = (int*) malloc(sumsSize);
//	err = cudaMemcpy(h_INCR_dbs, incr, sumsSize, cudaMemcpyDeviceToHost);
//	CUDA_ERROR(err, "Failed to copy vector sums to h_INCR_dbs");
//	for (int i = 0; i < numSegments; i++)
//	{
//		printf("incr[%d] = %d\n", i, h_INCR_dbs[i]);
//	}
//	free(h_INCR_dbs);

	d_uniform_add<<<numSegments, segSize/2>>>(n, incr, out);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_uniform_add kernel");

	err = cudaFree(sums);
	CUDA_ERROR(err, "Failed to free device vector sums");
	err = cudaFree(sums2);
	CUDA_ERROR(err, "Failed to free device vector sums2");
	err = cudaFree(incr);
	CUDA_ERROR(err, "Failed to free device vector incr");
}

void three_level_scan(int n, int numSegments, int segSize, int *in, int *out)
{
	cudaError_t err = cudaSuccess;

	size_t sumsSize = numSegments * sizeof(int);
	int *sums = NULL;
	err = cudaMalloc((void**) &sums, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sums");

	d_block_scan<<<numSegments, segSize/2>>>(n, in, out, sums);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	int *sumsScanned = NULL;
	err = cudaMalloc((void**) &sumsScanned, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sumsScanned");

	int sumsNumSegments = 1 + ((numSegments - 1) / (segSize));

	size_t sums2Size = sumsNumSegments * sizeof(int);
	int *sums2 = NULL;
	err = cudaMalloc((void**) &sums2, sums2Size);
	CUDA_ERROR(err, "Failed to allocate device vector sums2");

	d_block_scan<<<sumsNumSegments, segSize/2>>>(numSegments, sums, sumsScanned, sums2);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	int *sums2Scanned = NULL;
	err = cudaMalloc((void**) &sums2Scanned, sums2Size);
	CUDA_ERROR(err, "Failed to allocate device vector sums2Scanned");

	// Unused
	int *sums3 = NULL;
	err = cudaMalloc((void**) &sums3, sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector sums3");

	d_block_scan<<<1, segSize/2>>>(sumsNumSegments, sums2, sums2Scanned, sums3);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	d_uniform_add<<<sumsNumSegments, segSize/2>>>(numSegments, sums2Scanned, sumsScanned);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_uniform_add kernel");

	d_uniform_add<<<numSegments, segSize/2>>>(n, sumsScanned, out);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_uniform_add kernel");

	err = cudaFree(sums);
	CUDA_ERROR(err, "Failed to free device vector sums");
	err = cudaFree(sums2);
	CUDA_ERROR(err, "Failed to free device vector sums2");
	err = cudaFree(sums3);
	CUDA_ERROR(err, "Failed to free device vector sums3");
	err = cudaFree(sumsScanned);
	CUDA_ERROR(err, "Failed to free device vector sumsScanned");
	err = cudaFree(sums2Scanned);
	CUDA_ERROR(err, "Failed to free device vector sums2Scanned");
}

void full_scan(int n, int numSegments, int segSize, int *in, int *out)
{
	if (n <= segSize)
	{
		// 1 Level Scan
		printf("[D_FULL_SCAN] Full scan %d elements with segment size %d => 1 level scan\n", n, segSize);
		one_level_scan(n, numSegments, segSize, in, out);
	}
	else if (n > segSize && n <= segSize * segSize)
	{
		// 2 Level Scan
		printf("[D_FULL_SCAN] Full scan %d elements with segment size %d => 2 level scan\n", n, segSize);
		two_level_scan(n, numSegments, segSize, in, out);
	}
	else if (n > segSize * segSize)
	{
		// 3 Level Scan
		printf("[D_FULL_SCAN] Full scan %d elements with segment size %d => 3 level scan\n", n, segSize);
		three_level_scan(n, numSegments, segSize, in, out);
	}
	else {
		printf("Invalid number of elements %d", n);
		exit(EXIT_FAILURE);
	}
}

bool correct_results_block(int length, int numSegments, int segSize, int *result, int *sums, int *expected, int *expectedSums)
{
	for (int s = 0; s < numSegments; s++)
	{
		for (int i = 0; i < segSize; i++)
		{
			if (result[i] != expected[i])
			{
				printf("TEST FAILED at element %d: %d received, %d expected\n", i, result[i], expected[i]);
				return false;
			}
		}
		if (sums[s] != expectedSums[s])
		{
			printf("TEST FAILED at sum for segment %d: %d received, %d expected\n", s, sums[s], expectedSums[s]);
			return false;
		}
	}
	return true;
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
	int segSize = blockSize * 2;
//	int numElements = 2048;
//	int numElements = 2048 * 2048;
//	int numElements = 8388608;
	int numElements = 10000000;
//	int numElements = 5000;
	size_t size = numElements * sizeof(int);
//	int numBlocks = 1 + ((numElements - 1) / blockSize);
	int numSegments = 1 + ((numElements - 1) / (segSize));

	// Create and initialise host input vector
	int *h_A = (int*) malloc(size);
	printf("Sum Scan of %d elements in %d segments\n", numElements, numSegments);

	for (int i = 0; i < numElements; i++)
	{
		h_A[i] = rand() % 10;
//		if (i % segSize == 0)
//		{
//			printf("seg %d start = %d\n", i / segSize, h_A[i]);
//		}
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
	h_block_scan(numElements, numSegments, segSize, h_A, h_B_hbs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_BLOCK_SCAN] for %d elements in %.5fmSecs\n", numElements, h_msecs);

	//
	// D_BLOCK_SCAN (listing 2)
	//

	int *d_B = NULL;
	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	size_t sumsSize = numSegments * sizeof(int);
	int *d_SUMS = NULL;
	err = cudaMalloc((void**) &d_SUMS, sumsSize);

	cudaEventRecord(start, 0);
	d_block_scan<<<numSegments, blockSize>>>(numElements, d_A, d_B, d_SUMS);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[D_BLOCK_SCAN] Executed block scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

//	// Debug sums output
//	int *h_SUMS_dbs = (int*) malloc(sumsSize);
//	err = cudaMemcpy(h_SUMS_dbs, d_SUMS, sumsSize, cudaMemcpyDeviceToHost);
//	CUDA_ERROR(err, "Failed to copy vector d_SUMS to h_SUMS_dbs");
//	for (int i = 0; i < numSegments; i++)
//	{
//		printf("d_SUMS[%d] = %d\n", i, h_SUMS_dbs[i]);
//	}
//	free(h_SUMS_dbs);

	// Verify result against result of h_block_scan
	int *h_B_dbs = (int*) malloc(size);
	err = cudaMemcpy(h_B_dbs, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dbs");
	if (correct_results_full(numElements, h_B_dbs, h_B_hbs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dbs);

	err = cudaFree(d_SUMS);
	CUDA_ERROR(err, "Failed to free device vector d_SUMS");
	err = cudaFree(d_B);
	CUDA_ERROR(err, "Failed to free device vector d_B");

	//
	// D_BLOCK_SCAN_BCAO
	//

	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

//	size_t sumsSize = numSegments * sizeof(int);
	err = cudaMalloc((void**) &d_SUMS, sumsSize);

	cudaEventRecord(start, 0);
	d_block_scan_bcao<<<numSegments, blockSize>>>(numElements, d_A, d_B, d_SUMS);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[D_BLOCK_SCAN_BCAO] Executed block scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

	// Verify result against result of h_block_scan
	int *h_B_dbs_bcao = (int*) malloc(size);
	err = cudaMemcpy(h_B_dbs_bcao, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dbs_bcao");
	if (correct_results_full(numElements, h_B_dbs_bcao, h_B_hbs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dbs_bcao);

	err = cudaFree(d_SUMS);
	CUDA_ERROR(err, "Failed to free device vector d_SUMS");
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
	// D_FULL_SCAN
	//

	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	cudaEventRecord(start, 0);
	full_scan(numElements, numSegments, segSize, d_A, d_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[D_FULL_SCAN] Executed full scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

	// Verify result against result of h_full_scan
	int *h_B_dfs = (int*) malloc(size);
	err = cudaMemcpy(h_B_dfs, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dfs");
	if (correct_results_full(numElements, h_B_dfs, h_B_hfs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dfs);

	err = cudaFree(d_B);
	CUDA_ERROR(err, "Failed to free device vector d_B");

	//
	// TEARDOWN
	//

	// Free device input vector d_A
	err = cudaFree(d_A);
	CUDA_ERROR(err, "Failed to free device vector d_A");

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
