//
// Sam Chatfield
// 1559986
//
// Working: block scan, full scan for large vectors (limited by GPU memory), BCAO
//
// Test results for 10 million element vector:
// * Block scan           - 3.20790mSecs
// * Block scan with BCAO - 2.45702mSecs
// * Full scan            - 4.81302mSecs
// * Full scan with BCAO  - 4.08365mSecs
//
// CPU - Intel Core i5-6500
// GPU - GeForce GTX 960
//
// Minimised divergence by padding out the shared temp array within the kernels so that all threads
// in a non-filled block don't diverge.
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

__global__ void d_block_scan(int n, int *in, bool returnSums, int *out, int *sums)
{
	__shared__ int temp[BLOCK_SIZE * 2];

	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;
	int offset = 1;

	// Load input into shared memory
	if (segStartIdx + 2*thid < n) temp[2*thid] = in[segStartIdx + 2*thid];
	else temp[2*thid] = 0;
	if (segStartIdx + 2*thid+1 < n) temp[2*thid+1] = in[segStartIdx + 2*thid+1];
	else temp[2*thid+1] = 0;

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
		if (returnSums == true) sums[blockIdx.x] = temp[segSize-1];
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
	if (segStartIdx + 2*thid < n) out[segStartIdx + 2*thid] = temp[2*thid];
	if (segStartIdx + 2*thid+1 < n) out[segStartIdx + 2*thid+1] = temp[2*thid+1];
}

__global__ void d_block_scan_bcao(int n, int *in, bool returnSums, int *out, int *sums)
{
	__shared__ int temp[BLOCK_SIZE * 2 + 64];

	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + blockDim.x;

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	// Load input into shared memory
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
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Zero the last element
	if (thid == 0) {
		// Move total sum to sums array
		if (returnSums == true) sums[blockIdx.x] = temp[segSize-1 + CONFLICT_FREE_OFFSET(segSize-1)];
		// Zero the last element
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
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// Write results to global memory
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
	int segSize = blockDim.x * 2;
	int thid = threadIdx.x;
	int segStartIdx = segSize * blockIdx.x;

	if (segStartIdx + 2*thid < n) out[segStartIdx + 2*thid] += incr[blockIdx.x];
	if (segStartIdx + 2*thid+1 < n) out[segStartIdx + 2*thid+1] += incr[blockIdx.x];
}

void one_level_scan(int n, int numSegments, int segSize, int *in, bool bcao, int *out)
{
	cudaError_t err = cudaSuccess;

	if (bcao == true)
		d_block_scan_bcao<<<numSegments, segSize/2>>>(n, in, false, out, NULL);
	else
		d_block_scan<<<numSegments, segSize/2>>>(n, in, false, out, NULL);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");
}

void two_level_scan(int n, int numSegments, int segSize, int *in, bool bcao, int *out)
{
	cudaError_t err = cudaSuccess;

	size_t sumsSize = numSegments * sizeof(int);
	int *sums = NULL;
	err = cudaMalloc((void**) &sums, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sums");

	if (bcao == true)
		d_block_scan_bcao<<<numSegments, segSize/2>>>(n, in, true, out, sums);
	else
		d_block_scan<<<numSegments, segSize/2>>>(n, in, true, out, sums);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	int *incr = NULL;
	err = cudaMalloc((void**) &incr, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector incr");

	if (bcao == true)
		d_block_scan_bcao<<<1, segSize/2>>>(numSegments, sums, false, incr, NULL);
	else
		d_block_scan<<<1, segSize/2>>>(numSegments, sums, false, incr, NULL);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	d_uniform_add<<<numSegments, segSize/2>>>(n, incr, out);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_uniform_add kernel");

	err = cudaFree(sums);
	CUDA_ERROR(err, "Failed to free device vector sums");
	err = cudaFree(incr);
	CUDA_ERROR(err, "Failed to free device vector incr");
}

void three_level_scan(int n, int numSegments, int segSize, int *in, bool bcao, int *out)
{
	cudaError_t err = cudaSuccess;

	size_t sumsSize = numSegments * sizeof(int);
	int *sums = NULL;
	err = cudaMalloc((void**) &sums, sumsSize);
	CUDA_ERROR(err, "Failed to allocate device vector sums");

	if (bcao == true)
		d_block_scan_bcao<<<numSegments, segSize/2>>>(n, in, true, out, sums);
	else
		d_block_scan<<<numSegments, segSize/2>>>(n, in, true, out, sums);
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

	if (bcao == true)
		d_block_scan_bcao<<<sumsNumSegments, segSize/2>>>(numSegments, sums, true, sumsScanned, sums2);
	else
		d_block_scan<<<sumsNumSegments, segSize/2>>>(numSegments, sums, true, sumsScanned, sums2);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch d_block_scan kernel");

	int *sums2Scanned = NULL;
	err = cudaMalloc((void**) &sums2Scanned, sums2Size);
	CUDA_ERROR(err, "Failed to allocate device vector sums2Scanned");

	if (bcao == true)
		d_block_scan_bcao<<<1, segSize/2>>>(sumsNumSegments, sums2, false, sums2Scanned, NULL);
	else
		d_block_scan<<<1, segSize/2>>>(sumsNumSegments, sums2, false, sums2Scanned, NULL);
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
	err = cudaFree(sumsScanned);
	CUDA_ERROR(err, "Failed to free device vector sumsScanned");
	err = cudaFree(sums2Scanned);
	CUDA_ERROR(err, "Failed to free device vector sums2Scanned");
}

void full_scan(int n, int numSegments, int segSize, int *in, bool bcao, int *out)
{
	if (n <= segSize)
	{
		// 1 Level Scan
		printf("[FULL_SCAN] Full scan %d elements with segment size %d => 1 level scan\n", n, segSize);
		one_level_scan(n, numSegments, segSize, in, bcao, out);
	}
	else if (n > segSize && n <= segSize * segSize)
	{
		// 2 Level Scan
		printf("[FULL_SCAN] Full scan %d elements with segment size %d => 2 level scan\n", n, segSize);
		two_level_scan(n, numSegments, segSize, in, bcao, out);
	}
	else if (n > segSize * segSize)
	{
		// 3 Level Scan
		printf("[FULL_SCAN] Full scan %d elements with segment size %d => 3 level scan\n", n, segSize);
		three_level_scan(n, numSegments, segSize, in, bcao, out);
	}
	else {
		printf("Invalid number of elements %d", n);
		exit(EXIT_FAILURE);
	}
}

bool compare_results(int length, int *result, int *expected)
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
	int numElements = 10000000;
	size_t size = numElements * sizeof(int);
	int numSegments = 1 + ((numElements - 1) / (segSize));

	// Create and initialise host input vector
	int *h_A = (int*) malloc(size);
	printf("Sum Scan of %d elements in %d segments\n", numElements, numSegments);

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

	bool resultCorrectHBS = compare_results(hbsTestSize, h_T_hbs, h_T_hbs_exp);
	if (resultCorrectHBS)
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);

	free(h_T_hbs);

	// ACTUAL CALCULATION

	int *h_B_hbs = (int*) malloc(size);

	sdkResetTimer(&timer);
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

	cudaEventRecord(start, 0);
	d_block_scan<<<numSegments, blockSize>>>(numElements, d_A, false, d_B, NULL);
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
	if (compare_results(numElements, h_B_dbs, h_B_hbs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dbs);

	err = cudaFree(d_B);
	CUDA_ERROR(err, "Failed to free device vector d_B");

	//
	// D_BLOCK_SCAN_BCAO
	//

	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	cudaEventRecord(start, 0);
	d_block_scan_bcao<<<numSegments, blockSize>>>(numElements, d_A, false, d_B, NULL);
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
	if (compare_results(numElements, h_B_dbs_bcao, h_B_hbs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dbs_bcao);

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

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	h_full_scan(hfsTestSize, h_T_hfs_in, h_T_hfs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_FULL_SCAN] for test case in %.5fmSecs\n", h_msecs);

	bool resultCorrectHFS = compare_results(hfsTestSize, h_T_hfs, h_T_hfs_exp);
	if (resultCorrectHFS)
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);

	free(h_T_hfs);

	// ACTUAL CALCULATION

	int *h_B_hfs = (int*) malloc(size);

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	h_full_scan(numElements, h_A, h_B_hfs);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[H_FULL_SCAN] for %d elements in %.5fmSecs\n", numElements, h_msecs);

	//
	// FULL_SCAN
	//

	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	cudaEventRecord(start, 0);
	full_scan(numElements, numSegments, segSize, d_A, false, d_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[FULL_SCAN] Executed full scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

	// Verify result against result of h_full_scan
	int *h_B_dfs = (int*) malloc(size);
	err = cudaMemcpy(h_B_dfs, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dfs");
	if (compare_results(numElements, h_B_dfs, h_B_hfs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dfs);

	err = cudaFree(d_B);
	CUDA_ERROR(err, "Failed to free device vector d_B");

	//
	// FULL_SCAN_BCAO
	//

	err = cudaMalloc((void**) &d_B, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_B");

	cudaEventRecord(start, 0);
	full_scan(numElements, numSegments, segSize, d_A, true, d_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[FULL_SCAN_BCAO] Executed full scan in %d blocks of %d threads in = %.5fmSecs\n", numSegments, blockSize, d_msecs);

	// Verify result against result of h_full_scan
	int *h_B_dfs_bcao = (int*) malloc(size);
	err = cudaMemcpy(h_B_dfs_bcao, d_B, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_B to h_B_dfs_bcao");
	if (compare_results(numElements, h_B_dfs_bcao, h_B_hfs))
		printf("Test passed\n");
	else
		exit(EXIT_FAILURE);
	free(h_B_dfs_bcao);

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
