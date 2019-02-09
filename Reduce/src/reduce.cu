//
// Sam Chatfield
// 1559986
//

#include <stdio.h>
#include <math.h>
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

// Host sequential version of block reduce using a single thread
__host__ void host_blk_reduce(int numElements, int blockSize, float *A)
{
	int segSize = blockSize * 2;
	int numSeg = ceil((float) numElements / (float) segSize);
	// Loop over each segment
	for (int segNum = 0; segNum < numSeg; segNum++)
	{
		int segStartIdx = segNum * segSize;
		// Loop over each element in this segment starting at 1 (0 is the accumulator)
		for (int i = 1; i < segSize; i++)
		{
			int idx = segStartIdx + i;
			if (idx > numElements - 1)
			{
				break;
			}
			A[segStartIdx] += A[idx];
		}
	}
}

// Kernel sequential version of block reduce using a single thread
__global__ void single_thread_blk_reduce(int numElements, int blockSize, float *A)
{
	int segSize = blockSize * 2;
	int numSeg = ceil((float) numElements / (float) segSize);
	// Loop over each segment
	for (int segNum = 0; segNum < numSeg; segNum++)
	{
		int segStartIdx = segNum * segSize;
		// Loop over each element in this segment starting at 1 (0 is the accumulator)
		for (int i = 1; i < segSize; i++)
		{
			int idx = segStartIdx + i;
			if (idx > numElements - 1)
			{
				break;
			}
			A[segStartIdx] += A[idx];
		}
	}
}

// Kernel parallel version of block reduce using global memory
__global__ void global_blk_reduce(int numElements, float *A)
{
	// The size of a data segment is 2 x blockSize
	int segSize = blockDim.x * 2;

	// Work out the index of the vector that this thread is working on
	int i = blockIdx.x * segSize + threadIdx.x;

	if (i < numElements)
	{
		for (uint stride = blockDim.x; stride > 0; stride /= 2)
		{
			__syncthreads();
//			if (blockIdx.x == 0 && threadIdx.x == 0)
//			{
//				printf("stride: %d\n", stride);
//				printf("A[%d] (%0.1f) += A[%d + %d] (%0.1f)\n", i, A[i], i, stride, A[i + stride]);
//			}
			if (threadIdx.x < stride && i + stride < numElements)
			{
				A[i] += A[i + stride];
			}
//			if (blockIdx.x == 0 && threadIdx.x == 0)
//			{
//				printf("A[%d] = %0.1f\n", i, A[i]);
//			}
		}
	}
}

// Kernel parallel version of block reduce using shared memory
__global__ void shared_blk_reduce(int numElements, float *A)
{
	// The size of a data segment is 2 x blockSize
	int segSize = blockDim.x * 2;

	// Work out the index of the vector that this thread is working on
	int i = blockIdx.x * segSize + threadIdx.x;

	// Allocate space for the current segment in device shared memory
	__device__ __shared__ float segment[BLOCK_SIZE * 2];

	// Copy the element to the shared vector
	if (i < numElements)
	{
		segment[threadIdx.x] = A[i];
		segment[threadIdx.x + blockDim.x] = A[i + blockDim.x];

		for (uint stride = blockDim.x; stride > 0; stride /= 2)
		{
			__syncthreads();
//			if (blockIdx.x == 0 && threadIdx.x == 0)
//			{
//				printf("stride: %d\n", stride);
//				printf("segment[%d] (%0.1f) += segment[%d + %d] (%0.1f)\n", threadIdx.x, segment[threadIdx.x], threadIdx.x, stride, segment[threadIdx.x + stride]);
//			}
			if (threadIdx.x < stride && i + stride < numElements)
			{
				segment[threadIdx.x] += segment[threadIdx.x + stride];
			}
//			if (blockIdx.x == 0 && threadIdx.x == 0)
//			{
//				printf("segment[%d] = %0.1f\n", threadIdx.x, segment[threadIdx.x]);
//			}
		}
	}

	// Copy only the result back into the global vector
	if (threadIdx.x == 0)
	{
		A[i] = segment[0];
	}
}

// Host sequential version of full reduce using a single thread
__host__ void host_full_reduce(int numElements, float *A)
{
	for (int i = 1; i < numElements; i++)
	{
		A[0] += A[i];
	}
	printf("Sum for host_full_reduce = %0.1f\n", A[0]);
}

__global__ void full_reduce_compress(int numSegments, int segSize, float *A)
{
	int seg = blockIdx.x * blockDim.x + threadIdx.x;

	if (seg < numSegments)
	{
		A[seg] = A[seg * segSize];
	}
}

// Kernel parallel version of full reduce using global memory
__host__ void global_full_reduce(int numElements, int numSegments, int blockSize, float *A)
{
	int remainingElements = numElements;
	int remainingSegments = numSegments;

	while (remainingElements > 1)
	{
		global_blk_reduce<<<remainingSegments, blockSize>>>(remainingElements, A);

		remainingElements = remainingSegments;
		remainingSegments = 1 + ((remainingElements - 1) / (blockSize * 2));

		full_reduce_compress<<<remainingSegments * 2, blockSize>>>(remainingElements, blockSize * 2, A);
	}
}

// Kernel parallel version of full reduce using shared memory
__host__ void shared_full_reduce(int numElements, int numSegments, int blockSize, float *A)
{
	int remainingElements = numElements;
	int remainingSegments = numSegments;

	while (remainingElements > 1)
	{
		shared_blk_reduce<<<remainingSegments, blockSize>>>(remainingElements, A);

		remainingElements = remainingSegments;
		remainingSegments = 1 + ((remainingElements - 1) / (blockSize * 2));

		full_reduce_compress<<<remainingSegments * 2, blockSize>>>(remainingElements, blockSize * 2, A);
	}
}

// Compare two vectors for equality as expected by block reduce
void compare_results_blk(float *A, float *B, int numElements, int blockSize)
{
	int segSize = blockSize * 2;
	for (int i = 0; i < numElements; i += segSize)
	{
		if (A[i] != B[i])
		{
			printf("TEST FAILED at element %d where A=%0.1f and B=%0.1f\n", i, A[i], B[i]);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test passed\n");
}

// Check that the result in the first element of the vector is the number of elements in the vector
void compare_results_full(int numElements, float *A)
{
	if (A[0] != numElements)
	{
		printf("TEST FAILED at element 0 where A=%0.1f and numElements=%d\n", A[0], numElements);
		exit(EXIT_FAILURE);
	}
	printf("Test passed\n");
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

	// Use this to disable the single_thread_blk_reduce kernel which hugely slows down the computation for large vectors
	int doSingleThread = true;

	int blockSize = 1024;
	int numElements = 1000000;
//	int numElements = 10000000;
	int numBlocks = 1 + ((numElements - 1) / blockSize);
	int numSegments = 1 + ((numElements - 1) / (blockSize * 2));
	printf("numBlocks=%d, numSegments=%d\n", numBlocks, numSegments);
	size_t size = numElements * sizeof(float);
	printf("[Sum Reduce of %d elements with %d blocks of size %d]\n", numElements, numBlocks, blockSize);

	// Allocate host vector A
	float *h_A = (float*) malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL)
	{
		fprintf(stderr, "Failed to allocate host vector h_A\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host vector
	for (int i = 0; i < numElements; i++)
	{
		h_A[i] = 1.0f;
	}

	//
	// HOST_BLK_REDUCE
	//

	// Allocate memory on host for host_blk_reduce, and copy values from h_A
	float *h_A_hbr = (float*) malloc(size);
	memcpy(h_A_hbr, h_A, size);

	sdkStartTimer(&timer);
	host_blk_reduce(numElements, blockSize, h_A_hbr);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[HOST_BLK_REDUCE] Executed sum reduce of %d elements on the host in = %.5fmSecs\n", numElements, h_msecs);

	//
	// SINGLE_THREAD_BLK_REDUCE
	//

	if (doSingleThread == true)
	{
		// Allocate memory on device for single_thread_blk_reduce, and copy values from h_A
		float *d_A_stbr = NULL;
		err = cudaMalloc((void**) &d_A_stbr, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_A_stbr");
		err = cudaMemcpy(d_A_stbr, h_A, size, cudaMemcpyHostToDevice);
		CUDA_ERROR(err, "Failed to copy vector h_A to d_A_stbr");

		cudaEventRecord(start, 0);
		single_thread_blk_reduce<<<1, 1>>>(numElements, blockSize, d_A_stbr);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		// Wait for device to finish
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "Failed to launch single_thread_blk_reduce kernel");

		err = cudaEventElapsedTime(&d_msecs, start, stop);
		CUDA_ERROR(err, "Failed to get elapsed time");
		printf("[SINGLE_THREAD_BLK_REDUCE] Executed sum reduce of %d elements on device in a single thread in = %.5fmSecs\n", numElements, d_msecs);

		// Verify result against result of host_blk_reduce
		float *h_A_stbr = (float*) malloc(size);
		err = cudaMemcpy(h_A_stbr, d_A_stbr, size, cudaMemcpyDeviceToHost);
		CUDA_ERROR(err, "Failed to copy vector d_A_stbr to h_A_stbr");
		compare_results_blk(h_A_hbr, h_A_stbr, numElements, blockSize);
		free(h_A_stbr);

		err = cudaFree(d_A_stbr);
		CUDA_ERROR(err, "Failed to free device vector d_A_stbr");
	}

	//
	// GLOBAL_BLK_REDUCE
	//

	// Allocate memory on device for global_blk_reduce, and copy values from h_A
	float *d_A_gbr = NULL;
	err = cudaMalloc((void**) &d_A_gbr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_gbr");
	err = cudaMemcpy(d_A_gbr, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector h_A to d_A_gbr");

	cudaEventRecord(start, 0);
	global_blk_reduce<<<numSegments, blockSize>>>(numElements, d_A_gbr);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch global_blk_reduce kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[GLOBAL_BLK_REDUCE] Executed sum reduce of %d elements on device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	// Verify result against result of host_blk_reduce
	float *h_A_gbr = (float*) malloc(size);
	err = cudaMemcpy(h_A_gbr, d_A_gbr, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_A_gbr to h_A_gbr");
	compare_results_blk(h_A_hbr, h_A_gbr, numElements, blockSize);
	free(h_A_gbr);

	err = cudaFree(d_A_gbr);
	CUDA_ERROR(err, "Failed to free device vector d_A_gbr");

	//
	// SHARED_BLK_REDUCE
	//

	// Allocate memory on device for shared_blk_reduce, and copy values from h_A
	float *d_A_sbr = NULL;
	err = cudaMalloc((void**) &d_A_sbr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_sbr");
	err = cudaMemcpy(d_A_sbr, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector h_A to d_A_sbr");

	cudaEventRecord(start, 0);
	shared_blk_reduce<<<numSegments, blockSize>>>(numElements, d_A_sbr);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch shared_blk_reduce kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[SHARED_BLK_REDUCE] Executed sum reduce of %d elements on device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	// Verify result against result of host_blk_reduce
	float *h_A_sbr = (float*) malloc(size);
	err = cudaMemcpy(h_A_sbr, d_A_sbr, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_A_sbr to h_A_sbr");
	compare_results_blk(h_A_hbr, h_A_sbr, numElements, blockSize);
	free(h_A_sbr);

	err = cudaFree(d_A_sbr);
	CUDA_ERROR(err, "Failed to free device vector d_A_sbr");

	//
	// BLK_REDUCE TEARDOWN
	//

	free(h_A_hbr);

	//
	// HOST_FULL_REDUCE
	//

	// Allocate memory on host for host_blk_reduce, and copy values from h_A
	float *h_A_hfr = (float*) malloc(size);
	if (h_A_hfr == NULL)
	{
		fprintf(stderr, "Failed to allocate host vector h_A_hfr\n");
		exit(EXIT_FAILURE);
	}
	memcpy(h_A_hfr, h_A, size);

	sdkStartTimer(&timer);
	host_full_reduce(numElements, h_A_hfr);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[HOST_FULL_REDUCE] Executed sum reduce of %d elements on the host in = %.5fmSecs\n", numElements, h_msecs);

	free(h_A_hfr);

	//
	// GLOBAL_FULL_REDUCE
	//

	// Allocate memory on device for global_blk_reduce, and copy values from h_A
	float *d_A_gfr = NULL;
	err = cudaMalloc((void**) &d_A_gfr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_gfr");
	err = cudaMemcpy(d_A_gfr, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector h_A to d_A_gfr");

	cudaEventRecord(start, 0);
	global_full_reduce(numElements, numSegments, blockSize, d_A_gfr);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[GLOBAL_FULL_REDUCE] Executed sum reduce of %d elements on device in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	// Verify result against numElements
	float *h_A_gfr = (float*) malloc(size);
	err = cudaMemcpy(h_A_gfr, d_A_gfr, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_A_gfr to h_A_gfr");
	compare_results_full(numElements, h_A_gfr);
	free(h_A_gfr);

	err = cudaFree(d_A_gfr);
	CUDA_ERROR(err, "Failed to free device vector d_A_gfr");

	//
	// SHARED_FULL_REDUCE
	//

	// Allocate memory on device for global_blk_reduce, and copy values from h_A
	float *d_A_sfr = NULL;
	err = cudaMalloc((void**) &d_A_sfr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_sfr");
	err = cudaMemcpy(d_A_sfr, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector h_A to d_A_sfr");

	cudaEventRecord(start, 0);
	shared_full_reduce(numElements, numSegments, blockSize, d_A_sfr);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	printf("[SHARED_FULL_REDUCE] Executed sum reduce of %d elements on device in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	// Verify result against numElements
	float *h_A_sfr = (float*) malloc(size);
	err = cudaMemcpy(h_A_sfr, d_A_sfr, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_A_sfr to h_A_sfr");
	compare_results_full(numElements, h_A_sfr);
	free(h_A_sfr);

	err = cudaFree(d_A_sfr);
	CUDA_ERROR(err, "Failed to free device vector d_A_sfr");

	//
	// TEARDOWN
	//

	// Free host memory
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
