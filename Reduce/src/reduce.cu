#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 1024

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
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
	printf("%d segments of size %d for %d elements\n", numSeg, segSize, numElements);
	// Loop over each segment
	for (int segNum = 0; segNum < numSeg; segNum++)
	{
		int segStartIdx = segNum * segSize;
//		printf("segStartIdx = %d\n", segStartIdx);
		// Loop over each element in this segment starting at 1 (0 is the accumulator)
		for (int i = 1; i < segSize - 1; i++)
		{
			int idx = segStartIdx + i;
			if (idx > numElements - 1)
			{
//				printf("Segment finished early at i=%d\n", i);
				break;
			}
//			printf("Add i=%d: %f to acc i=%d: %f\n", idx, A[idx], segStartIdx, A[segStartIdx]);
			A[segStartIdx] += A[idx];
		}
		printf("Sum for segment %d = %f\n", segNum, A[segStartIdx]);
	}
}

// Kernel sequential version of block reduce using a single thread
__global__ void single_thread_blk_reduce(int numElements, int blockSize, float *A)
{
	int segSize = blockSize * 2;
	int numSeg = ceil((float) numElements / (float) segSize);
	printf("%d segments of size %d for %d elements\n", numSeg, segSize, numElements);
	// Loop over each segment
	for (int segNum = 0; segNum < numSeg; segNum++)
	{
		int segStartIdx = segNum * segSize;
//		printf("segStartIdx = %d\n", segStartIdx);
		// Loop over each element in this segment starting at 1 (0 is the accumulator)
		for (int i = 1; i < segSize - 1; i++)
		{
			int idx = segStartIdx + i;
			if (idx > numElements - 1)
			{
//				printf("Segment finished early at i=%d\n", i);
				break;
			}
//			printf("Add i=%d: %f to acc i=%d: %f\n", idx, A[idx], segStartIdx, A[segStartIdx]);
			A[segStartIdx] += A[idx];
		}
//		printf("Sum for segment %d = %f\n", segNum, A[segStartIdx]);
	}
}

// Kernel parallel version of block reduce using global memory
__global__ void global_blk_reduce(int numElements, float *A)
{
	// The size of a data segment is 2 x blockSize
	int segSize = blockDim.x * 2;

	// Work out the index of the vector that this thread is working on
	int i = blockIdx.x * segSize + threadIdx.x;

	for (uint stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		// TODO: remove
//		if (blockIdx.x == 24 && threadIdx.x == 0)
//		{
//			printf("stride: %d\n", stride);
//			printf("A[%d] (%f) += A[%d + %d] (%f)\n", i, A[i], i, stride, A[i + stride]);
//		}
		if (threadIdx.x < stride)
		{
			A[i] += A[i + stride];
		}
//		if (blockIdx.x == 24 && threadIdx.x == 0)
//		{
//			printf("A[%d] = %f\n", i, A[i]);
//		}
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
	}

	for (uint stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		// TODO: remove
//		if (blockIdx.x == 0 && threadIdx.x == 0)
//		{
//			printf("stride: %d\n", stride);
//			printf("segment[%d] (%f) += segment[%d + %d] (%f)\n", threadIdx.x, segment[threadIdx.x], threadIdx.x, stride, segment[threadIdx.x + stride]);
//		}
		if (threadIdx.x < stride)
		{
			segment[threadIdx.x] += segment[threadIdx.x + stride];
		}
//		if (blockIdx.x == 0 && threadIdx.x == 0)
//		{
//			printf("segment[%d] = %f\n", threadIdx.x, segment[threadIdx.x]);
//		}
	}

	// Copy only the result back into the global vector
	if (threadIdx.x == 0)
	{
		A[i] = segment[0];
	}
}

// Host sequential version of full reduce using a single thread
__host__ void host_full_reduce()
{

}

// Kernel parallel version of full reduce using global memory
__global__ void global_full_reduce()
{

}

// Kernel parallel version of full reduce using shared memory
__global__ void shared_full_reduce()
{

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
//	int numElements = 51200;
	int numElements = 50000;
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

	// Allocate the device vector A
	float *d_A = NULL;
	err = cudaMalloc((void**) &d_A, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A");

	// Initialise the device vector by copying from the host vector
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector A from host to device");

	//
	// host_blk_reduce
	//

	sdkStartTimer(&timer);
	host_blk_reduce(numElements, blockSize, h_A);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("[host_blk_reduce] Executed Sum Reduce of %d elements on the Host in = %.5fmSecs\n", numElements, h_msecs);

	//
	// single_thread_blk_reduce
	//

	// Allocate memory on device for single_thread_blk_reduce, and copy values from d_A
	float *d_A_stbr = NULL;
	err = cudaMalloc((void**) &d_A_stbr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_stbr");
	err = cudaMemcpy(d_A_stbr, d_A, size, cudaMemcpyDeviceToDevice);
	CUDA_ERROR(err, "Failed to copy vector d_A to d_A_stbr");

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
	printf("[single_thread_blk_reduce] Executed Sum Reduce of %d elements on the Device in a SINGLE THREAD in = %.5fmSecs\n", numElements, d_msecs);

	err = cudaFree(d_A_stbr);
	CUDA_ERROR(err, "Failed to free device vector d_A_stbr");

	//
	// global_blk_reduce
	//

	// Allocate memory on device for global_blk_reduce, and copy values from d_A
	float *d_A_gbr = NULL;
	err = cudaMalloc((void**) &d_A_gbr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_gbr");
	err = cudaMemcpy(d_A_gbr, d_A, size, cudaMemcpyDeviceToDevice);
	CUDA_ERROR(err, "Failed to copy vector d_A to d_A_gbr");

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
	printf("[global_blk_reduce] Executed Sum Reduce of %d elements on the Device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	err = cudaFree(d_A_gbr);
	CUDA_ERROR(err, "Failed to free device vector d_A_gbr");

	//
	// shared_blk_reduce
	//

	// Allocate memory on device for global_blk_reduce, and copy values from d_A
	float *d_A_sbr = NULL;
	err = cudaMalloc((void**) &d_A_sbr, size);
	CUDA_ERROR(err, "Failed to allocate device vector d_A_sbr");
	err = cudaMemcpy(d_A_sbr, d_A, size, cudaMemcpyDeviceToDevice);
	CUDA_ERROR(err, "Failed to copy vector d_A to d_A_sbr");

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
	printf("[shared_blk_reduce] Executed Sum Reduce of %d elements on the Device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, numSegments, blockSize, d_msecs);

	err = cudaFree(d_A_sbr);
	CUDA_ERROR(err, "Failed to free device vector d_A_sbr");

	//
	// Teardown
	//

	// Free device global memory
	err = cudaFree(d_A);
	CUDA_ERROR(err, "Failed to free device vector d_A");

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
	return 0;
}
