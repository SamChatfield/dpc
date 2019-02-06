#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

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
		printf("segStartIdx = %d\n", segStartIdx);
		// Loop over each element in this segment starting at 1 (0 is the accumulator)
		for (int i = 1; i < segSize - 1; i++)
		{
			int idx = segStartIdx + i;
			if (idx > numElements - 1)
			{
				printf("Segment finished early at i=%d\n", i);
				break;
			}
//			printf("Add i=%d: %f to acc i=%d: %f\n", idx, A[idx], segStartIdx, A[segStartIdx]);
			A[segStartIdx] += A[idx];
		}
		printf("Sum for segment %d = %f\n", segNum, A[segStartIdx]);
	}
}

// Kernel sequential version of block reduce using a single thread
__global__ void single_thread_blk_reduce()
{

}

// Kernel parallel version of block reduce using global memory
__global__ void global_blk_reduce()
{

}

// Kernel parallel version of block reduce using shared memory
__global__ void shared_blk_reduce()
{

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
	int numElements = 51200;
	size_t size = numElements * sizeof(float);
	printf("[Sum Reduce of %d elements]\n", numElements);

	// Allocate host vector A
	float *h_A = (float*) malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL)
	{
		fprintf(stderr, "Failed to allocate host vector A!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host vector
	for (int i = 0; i < numElements; i++)
	{
		h_A[i] = 1.0f;
	}

	// Execute the vector addition on the Host and time it:
	sdkStartTimer(&timer);
	host_blk_reduce(numElements, blockSize, h_A);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Executed Sum Reduce of %d elements on the Host in = %.5fmSecs\n", numElements, h_msecs);

	return 0;
}
