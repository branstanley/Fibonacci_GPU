
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "Fibonacci.h"
#include <stdio.h>
#include <memory>
#include <math.h>


int initial_fibonacci_run(int fib, int currentDepth, int targetDepth);
__global__ void CUDA_Fibonacci(int *fib, int *result);
__device__ void recursive_fibonacci(int fib, int *result);

// Method to get the number of Stream Processors on the current device
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

// Perform the Recursive solution to the Fibonacci Sequence
// Note: This assumes only one graphics card is installed
// Always grabs the first card
int calc_CUDA_Fibonacci(int number){
    //get number of cores

    // The number of Graphics cards in this computer
    int deviceCount = 0;
    // The Index Device we're going to use (Default 0)
    int currentDevice = 0;
    // The number of CUDA cores on the device being used
    int CUDACoreCount = 0;
    // How deep we need to go into recursion to get the optimal number of threads.
    int depth = 0;
    // Error logging
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0){
        printf("Cannot find any CUDA device.");
        exit(EXIT_FAILURE);
    }

    // Get the information on the current device
    cudaSetDevice(currentDevice);
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, currentDevice);

    // Get the number of CUDA Cores on the current device
    CUDACoreCount = getSPcores(deviceProperties);

    // How deep we need to go into the recursion before we spawn threads
    depth = floor(log2((double)CUDACoreCount));

    return initial_fibonacci_run(number, 0, depth);


}

// Method for drilling down to the right level to spawn the correct number of GPU threads
int initial_fibonacci_run(int fib, int currentDepth, int targetDepth){
    if (fib <= 1)
        return fib;

    if (currentDepth < targetDepth)
        return initial_fibonacci_run(fib - 1, currentDepth++, targetDepth) + initial_fibonacci_run(fib - 2, currentDepth++, targetDepth);

    int *d_fib, *d_result;
    int size = sizeof(int);

    cudaMalloc((void**)&d_fib, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_fib, &fib, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, 0, size, cudaMemcpyHostToDevice);

    CUDA_Fibonacci<<<1, 1>>>(d_fib, d_result);
    int result = 0;
    cudaMemcpy(&result, d_result, size, cudaMemcpyDeviceToHost);    

    cudaFree(d_fib);
    cudaFree(d_result);
    return result;
}

// Call to the device to start being recursive on a thread
__global__ void CUDA_Fibonacci(int *fib, int *result){
    recursive_fibonacci(*fib, result);
}

// Recursive Fibonacci to run on the GPU Thread
__device__ void recursive_fibonacci(int fib, int *result){
    if (fib <= 1){
        *result += fib;
        return;
    }
    recursive_fibonacci(fib - 1, result);
    recursive_fibonacci(fib - 2, result);
}
