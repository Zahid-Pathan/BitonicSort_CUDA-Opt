#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <iostream> // Include for std::cout and std::cerr

// Function to calculate the next power of 2
int nextPowerOf2(int n) {
    if (n == 0) return 1;
    if ((n & (n - 1)) == 0) return n;
    int count = 0;
    while (n != 0) {
        n >>= 1;
        count += 1;
    }
    return 1 << count;
}


__device__ void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}


__global__ void bitonicSortKernel(int* arr, int size, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i && ixj < size) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                swap(arr, i, ixj);
            }
        } else {
            if (arr[i] < arr[ixj]) {
                swap(arr, i, ixj);
            }
        }
    }
}

// Function to perform bitonic sort on GPU
void bitonicSort(int* d_arr, int size) {
    int numThreads = 512;
    int numBlocks = (size + numThreads - 1) / numThreads;

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonicSortKernel<<<numBlocks, numThreads>>>(d_arr, size, j, k);
            cudaDeviceSynchronize(); // Add synchronization to ensure kernel completes
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int originalSize = atoi(argv[1]);
    int paddedSize = nextPowerOf2(originalSize);

    srand(time(NULL));

    // ======================================================================
    // arrCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(originalSize * sizeof(int));
    int* arrPaddedCpu = (int*)malloc(paddedSize * sizeof(int));
    int* arrSortedGpu = (int*)malloc(originalSize * sizeof(int));

    // Initialize original array
    printf("Initial array: \n");
    for (int i = 0; i < originalSize; i++) {
        arrCpu[i] = rand() % 1000;
        printf("%d ", arrCpu[i]);
    }
    printf("\n");

    // Initialize padded array (fill excess with INT_MAX)
    for (int i = 0; i < originalSize; i++) {
        arrPaddedCpu[i] = arrCpu[i];
    }
    for (int i = originalSize; i < paddedSize; i++) {
        arrPaddedCpu[i] = INT_MAX;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer padded data (arrPaddedCpu) to device
    // ======================================================================
    int* arrGpu;
    cudaMalloc(&arrGpu, paddedSize * sizeof(int));
    cudaMemcpy(arrGpu, arrPaddedCpu, paddedSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================
    bitonicSort(arrGpu, paddedSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arrPaddedCpu for comparison)
    // ======================================================================
    cudaMemcpy(arrPaddedCpu, arrGpu, paddedSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    // Print sorted array (only original size)
    printf("Sorted array: \n");
    for (int i = 0; i < originalSize; i++) {
        printf("%d ", arrPaddedCpu[i]);
    }
    printf("\n");


    auto startTime = std::chrono::high_resolution_clock::now();

    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + originalSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    // Remove padding and compare

    bool match = true;
    for (int i = 0; i < originalSize; i++) {
        if (arrCpu[i] != arrPaddedCpu[i]) {
            match = false;
            break;
        }
    }

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONAL FAIL\n\033[0m");
        // Print CPU sorted array for debugging if functional test fails
        printf("CPU sorted array: \n");
        for (int i = 0; i < originalSize; i++) {
            printf("%d ", arrCpu[i]);
        }
        printf("\n");
        return 0;
    }

    printf("\033[1;34mArray size         :\033[0m %d\n", originalSize);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    float speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = originalSize / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %fx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m %fx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    free(arrCpu);
    free(arrPaddedCpu);
    free(arrSortedGpu);
    cudaFree(arrGpu);

    return 0;
}
