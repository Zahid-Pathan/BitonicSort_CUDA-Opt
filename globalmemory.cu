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
}
