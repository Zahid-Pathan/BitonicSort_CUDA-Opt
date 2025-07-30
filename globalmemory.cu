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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int originalSize = atoi(argv[1]);
    int paddedSize = nextPowerOf2(originalSize);
}
