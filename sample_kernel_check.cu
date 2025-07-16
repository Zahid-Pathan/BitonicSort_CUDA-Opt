#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA!\n");
}

int main() {
    // Launch the kernel
    hello_kernel<<<1, 1>>>();

    // Check for errors during kernel launch
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
        return 1;
    }
    std::cout << "Kernel launched." << std::endl;

    // Synchronize and check for errors during kernel execution
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(sync_error) << std::endl;
        return 1;
    }
    std::cout << "Device synchronized." << std::endl;

    return 0;
}