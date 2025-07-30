#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

#define BLOCK_SIZE 512
#define MAX_SHARED_SIZE 2048

// Function to calculate the next power of 2
int nextPowerOf2(int n)
{
  if (n == 0)
    return 1;
  if ((n & (n - 1)) == 0)
    return n;
  int count = 0;
  while (n != 0)
  {
    n >>= 1;
    count += 1;
  }
  return 1 << count;
}

__device__ void swap(int &a, int &b)
{
  int temp = a;
  a = b;
  b = temp;
}

__global__ void bitonicSortSharedKernel(int *arr, int size, int originalSize)
{
  __shared__ int sharedMem[MAX_SHARED_SIZE];
  for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)

  {

    int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;

    if (i < originalSize)
    {
      sharedMem[threadIdx.x + iter * blockDim.x] = arr[i];
    }
    else

    {
      sharedMem[threadIdx.x + iter * blockDim.x] = INT_MAX;
    }
    __syncthreads();
  }
  for (int k = 2; k <= min(MAX_SHARED_SIZE, size); k *= 2)
  {
    for (int j = k / 2; j > 0; j /= 2)
    {
      for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)
      {
        int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;
        if (i < size)
        {
          int p = iter * blockDim.x + threadIdx.x;

          int ixj = p ^ j;

          if (ixj > p)
          {

            if ((i & k) == 0)
            {

              if (sharedMem[p] > sharedMem[ixj])
              {
                swap(sharedMem[p], sharedMem[ixj]);
              }
            }
            else
            {
              if (sharedMem[p] < sharedMem[ixj])
              {
                swap(sharedMem[p], sharedMem[ixj]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
  for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)
  {
    int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;
    if (i < size)
    {

      arr[i] = sharedMem[threadIdx.x + iter * blockDim.x];
    }
  }
}

__global__ void bitonic_JSortSharedKernel(int *arr, int size, int k)
{
  __shared__ int sharedMem[MAX_SHARED_SIZE];
  for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)
  {
    int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;
    if (i < size)
    {
      sharedMem[threadIdx.x + iter * blockDim.x] = arr[i];

      __syncthreads();
    }
  }
  for (int j = MAX_SHARED_SIZE / 2; j > 0; j /= 2)
  {

    for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)
    {
      int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;
      if (i < size)
      {
        int p = iter * blockDim.x + threadIdx.x;

        int ixj = p ^ j;
        if (ixj > p)
        {

          if ((i & k) == 0)
          {
            if (sharedMem[p] > sharedMem[ixj])
            {
              swap(sharedMem[p], sharedMem[ixj]);
            }
          }
          else
          {
            if (sharedMem[p] < sharedMem[ixj])
            {
              swap(sharedMem[p], sharedMem[ixj]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  for (int iter = 0; iter < MAX_SHARED_SIZE / BLOCK_SIZE; iter++)
  {
    int i = iter * blockDim.x + threadIdx.x + MAX_SHARED_SIZE * blockIdx.x;
    if (i < size)
    {

      arr[i] = sharedMem[threadIdx.x + iter * blockDim.x];
    }
  }
}

__global__ void bitonicSortGlobalKernel(int *arr, int size, int j, int k)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int ixj = i ^ j;

  if (ixj > i && ixj < size)
  {
    if ((i & k) == 0)
    {

      if (arr[i] > arr[ixj])
      {
        swap(arr[i], arr[ixj]);
      }
    }
    else
    {
      if (arr[i] < arr[ixj])
      {
        swap(arr[i], arr[ixj]);
      }
    }
  }
}

void bitonicSort(int *d_arr, int size, int originalSize)
{
  int numThreads = BLOCK_SIZE;
  int numBlocks = max((size) / MAX_SHARED_SIZE, 1);
  int numBlocksGlobal = (size) / BLOCK_SIZE;

  bitonicSortSharedKernel<<<numBlocks, numThreads, MAX_SHARED_SIZE * sizeof(int)>>>(d_arr, size, originalSize);

  for (int k = MAX_SHARED_SIZE; k <= size; k <<= 1)
  {

    for (int j = k >> 1; j > MAX_SHARED_SIZE / 2; j = j >> 1)
    {

      bitonicSortGlobalKernel<<<numBlocksGlobal, numThreads>>>(d_arr, size, j, k);
    }
    bitonic_JSortSharedKernel<<<numBlocks, numThreads, MAX_SHARED_SIZE * sizeof(int)>>>(d_arr, size, k);
  }

  cudaDeviceSynchronize();
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Usage: %s <array_size>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);

  srand(time(NULL));

  // ======================================================================
  // arrCpu contains the input random array
  // arrSortedGpu should contain the sorted array copied from GPU to CPU
  // ======================================================================
  int *arrCpu = (int *)malloc(size * sizeof(int));
  int *arrSortedGpu = (int *)malloc(size * sizeof(int));

  for (int i = 0; i < size; i++)
  {
    arrCpu[i] = rand() % 1000;
  }

  float gpuTime, h2dTime, d2hTime, cpuTime = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // ======================================================================
  // Transfer data (arr_cpu) to device
  // ======================================================================

  // your code goes here .......

  int paddedSize = nextPowerOf2(size);

  // cudaHostRegister(arrPaddedCpu, paddedSize * sizeof(int), cudaHostRegisterDefault);
  int *arrGpu;
  cudaMalloc(&arrGpu, paddedSize * sizeof(int));
  cudaMemcpy(arrGpu, arrCpu, size * sizeof(int), cudaMemcpyHostToDevice);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&h2dTime, start, stop);

  cudaEventRecord(start);

  // ======================================================================
  // Perform bitonic sort on GPU
  // ======================================================================
  bitonicSort(arrGpu, paddedSize, size);
  cudaHostRegister(arrSortedGpu, size * sizeof(int), cudaHostRegisterDefault);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  cudaEventRecord(start);

  // ======================================================================
  // Transfer sorted data back to host (copied to arrSortedGpu)
  // ======================================================================

  cudaMemcpy(arrSortedGpu, arrGpu, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&d2hTime, start, stop);

  auto startTime = std::chrono::high_resolution_clock::now();

  // CPU sort for performance comparison
  std::sort(arrCpu, arrCpu + size);

  auto endTime = std::chrono::high_resolution_clock::now();
  cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
  cpuTime = cpuTime / 1000;

  int match = 1;
  for (int i = 0; i < size; i++)
  {
    if (arrSortedGpu[i] != arrCpu[i])
    {
      match = 0;
      break;
    }
  }

  free(arrCpu);
  free(arrSortedGpu);

  if (match)
    printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
  else
  {
    printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
    return 0;
  }

  printf("\033[1;34mArray size         :\033[0m %d\n", size);
  printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
  float gpuTotalTime = h2dTime + gpuTime + d2hTime;
  int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime / cpuTime) : (cpuTime / gpuTotalTime);
  float meps = size / (gpuTotalTime * 0.001) / 1e6;
  printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
  printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
  if (gpuTotalTime < cpuTime)
  {
    printf("\033[1;32mPERF PASSING\n\033[0m");
    printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
    printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
    printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
    printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
  }
  else
  {
    printf("\033[1;31mPERF FAILING\n\033[0m");
    printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
    printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
    printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
    printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    return 0;
  }

  return 0;
}
