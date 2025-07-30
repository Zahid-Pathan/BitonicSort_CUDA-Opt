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
