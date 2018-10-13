#include "common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <stdlib.h>

#define TS 32

using namespace std;

void initializeMatrix(float *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++){
    matrix[i] = rand() % 10 + 1;
  }
}

void printMatrix(float *matrix, const int rows, const int cols){
  printf("PRINTING MATRIX\n");
  for (size_t i = 0; i < rows; i++){
    for (size_t j = 0; j < cols; j++) {
      printf("%f ", matrix[rows * i + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void multiplyMatricesWithCuda(float *matrixA, float *matrixB, float *result, const int rows, const int cols){
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  if(ix < rows && iy < cols){
    long sum = 0;
    for (size_t i = 0; i < cols; i++) {
      sum += matrixA[iy * rows + i] * matrixB[i * cols + ix];
    }
    result[iy * rows + ix] = sum;
  }
}

__global__ void tilledMatrixMultiplication(float *matrixA, float *matrixB, float *result, const int rows, const int cols){
  __shared__ float tile_A[TS * TS];
  __shared__ float tile_B[TS * TS];
  unsigned int by = blockIdx.y;
  unsigned int bx = blockIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int offset = 0;

  float sum = 0;
  const int offset_limit = (int)ceil((float)cols / TS);
  while (offset < offset_limit) {
    tile_A[TS * ty + tx] = matrixA[rows * by * TS + rows * ty + tx + TS * offset];
    tile_B[TS * ty + tx] = matrixB[bx * TS + rows * ty + tx + rows * TS * offset];
    __syncthreads();
    for (int i = 0; i < TS; i++) {
      sum += tile_A[TS * ty + i] * tile_B[tx + TS * i];
    }
    __syncthreads();
    offset += 1;
  }
  result[rows * (TS * by + ty) + (TS * bx + tx)] = sum;
}

int main(int argc, char const *argv[]) {
  // Set random generator seed
  srand(time(NULL));
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  SAFE_CALL(cudaSetDevice(dev), "Error setting device");

  // Declare matrices
  float *matrixA;
  float *matrixB;
  float *result;
  float *dev_matrixA;
  float *dev_matrixB;
  float *dev_result;

  // Set up size of matrix
  const int rows = 2000;
  const int cols = 2000;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  int bytes = rows * cols * sizeof(float);

  // Allocate matrices memory
  matrixA = (float *) malloc(bytes);
  matrixB = (float *) malloc(bytes);
  result = (float *) malloc(bytes);

  // Allocate device global memory
  SAFE_CALL(cudaMalloc((void **)&dev_matrixA, bytes), "Error allocating dev_matrixA");
  SAFE_CALL(cudaMalloc((void **)&dev_matrixB, bytes), "Error allocating dev_matrixB");
  SAFE_CALL(cudaMalloc((void **)&dev_result, bytes), "Error allocating dev_result");

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // printMatrix(matrixA, rows, cols);
  // printMatrix(matrixB, rows, cols);

  // Transfer data from host to device
  SAFE_CALL(cudaMemcpy(dev_matrixA, matrixA, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixA");
  SAFE_CALL(cudaMemcpy(dev_matrixB, matrixB, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixB");

  // Invoke kernel at host side
  // dim3 block(512, 1);
  // dim3 grid((rows + block.x - 1) / block.x, cols);
  // auto start_at = chrono::high_resolution_clock::now();
  // multiplyMatricesWithCuda<<<grid, block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  // SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  // auto end_at = chrono::high_resolution_clock::now();
  // chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  // printf("Multiply matrices on GPU without tiles <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n",
  //       grid.x, grid.y, block.x, block.y, duration_ms.count(), duration_ms.count() / 1000);

  // Invoke kernel at host side
  dim3 tiles_block(TS, TS);
  dim3 tiles_grid((int)ceil((float)rows / TS), (int)ceil((float)cols / TS));
  auto start_at = chrono::high_resolution_clock::now();
  tilledMatrixMultiplication<<<tiles_grid, tiles_block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  auto end_at = chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  printf("Multiply matrices on GPU with tiles <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n",
        tiles_grid.x, tiles_grid.y, tiles_block.x, tiles_block.y, duration_ms.count(), duration_ms.count() / 1000);

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(result, dev_result, bytes, cudaMemcpyDeviceToHost), "Error copying dev_result");

  // printMatrix(result, rows, cols);

  // Free device global memory
  SAFE_CALL(cudaFree(dev_matrixA), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_matrixB), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_result), "Error freeing memory");

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(result);

  // Reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return 0;
}
