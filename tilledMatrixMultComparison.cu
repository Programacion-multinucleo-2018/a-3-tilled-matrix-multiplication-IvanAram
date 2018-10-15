#include "common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <stdlib.h>

#define TS 8
#define N 2000

using namespace std;

void initializeMatrix(float *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++){
    matrix[i] = rand() % 10 + 1;
  }
}

int checkResult(float *ref1, float *ref2, const int rows, const int cols){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < rows * cols; i++){
        if (abs(ref1[i] - ref2[i]) > epsilon){
            match = 0;
            printf("\nhost: %f | gpu: %f", ref1[i], ref2[i]);
            break;
        }
    }
    return match;
}

void multiplyMatrices(float *matrixA, float *matrixB, float *result, const int rows, const int cols){
  long sum;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      sum = 0.f;
      for (size_t k = 0; k < cols; k++) {
        sum += matrixA[i * rows + k] * matrixB[k * cols + j];
      }
      result[i * rows + j] = sum;
    }
  }
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
    if (TS * offset + tx < cols && by * TS + ty < rows) {
      tile_A[TS * ty + tx] = matrixA[rows * by * TS + rows * ty + tx + TS * offset];
    } else {
      tile_A[TS * ty + tx] = 0;
    }
    if (TS * offset + ty < rows && bx * TS + tx < cols) {
      tile_B[TS * ty + tx] = matrixB[bx * TS + rows * ty + tx + rows * TS * offset];
    } else{
      tile_B[TS * ty + tx] = 0;
    }
    __syncthreads();
    for (int i = 0; i < TS; i++) {
      sum += tile_A[TS * ty + i] * tile_B[tx + TS * i];
    }
    offset += 1;
    __syncthreads();
  }
  if (TS * by + ty < rows && TS * bx + tx < cols) {
    result[rows * (TS * by + ty) + (TS * bx + tx)] = sum;
  }
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
  float *resultCPU;
  float *result;
  float *dev_matrixA;
  float *dev_matrixB;
  float *dev_result;

  // Set up size of matrix
  const int rows = N;
  const int cols = N;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  int bytes = rows * cols * sizeof(float);

  // Allocate matrices memory
  matrixA = (float *) malloc(bytes);
  matrixB = (float *) malloc(bytes);
  result = (float *) malloc(bytes);
  resultCPU = (float *) malloc(bytes);

  // Allocate device global memory
  SAFE_CALL(cudaMalloc((void **)&dev_matrixA, bytes), "Error allocating dev_matrixA");
  SAFE_CALL(cudaMalloc((void **)&dev_matrixB, bytes), "Error allocating dev_matrixB");
  SAFE_CALL(cudaMalloc((void **)&dev_result, bytes), "Error allocating dev_result");

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // Transfer data from host to device
  SAFE_CALL(cudaMemcpy(dev_matrixA, matrixA, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixA");
  SAFE_CALL(cudaMemcpy(dev_matrixB, matrixB, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixB");

  // Multiply matrices in CPU
  auto start_at = chrono::high_resolution_clock::now();
  multiplyMatrices(matrixA, matrixB, resultCPU, rows, cols);
  auto end_at = chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  printf("Multiply matrices on CPU elapsed: %f ms (%.2f seconds)\n", duration_ms.count(), duration_ms.count() / 1000);

  // Invoke kernel at host side
  dim3 block(512, 1);
  dim3 grid((rows + block.x - 1) / block.x, cols);
  start_at = chrono::high_resolution_clock::now();
  multiplyMatricesWithCuda<<<grid, block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  end_at = chrono::high_resolution_clock::now();
  duration_ms = end_at - start_at;
  printf("Multiply matrices on GPU without tiles <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n",
        grid.x, grid.y, block.x, block.y, duration_ms.count(), duration_ms.count() / 1000);

  // Invoke kernel at host side
  dim3 tiles_block(TS, TS);
  dim3 tiles_grid((int)ceil((float)rows / TS), (int)ceil((float)cols / TS));
  start_at = chrono::high_resolution_clock::now();
  tilledMatrixMultiplication<<<tiles_grid, tiles_block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  end_at = chrono::high_resolution_clock::now();
  duration_ms = end_at - start_at;
  printf("Multiply matrices on GPU with square tiles of size %d <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n", TS,
        tiles_grid.x, tiles_grid.y, tiles_block.x, tiles_block.y, duration_ms.count(), duration_ms.count() / 1000);

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(result, dev_result, bytes, cudaMemcpyDeviceToHost), "Error copying dev_result");

  // Check CPU and GPU results
  if (checkResult(result, resultCPU, rows, cols)) {
    printf("Multiplication in CPU and GPU matches\n");
  } else {
    printf("Multiplication in CPU and GPU doesnt match\n");
  }

  // Free device global memory
  SAFE_CALL(cudaFree(dev_matrixA), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_matrixB), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_result), "Error freeing memory");

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(result);
  free(resultCPU);

  // Reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return 0;
}
