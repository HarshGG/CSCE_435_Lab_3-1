/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// float cudaMemcpy_host_to_device_time = 0.0f;
// float cudaMemcpy_device_to_host_time = 0.0f;
// float bitonic_sort_step_time = 0.0f;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaEvent_t startEvent;
  cudaEventCreate(&startEvent);
  cudaEvent_t stopEvent;
  cudaEventCreate(&stopEvent);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(startEvent, 0);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, startEvent, stopEvent);
  CALI_MARK_END(cudaMemcpy_host_to_device);


  cudaEventRecord(startEvent, 0);
  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  
  cudaDeviceSynchronize();
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&bitonic_sort_step_time, startEvent, stopEvent);

  effective_bandwidth_gb_s = ((2.0f * NUM_VALS * sizeof(float)) / (1 << 30)) / (bitonic_sort_step_time / 1000.0);
  printf("Effective Bandwidth: %.3f GB/s\n", effective_bandwidth_gb_s);

  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEventRecord(startEvent, 0);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, startEvent, stopEvent);
  CALI_MARK_END(cudaMemcpy_device_to_host);

  cudaFree(dev_values);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  clock_t start = clock();
  bitonic_sort(values); /* Inplace */
  clock_t stop = clock();

  print_elapsed(start, stop);

  // Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;
  int kernel_iterations;

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);
  adiak::value("kernel_iterations", kernel_iterations);

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  array_print(values, NUM_VALS);
}
