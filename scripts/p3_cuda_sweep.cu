// Phase 3 CUDA device-memory sweep probe.
// Allocates a CLI-configurable device buffer, fills a fixed pattern, holds a
// device-side counter (in a device buffer, not a __device__ global), and writes
// "<counter> <checksum>" to a status file each second.
//
// Usage: p3_cuda_sweep <MB> <status_file> [iters]
// Build: nvcc -O2 -arch=sm_89 scripts/p3_cuda_sweep.cu -o scripts/p3_cuda_sweep
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define SEED 0x9e3779b9u

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

__global__ void fill(float *p, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (float)((uint64_t)(i * (long)SEED) % 1024) * 0.001f;
}

__global__ void bump(int *c) { *c += 1; }

__global__ void accumulate(const float *p, long n, unsigned long long *out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Pattern values are exactly k*0.001f; recover integer k and accumulate as
    // an integer so the sum is independent of atomicAdd ordering.
    unsigned long long k = (unsigned long long)(p[i] * 1000.0f + 0.5f);
    atomicAdd(out, k);
  }
}

static unsigned grid_for(long n) {
  return (unsigned)((n + 255) / 256);
}

static void write_state(const char *state, int ctr, unsigned long long sum) {
  FILE *f = fopen(state, "w");
  if (f) { fprintf(f, "%d %llu\n", ctr, sum); fclose(f); }
}

int main(int argc, char **argv) {
  long mb = argc > 1 ? strtol(argv[1], 0, 10) : 1024;
  const char *state = argc > 2 ? argv[2] : "/tmp/p3_cuda/state";
  long iters = argc > 3 ? strtol(argv[3], 0, 10) : 1000000000L;
  long n = (mb << 20) / 4;

  float *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, (size_t)n * sizeof(float)));
  int *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&dc, sizeof(int)));
  CUDA_CHECK(cudaMemset(dc, 0, sizeof(int)));
  unsigned long long *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(unsigned long long)));

  fill<<<grid_for(n), 256>>>(p, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemset(d, 0, sizeof(unsigned long long)));
  accumulate<<<grid_for(n), 256>>>(p, n, d);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long sum = 0;
  CUDA_CHECK(cudaMemcpy(&sum, d, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));

  write_state(state, 0, sum);
  printf("ready pid=%d mb=%ld checksum=%llu\n", (int)getpid(), mb, sum);
  fflush(stdout);

  for (long i = 0; i < iters; i++) {
    int ctr = 0;
    bump<<<1, 1>>>(dc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&ctr, dc, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d, 0, sizeof(unsigned long long)));
    accumulate<<<grid_for(n), 256>>>(p, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&sum, d, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    write_state(state, ctr, sum);
    sleep(1);
  }
  cudaFree(p);
  cudaFree(dc);
  cudaFree(d);
  return 0;
}
