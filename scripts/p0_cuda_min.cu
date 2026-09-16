// Minimal CUDA checkpoint/restore test program.
// Allocates device memory, fills a known pattern, holds a device-side counter,
// and writes "<counter> <checksum>" to a status file every second. Prints
// "ready" once initialized.
//
// Usage: p0_cuda_min <status_file> [iters]
// Build: nvcc p0_cuda_min.cu -o p0_cuda_min
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>

#define N (64 * 1024 * 1024)  // 64M floats = 256 MiB
#define SEED 0x9e3779b9u

__device__ int g_counter = 0;

__global__ void fill(float *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (float)((i * SEED) % 1024) * 0.001f;
}

__global__ void bump(int *c) { *c += 1; }

__global__ void accumulate(const float *p, int n, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, (double)p[i]);
}

static double checksum(const float *p, int n) {
  double *d = nullptr;
  cudaMalloc(&d, sizeof(double));
  cudaMemset(d, 0, sizeof(double));
  accumulate<<<(n + 255) / 256, 256>>>(p, n, d);
  double h = 0;
  cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return h;
}

int main(int argc, char **argv) {
  const char *state = argc > 1 ? argv[1] : "/tmp/p0_cuda/state";
  int iters = argc > 2 ? atoi(argv[2]) : 1000000000;
  float *p = nullptr;
  cudaMalloc(&p, N * sizeof(float));
  fill<<<(N + 255) / 256, 256>>>(p, N);
  cudaDeviceSynchronize();

  double cs = checksum(p, N);
  FILE *f = fopen(state, "w");
  if (f) { fprintf(f, "0 %.3f\n", cs); fclose(f); }
  printf("ready pid=%d checksum=%.3f\n", (int)getpid(), cs);
  fflush(stdout);

  for (int i = 0; i < iters; i++) {
    int ctr = 0;
    bump<<<1, 1>>>(&g_counter);
    cudaMemcpyFromSymbol(&ctr, g_counter, sizeof(ctr));
    double cur = checksum(p, N);
    f = fopen(state, "w");
    if (f) { fprintf(f, "%d %.3f\n", ctr, cur); fclose(f); }
    sleep(1);
  }
  cudaFree(p);
  return 0;
}
