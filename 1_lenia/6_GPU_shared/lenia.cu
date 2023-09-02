#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>

extern char *optarg;
extern int optopt;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ float bell(const float val, const float mean, const float std)
{
    const float t = (val - mean) / std;
    return expf(-(t * t) / 2.f);
}

__host__ __device__ float clamp(float d, float min, float max) {
  const float t = d < min ? min : d;
  return t > max ? max : t;
}

void init_kernel(float *const kernel, const size_t size, const size_t R, const float mean, const float std)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            double res = 0;
            double dist2 = hypot((double)i - R, (double)j - R);
            if (dist2 < R) {
                res = bell(dist2 / R, mean, std);
            }
            sum += res;
            kernel[i * size + j] = res;
        }
    }    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            kernel[i * size + j] /= sum;
        }
    }
}

__global__ void make_step(
        float const *const __restrict__ current_state,
        float *const __restrict__ next_state, const size_t size,
        float const *const __restrict__ kernel, const size_t kernel_size,
        const float dT, const float growth_mean, const float growth_std
)
{
    const size_t R = (kernel_size - 1) / 2;

    extern __shared__ char shared_data[];
    float * const __restrict__ kernel_shared = (float*)shared_data;
    float * const __restrict__ state_shared = (float*)shared_data + kernel_size * kernel_size;

    const size_t thread_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t thread_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t BLOCK_SIZE_FLAT = blockDim.x * blockDim.y;
    const size_t THREAD_IDX = threadIdx.x + blockDim.x * threadIdx.y;

    const size_t KERNEL_DATA_SIZE = kernel_size * kernel_size;
    const size_t KERNEL_NCOPY = (KERNEL_DATA_SIZE + BLOCK_SIZE_FLAT - 1) / BLOCK_SIZE_FLAT;
    for (size_t i = 0, p = THREAD_IDX * KERNEL_NCOPY; i < KERNEL_NCOPY && p < KERNEL_DATA_SIZE; ++i, ++p) {
        kernel_shared[p] = kernel[p];
    }

    const size_t STATE_DATA_SIZE_X = blockDim.x + 2 * R;
    const size_t STATE_DATA_SIZE_Y = blockDim.y + 2 * R;
    const size_t STATE_DATA_SIZE = STATE_DATA_SIZE_X * STATE_DATA_SIZE_Y;
    const size_t STATE_NCOPY = (STATE_DATA_SIZE + BLOCK_SIZE_FLAT - 1) / BLOCK_SIZE_FLAT;

    const size_t STATE_SRC_Y_OFFSET = blockDim.y * blockIdx.y;
    const size_t STATE_SRC_X_OFFSET = blockDim.x * blockIdx.x;


    for (size_t i = 0, p = THREAD_IDX * STATE_NCOPY; i < STATE_NCOPY && p < STATE_DATA_SIZE; ++i, ++p) {
        const size_t STATE_Y = p / STATE_DATA_SIZE_X;
        const size_t STATE_X = p % STATE_DATA_SIZE_X;
        const size_t SRC_Y = (STATE_SRC_Y_OFFSET + STATE_Y + size - R) % size;
        const size_t SRC_X = (STATE_SRC_X_OFFSET + STATE_X + size - R) % size;
        state_shared[p] = current_state[SRC_Y * size + SRC_X];
    }

    if (thread_pos_x >= size || thread_pos_y >= size) {
        return;
    }

    __syncthreads();
    float next_state_val = 0;
    
    for (size_t ki = 0; ki < kernel_size; ++ki) {
        const size_t ki_offset = ki * kernel_size;
        const size_t src_i_offset = (threadIdx.y + ki) * STATE_DATA_SIZE_X + threadIdx.x;
        for (size_t kj = 0; kj < kernel_size; ++kj) {
            next_state_val += kernel_shared[ki_offset + kj] * state_shared[src_i_offset + kj];
        }
    }

    const size_t idx = (threadIdx.y + R) * STATE_DATA_SIZE_X + threadIdx.x + R;
    next_state_val = clamp(state_shared[idx] + dT * (bell(next_state_val, growth_mean, growth_std) * 2 - 1), 0, 1);
    next_state[thread_pos_y * size + thread_pos_x] = next_state_val;
}

int main(int argc, char *argv[])
{
    size_t nsteps = 32;
    char const *input_path = "input.bin";
    char const *output_path = "output.bin";

    int c;
    while ((c = getopt(argc, argv, "i:s:o:")) != -1)
    {
        switch (c)
        {
            char *endptr;
            case 's':
                nsteps = strtoul(optarg, &endptr, 10);
                break;
            case 'i':
                input_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case '?':
                printf("Unknown option `-%c'.\n", optopt);
                return 1;
            default:
                abort();
        }
    }

    FILE *inFile = fopen(input_path, "rb");
    
    size_t FIELD_SIZE = 256;
    float dT = 0.1;
    size_t R = 13;  // cells per kernel radius
    float GROWSH_MEAN = 0.15;
    float GROWSH_STD = 0.015;

    size_t FIELD_DATA_SIZE, RESULT_DATA_SIZE;
    float *result = NULL;
    if (inFile != NULL) {
        if (!(
            fread(&FIELD_SIZE, sizeof(size_t), 1, inFile)
            && fread(&dT, sizeof(float), 1, inFile)
            && fread(&R, sizeof(size_t), 1, inFile)
            && fread(&GROWSH_MEAN, sizeof(float), 1, inFile)
            && fread(&GROWSH_STD, sizeof(float), 1, inFile)
        )) {
            printf("Cannot read parameters\n");
            abort();
        }

        FIELD_DATA_SIZE = FIELD_SIZE * FIELD_SIZE;
        RESULT_DATA_SIZE = FIELD_DATA_SIZE * (nsteps + 1);
        cudaHostAlloc(&result, RESULT_DATA_SIZE * sizeof(float), 0);

        if (!fread(result, FIELD_DATA_SIZE * sizeof(float), 1, inFile)) {
            printf("Cannot read field\n");
            abort();
        }
        fclose(inFile);
    }
    else {
        printf("Cannot open file %s\n" , input_path);
        abort();
    }

    const float KERNEL_MEAN = 0.5;
    const float KERNEL_STD = 0.15;
    const size_t KERNEL_SIZE = 2 * R + 1;

    float *const kernel = (float*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(float));
    init_kernel(kernel, KERNEL_SIZE, R, KERNEL_MEAN, KERNEL_STD);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float *kernel_gpu;
    gpuErrchk(cudaMalloc(&kernel_gpu, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    gpuErrchk(cudaMemcpy(kernel_gpu, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    float *current_field_gpu, *next_field_gpu, *current_field_padded_gpu;
    gpuErrchk(cudaMalloc(&current_field_gpu, FIELD_DATA_SIZE * sizeof(float)));
    gpuErrchk(cudaMemcpy(current_field_gpu, result, FIELD_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&next_field_gpu, FIELD_DATA_SIZE * sizeof(float)));
    gpuErrchk(cudaMemset(next_field_gpu, 0, FIELD_DATA_SIZE * sizeof(float)));

    const size_t PADDED_SIZE = FIELD_SIZE + 2 * R;
    gpuErrchk(cudaMalloc(&current_field_padded_gpu, PADDED_SIZE * PADDED_SIZE * sizeof(float)));

    dim3 block_shape = dim3(32, 24);
    dim3 grid_shape  = dim3((FIELD_SIZE + block_shape.x - 1) / block_shape.x, (FIELD_SIZE + block_shape.y - 1) / block_shape.y);

    cudaStream_t kernel_stream, copy_stream;
    gpuErrchk(cudaStreamCreate(&kernel_stream));
    gpuErrchk(cudaStreamCreate(&copy_stream));

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    for (size_t it = 0; it < nsteps; ++it) {
        float *const next_state = &result[(it+1) * FIELD_DATA_SIZE];
        make_step<<<grid_shape, block_shape, (KERNEL_SIZE*KERNEL_SIZE + (block_shape.x+2*R)*(block_shape.y+2*R))*sizeof(int), kernel_stream>>>(current_field_gpu, next_field_gpu, FIELD_SIZE, kernel_gpu, KERNEL_SIZE, dT, GROWSH_MEAN, GROWSH_STD);
        gpuErrchk(cudaStreamSynchronize(kernel_stream));
        gpuErrchk(cudaStreamSynchronize(copy_stream));
        gpuErrchk(cudaMemcpyAsync(next_state, next_field_gpu, FIELD_DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
        float *tmp = current_field_gpu;
        current_field_gpu = next_field_gpu;
        next_field_gpu = tmp;
    }
    gpuErrchk(cudaStreamSynchronize(copy_stream));
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float time_diff_ms = 0;
    gpuErrchk(cudaEventElapsedTime(&time_diff_ms, start, stop));
    fprintf(stderr, "Program took %.4f ms\n%.8f ms per iteration\n", time_diff_ms, time_diff_ms / nsteps);
    FILE *outFile = fopen(output_path, "wb");
    if (outFile != NULL) {
        fwrite(&FIELD_SIZE, sizeof(size_t), 1, outFile);
        fwrite(&nsteps, sizeof(size_t), 1, outFile);
        fwrite(result, RESULT_DATA_SIZE * sizeof(float), 1, outFile);
        fclose(outFile);
    }
    cudaFree(kernel_gpu);
    cudaFree(current_field_gpu);
    cudaFree(next_field_gpu);
    cudaFree(current_field_padded_gpu);
    free(kernel);
    cudaFreeHost(result);
}