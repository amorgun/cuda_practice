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

__host__ void pad_field(
    float const *const __restrict__ src, float *const __restrict__ dst, const size_t src_size, const size_t pad_size, cudaStream_t streams[3]
)
{
    const size_t PADDED_SIZE = src_size + 2 * pad_size;
    gpuErrchk(cudaMemcpy2DAsync(&dst[pad_size * PADDED_SIZE + pad_size], PADDED_SIZE * sizeof(float), src, src_size * sizeof(float), src_size * sizeof(float), src_size, cudaMemcpyDeviceToDevice, streams[0]));
    gpuErrchk(cudaMemcpy2DAsync(&dst[pad_size * PADDED_SIZE], PADDED_SIZE * sizeof(float), &src[src_size - pad_size], src_size * sizeof(float), pad_size * sizeof(float), src_size, cudaMemcpyDeviceToDevice, streams[1]));
    gpuErrchk(cudaMemcpy2DAsync(&dst[pad_size * PADDED_SIZE + pad_size + src_size], PADDED_SIZE * sizeof(float), src, src_size * sizeof(float), pad_size * sizeof(float), src_size, cudaMemcpyDeviceToDevice, streams[2]));
    gpuErrchk(cudaStreamSynchronize(streams[0]));
    gpuErrchk(cudaStreamSynchronize(streams[1]));
    gpuErrchk(cudaStreamSynchronize(streams[2]));
    gpuErrchk(cudaMemcpyAsync(&dst[0], &dst[src_size * PADDED_SIZE], PADDED_SIZE * pad_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]));
    gpuErrchk(cudaMemcpyAsync(&dst[(pad_size + src_size)* PADDED_SIZE], &dst[pad_size * PADDED_SIZE], PADDED_SIZE * pad_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[1]));
    gpuErrchk(cudaStreamSynchronize(streams[0]));
    gpuErrchk(cudaStreamSynchronize(streams[1]));
}

__global__ void make_step(
        cudaTextureObject_t current_state,
        float *const __restrict__ next_state, const size_t size,
        float const *const __restrict__ kernel, const size_t kernel_size,
        const float dT, const float growth_mean, const float growth_std
)
{
    extern __shared__ char shared_data[];
    float * const __restrict__ kernel_shared = (float*)shared_data;

    const size_t thread_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t thread_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (thread_pos_x >= size || thread_pos_y >= size) {
        return;
    }
    const size_t BLOCK_SIZE_FLAT = blockDim.x * blockDim.y;
    const size_t KERNEL_DATA_SIZE = kernel_size * kernel_size;
    const size_t KERNEL_NCOPY = (KERNEL_DATA_SIZE + BLOCK_SIZE_FLAT - 1) / BLOCK_SIZE_FLAT;
    const size_t THREAD_IDX = threadIdx.x + blockDim.x * threadIdx.y;
    for (size_t i = 0, p = THREAD_IDX * KERNEL_NCOPY; i < KERNEL_NCOPY && p < KERNEL_DATA_SIZE; ++i, ++p) {
        kernel_shared[p] = kernel[p];
    }

    __syncthreads();

    float next_state_val = 0;
    const size_t R = (kernel_size - 1) / 2;
    for (size_t ki = 0; ki < kernel_size; ++ki) {
        const size_t ki_offset = ki * kernel_size;
        for (size_t kj = 0; kj < kernel_size; ++kj) {
            next_state_val += kernel_shared[ki_offset + kj] * tex2D<float>(current_state, thread_pos_x + kj + size - R, thread_pos_y + ki + size - R);
        }
    }

    next_state_val = clamp(tex2D<float>(current_state, thread_pos_x, thread_pos_y) + 0 * dT * (bell(next_state_val, growth_mean, growth_std) * 2 - 1), 0, 1);
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

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 

    cudaArray *texArray;
    cudaMallocArray(&texArray, &channelDesc, FIELD_SIZE, FIELD_SIZE);

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

    dim3 blocks_shape = dim3(24, 32);
    dim3 grid_shape  = dim3((FIELD_SIZE + blocks_shape.x - 1) / blocks_shape.x, (FIELD_SIZE + blocks_shape.y - 1) / blocks_shape.y);

    cudaStream_t pad_streams[3], kernel_stream, copy_stream;
    gpuErrchk(cudaStreamCreate(&pad_streams[0]));
    gpuErrchk(cudaStreamCreate(&pad_streams[1]));
    gpuErrchk(cudaStreamCreate(&pad_streams[2]));
    gpuErrchk(cudaStreamCreate(&kernel_stream));
    gpuErrchk(cudaStreamCreate(&copy_stream));

    for (size_t it = 0; it < nsteps; ++it) {
        gpuErrchk(cudaMemcpy2DToArrayAsync(texArray, 0, 0, current_field_gpu, FIELD_SIZE * sizeof(float), FIELD_SIZE * sizeof(float), FIELD_SIZE, cudaMemcpyDeviceToDevice, kernel_stream));
        
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = texArray;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        gpuErrchk(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));


        float *const next_state = &result[(it+1) * FIELD_DATA_SIZE];
        make_step<<<grid_shape, blocks_shape, KERNEL_SIZE*KERNEL_SIZE*sizeof(int), kernel_stream>>>(texObj, next_field_gpu, FIELD_SIZE, kernel_gpu, KERNEL_SIZE, dT, GROWSH_MEAN, GROWSH_STD);
        gpuErrchk(cudaStreamSynchronize(kernel_stream));
        gpuErrchk(cudaStreamSynchronize(copy_stream));
        gpuErrchk(cudaMemcpyAsync(next_state, next_field_gpu, FIELD_DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
        float *tmp = current_field_gpu;
        current_field_gpu = next_field_gpu;
        next_field_gpu = tmp;
        gpuErrchk(cudaMemsetAsync(next_field_gpu, 0, FIELD_DATA_SIZE * sizeof(float), kernel_stream));
        gpuErrchk(cudaDestroyTextureObject(texObj));
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