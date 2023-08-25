#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>

extern char *optarg;
extern int optopt;


static inline float bell(const float val, const float mean, const float std)
{
    const float t = (val - mean) / std;
    return expf(-(t * t) / 2.f);
}

static inline float clamp(float d, float min, float max) {
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

void make_step(
        float const *const restrict current_state, float *const restrict next_state, const size_t size,
        float const *const restrict kernel, const size_t kernel_size,
        const float dT, const float growth_mean, const float growth_std
)
{
    const size_t R = (kernel_size - 1) / 2;
    const size_t PADDED_SIZE = size + 2 * R;
    float * restrict padded = calloc(PADDED_SIZE * PADDED_SIZE, sizeof(float));
    for (size_t idx = 0; idx < size; ++idx) {
        memcpy(&padded[(idx + R) * PADDED_SIZE + R], &current_state[idx * size], size * sizeof(float));
        memcpy(&padded[(idx + R) * PADDED_SIZE], &current_state[idx * size + size - R], R * sizeof(float));
        memcpy(&padded[(idx + R) * PADDED_SIZE + R + size], &current_state[idx * size], R * sizeof(float));
    }
    for (size_t idx = 0; idx < R; ++idx) {
        memcpy(&padded[idx * PADDED_SIZE], &padded[(idx + size) * PADDED_SIZE], PADDED_SIZE * sizeof(float));
        memcpy(&padded[(idx + R + size)* PADDED_SIZE], &padded[(idx + R) * PADDED_SIZE], PADDED_SIZE * sizeof(float));
    }
    
    for (size_t ki = 0; ki < kernel_size; ++ki) {
        const size_t ki_offset = ki * kernel_size;
        for (size_t kj = 0; kj < kernel_size; ++kj) {
            const float k = kernel[ki_offset + kj];
            for (size_t i = 0; i < size; ++i) {
                const size_t i_offset = i * size;
                const size_t src_i_offset = (i + ki) * PADDED_SIZE + kj;
                for (size_t j = 0; j < size; j += 8) {
                    const size_t dst_offset = i_offset + j;
                    const size_t src_offset = src_i_offset + j;
                    next_state[dst_offset] += k * padded[src_offset];
                    next_state[dst_offset + 1] += k * padded[src_offset + 1];
                    next_state[dst_offset + 2] += k * padded[src_offset + 2];
                    next_state[dst_offset + 3] += k * padded[src_offset + 3];
                    next_state[dst_offset + 4] += k * padded[src_offset + 4];
                    next_state[dst_offset + 5] += k * padded[src_offset + 5];
                    next_state[dst_offset + 6] += k * padded[src_offset + 6];
                    next_state[dst_offset + 7] += k * padded[src_offset + 7];
                }
            }
        }
    }
    for (size_t i = 0; i < size; ++i) {
        const size_t i_offset = i * size;
        for (size_t j = 0; j < size; ++j) {
            const size_t idx = i_offset + j;
            next_state[idx] = clamp(current_state[idx] + dT * (bell(next_state[idx], growth_mean, growth_std) * 2 - 1), 0, 1);
        }
    }

    free(padded);
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
        result = calloc(RESULT_DATA_SIZE, sizeof(float));
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

    float *const kernel = calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(float));
    init_kernel(kernel, KERNEL_SIZE, R, KERNEL_MEAN, KERNEL_STD);

    const clock_t time_start = clock(); 
    for (size_t it = 0; it < nsteps; ++it) {
        float const *const current_state = &result[it * FIELD_DATA_SIZE];
        float *const next_state = &result[(it+1) * FIELD_DATA_SIZE];
        make_step(current_state, next_state, FIELD_SIZE, kernel, KERNEL_SIZE, dT, GROWSH_MEAN, GROWSH_STD);
    }
    const clock_t time_end = clock();
    const double time_diff_ms = (time_end - time_start) * 1000. / CLOCKS_PER_SEC;
    fprintf(stderr, "Program took %.4f ms\n%.8f ms per iteration\n", time_diff_ms, time_diff_ms / nsteps);
    FILE *outFile = fopen(output_path, "wb");
    if (outFile != NULL) {
        fwrite(&FIELD_SIZE, sizeof(size_t), 1, outFile);
        fwrite(&nsteps, sizeof(size_t), 1, outFile);
        fwrite(result, RESULT_DATA_SIZE * sizeof(float), 1, outFile);
        fclose(outFile);
    }
    free(kernel);
    free(result);
}