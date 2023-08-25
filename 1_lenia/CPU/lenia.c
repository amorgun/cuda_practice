#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

extern char *optarg;
extern int optopt;


float bell(const float val, const float mean, const float std)
{
    const float t = (val - mean) / std;
    return expf(-(t * t) / 2.f);
}

void init_kernel(float *const kernel, const size_t size, const size_t R, const float mean, const float std)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            double res = 0;
            double dist2 = hypot((double)i - R, (double)j - R);
            if (dist2 > R) {
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


void init_state(float *const state, const size_t size)
{}

void make_step(float const *const current_state, float *const next_state, const size_t size)
{}

int main(int argc, char *argv[])
{
    size_t size = 64;
    size_t nsteps = 32;
    char const *dst_path = "output.bin";

    int c;
    while ((c = getopt(argc, argv, "n:s:o:")) != -1)
    {
        switch (c)
        {
            char *endptr;
            case 'n':
                size = strtoul(optarg, &endptr, 10);
                break;
            case 's':
                nsteps = strtoul(optarg, &endptr, 10);
                break;
            case 'o':
                dst_path = optarg;
                break;
            case '?':
                printf("Unknown option `-%c'.\n", optopt);
                return 1;
            default:
                abort();
        }
    }

    const size_t FIELD_SIZE = size;
    const size_t FIELD_DATA_SIZE = FIELD_SIZE * FIELD_SIZE;
    const size_t RESULT_DATA_SIZE = FIELD_DATA_SIZE * (nsteps + 1);

    const float dT = 0.1;
    const size_t R = 13;  // cells per kernel radius
    const size_t T = 10;  // steps per unit time
    const float GROWSH_MEAN = 0.15;
    const float GROWSH_STD = 0.015;

    const float KERNEL_MEAN = 0.5;
    const float KERNEL_STD = 0.15;
    const size_t KERNEL_SIZE = 2 * R + 1;
    // "b": [1],

    float (*const result)[FIELD_DATA_SIZE] = malloc(RESULT_DATA_SIZE * sizeof(float));
    init_state(result[0], FIELD_SIZE);
    float *const kernel = malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    init_kernel(kernel, KERNEL_SIZE, R, KERNEL_MEAN, KERNEL_STD);
    for (size_t it = 0; it < nsteps; ++it) {
        float const *const current_state = result[it];
        float *const next_state = result[(it+1)];
        make_step((float const *)current_state, next_state, FIELD_SIZE);
    }
    printf("CALL WITH %zu size and %zu steps: %s\n", size, nsteps, dst_path);
    free(result);
}