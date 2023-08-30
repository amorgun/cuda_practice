# Lenia implementation
This is an implementation of [Lenia](https://chakazul.github.io/lenia.html).

## Comparison

### CPU
Tested on 256x256 field and 1024 iterations.

| Name | Time per iteration | Speedup |
| ----- | ----- | ----- |
| Naive approach | 45.8277 ms | x1 |
| +SIMD | 7.6579 ms | x5.98 |
| +OpenMP (8 threads) | 1.8397 ms | x24.91 |


### CUDA
Tested on 256x256 field and 4096 iterations.

| Name | Time per iteration | Speedup |
| ----- | ----- | ----- |
| CPU baseline | 1.8397 ms | x1 |
| CUDA naive | 0.2594 ms | x7.09 |
| +async | 0.1558 ms | x11.8 |