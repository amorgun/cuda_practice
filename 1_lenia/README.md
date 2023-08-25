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