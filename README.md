# mersenne2GPU

OpenCL search for large Mersenne primes using small Mersenne primes and an IBDWT/NTT approach.

## About

This project is an OpenCL implementation inspired by `mersenne2.cpp` originally written by Yves Gallot:

https://github.com/galloty/mersenne2

The main idea is to perform large Mersenne arithmetic with an irrational base discrete weighted transform over finite fields built from small Mersenne primes.

This version uses two fields:
- GF((2^61 - 1)^2)
- GF((2^31 - 1)^2)

Using both moduli gives 92-bit output values through CRT and helps reduce transform sizes for large exponents.

Weights are powers of two, so weighting and unweighting are implemented as shifts.

This code currently supports GPU execution through OpenCL and includes Lucas-Lehmer and PRP style workflows.

## Credit

Initial idea and reference implementation:
- Yves Gallot
- https://github.com/galloty/mersenne2

This GPU version:
- Copyright 2026, cherubrock

## Build

```bash
g++ -O3 -std=c++20 -march=native mersenne2_gpu.cpp -o mersenne2 $(pkg-config --cflags --libs OpenCL)
```

If your local file is still named differently, for example:

```bash
g++ -O3 -std=c++20 -march=native mersenne2_gpu_ll_prp_marin_style_v21b_twiddle_packed3_optimized_v13_fused_center.cpp -o mersenne2 $(pkg-config --cflags --libs OpenCL)
```

## Run

Basic usage:

```bash
./mersenne2 <p> [ll|prp] [report_every] [-d device_id] [-R report_seconds] [-W wg] [-B carry_pairs]
```

Examples:

```bash
./mersenne2 9941 prp -d 0 -R 2
./mersenne2 19937 prp -d 0 -R 2
./mersenne2 132049 prp -d 0 -R 2
./mersenne2 57885161 prp -d 0 -R 2
```

## Notes

Transform size depends on the exponent range. In this implementation the mapping follows the table embedded in the source code.

Example:
- p in 9437189-18874367 uses N = 524288
- p in 18874379-36700159 uses N = 1048576

## Status

This repository is focused on:
- OpenCL GPU execution
- fast weighted transforms over small Mersenne prime fields
- large exponent Mersenne testing

## License spirit

This code follows the same free source spirit as the original work.
Use, modify, improve, and share. Feedback is welcome.
