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

## Transform ranges

```text
Prime exponent p range   N            Structure
----------------------------------------------------
3-173                    4            2^2
179-349                  8            2^3
353-683                  16           2^4
691-1373                 32           2^5
1381-2687                64           2^6
2689-5351                128          2^7
5381-10487               256          2^8
10499-20983              512          2^9
21001-40949              1024         2^10
40961-81919              2048         2^11
81929-159739             4096         2^12
159763-319483            8192         2^13
319489-622577            16384        2^14
622603-1245169           32768        2^15
1245187-2424827          65536        2^16
2424833-4849651          131072       2^17
4849687-9437179          262144       2^18
9437189-18874367         524288       2^19
18874379-36700159        1048576      2^20
36700201-73400311        2097152      2^21
73400329-142606333       4194304      2^22
142606357-285212659      8388608      2^23
285212677-553648103      16777216     2^24
553648171-1107296251     33554432     2^25
1107296257-2147483647    67108864     2^26
2147483659-4294967291    134217728    2^27
4294967311-8321499089    268435456    2^28
8321499143-16642998269   536870912    2^29
16642998289-32212254719  1073741824   2^30
```

## Status

This repository is focused on:
- OpenCL GPU execution
- fast weighted transforms over small Mersenne prime fields
- large exponent Mersenne testing

## License spirit

This code follows the same free source spirit as the original work.
Use, modify, improve, and share. Feedback is welcome.
