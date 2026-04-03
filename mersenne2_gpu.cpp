/*
 * Copyright 2026, cherubrock
 *
 * This program is an OpenCL implementation inspired by "mersenne2.cpp" 
 * originally written by Yves Gallot (Copyright 2025). 
 * https://github.com/galloty/mersenne2
 * The original "mersenne2.cpp" was released as free source code. This version 
 * inherits the same spirit: it is free to use, modify, and redistribute.
 *
 * If you make improvements, please consider giving feedback to the author.
 * This code is provided in the hope that it will be useful.
 * Prime exponent p range   N            Structure
 * ----------------------------------------------------
 * 3-173                    4            2^2
 * 179-349                  8            2^3
 * 353-683                  16           2^4
 * 691-1373                 32           2^5
 * 1381-2687                64           2^6
 * 2689-5351                128          2^7
 * 5381-10487               256          2^8
 * 10499-20983              512          2^9
 * 21001-40949              1024         2^10
 * 40961-81919              2048         2^11
 * 81929-159739             4096         2^12
 * 159763-319483            8192         2^13
 * 319489-622577            16384        2^14
 * 622603-1245169           32768        2^15
 * 1245187-2424827          65536        2^16
 * 2424833-4849651          131072       2^17
 * 4849687-9437179          262144       2^18
 * 9437189-18874367         524288       2^19
 * 18874379-36700159        1048576      2^20
 * 36700201-73400311        2097152      2^21
 * 73400329-142606333       4194304      2^22
 * 142606357-285212659      8388608      2^23
 * 285212677-553648103      16777216     2^24
 * 553648171-1107296251     33554432     2^25
 * 1107296257-2147483647    67108864     2^26
 * 2147483659-4294967291    134217728    2^27
 * 4294967311-8321499089    268435456    2^28
 * 8321499143-16642998269   536870912    2^29
 * 16642998289-32212254719  1073741824   2^30
 */
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static const uint64_t Z61_p = (uint64_t(1) << 61) - 1;
static const uint32_t Z31_p = (uint32_t(1) << 31) - 1;

class Z61 {
    uint64_t _n;
    static uint64_t add(uint64_t a, uint64_t b) {
        uint64_t t = a + b;
        return t - (t >= Z61_p ? Z61_p : 0);
    }
    static uint64_t sub(uint64_t a, uint64_t b) {
        uint64_t t = a - b;
        return t + (a < b ? Z61_p : 0);
    }
    static uint64_t mul(uint64_t a, uint64_t b) {
        __uint128_t t = __uint128_t(a) * b;
        return add(uint64_t(t) & Z61_p, uint64_t(t >> 61));
    }
    static uint64_t lshift_raw(uint64_t a, uint8_t s) {
        __uint128_t t = __uint128_t(a) << s;
        return add(uint64_t(t) & Z61_p, uint64_t(t >> 61));
    }
    static uint64_t norm(uint64_t n) {
        uint64_t lo = n & Z61_p;
        uint64_t hi = n >> 61;
        uint64_t r = add(lo, hi);
        return (r == Z61_p) ? 0 : r;
    }
public:
    Z61() : _n(0) {}
    explicit Z61(uint64_t n) : _n(norm(n)) {}
    uint64_t get() const { return _n; }
    Z61 operator+(const Z61& o) const { return Z61(add(_n, o._n)); }
    Z61 operator-(const Z61& o) const { return Z61(sub(_n, o._n)); }
    Z61 operator*(const Z61& o) const { return Z61(mul(_n, o._n)); }
    Z61 sqr() const { return Z61(mul(_n, _n)); }
    Z61 lshift(int s) const {
        int ss = s;
        while (ss >= 61) ss -= 61;
        while (ss < 0) ss += 61;
        return (ss == 0) ? *this : Z61(lshift_raw(_n, static_cast<uint8_t>(ss)));
    }
    Z61 rshift(int s) const {
        int ss = s;
        while (ss >= 61) ss -= 61;
        while (ss < 0) ss += 61;
        return (ss == 0) ? *this : Z61(lshift_raw(_n, static_cast<uint8_t>(61 - ss)));
    }
};

class Z31 {
    uint32_t _n;
    static uint32_t add(uint32_t a, uint32_t b) {
        uint32_t t = a + b;
        return t - (t >= Z31_p ? Z31_p : 0);
    }
    static uint32_t sub(uint32_t a, uint32_t b) {
        uint32_t t = a - b;
        return t + (a < b ? Z31_p : 0);
    }
    static uint32_t mul(uint32_t a, uint32_t b) {
        uint64_t t = uint64_t(a) * b;
        return add(uint32_t(t) & Z31_p, uint32_t(t >> 31));
    }
    static uint32_t lshift_raw(uint32_t a, uint8_t s) {
        uint64_t t = uint64_t(a) << s;
        return add(uint32_t(t) & Z31_p, uint32_t(t >> 31));
    }
    static uint32_t norm(uint64_t n) {
        uint32_t lo = uint32_t(n) & Z31_p;
        uint32_t hi = uint32_t(n >> 31);
        uint32_t r = add(lo, hi);
        return (r == Z31_p) ? 0u : r;
    }
public:
    Z31() : _n(0) {}
    explicit Z31(uint32_t n) : _n(norm(n)) {}
    explicit Z31(uint64_t n) : _n(norm(n)) {}
    uint32_t get() const { return _n; }
    Z31 operator+(const Z31& o) const { return Z31(add(_n, o._n)); }
    Z31 operator-(const Z31& o) const { return Z31(sub(_n, o._n)); }
    Z31 operator*(const Z31& o) const { return Z31(mul(_n, o._n)); }
    Z31 sqr() const { return Z31(mul(_n, _n)); }
    Z31 lshift(int s) const {
        int ss = s;
        while (ss >= 31) ss -= 31;
        while (ss < 0) ss += 31;
        return (ss == 0) ? *this : Z31(lshift_raw(_n, static_cast<uint8_t>(ss)));
    }
    Z31 rshift(int s) const {
        int ss = s;
        while (ss >= 31) ss -= 31;
        while (ss < 0) ss += 31;
        return (ss == 0) ? *this : Z31(lshift_raw(_n, static_cast<uint8_t>(31 - ss)));
    }
};

class GF61 {
    Z61 _a, _b;
    static const uint64_t ORDER = uint64_t(1) << 62;
    static const uint64_t H0 = 264036120304204ull, H1 = 4677669021635377ull;
public:
    GF61() : _a(0), _b(0) {}
    GF61(const Z61& a, const Z61& b): _a(a), _b(b) {}
    GF61(uint64_t a, uint64_t b): _a(a), _b(b) {}
    uint64_t s0() const { return _a.get(); }
    uint64_t s1() const { return _b.get(); }
    GF61 operator+(const GF61& o) const { return GF61(_a + o._a, _b + o._b); }
    GF61 operator-(const GF61& o) const { return GF61(_a - o._a, _b - o._b); }
    GF61 addconj(const GF61& o) const { return GF61(_a + o._a, _b - o._b); }
    GF61 subconj(const GF61& o) const { return GF61(_a - o._a, _b + o._b); }
    GF61 sub_conj(const GF61& o) const { return GF61(_a - o._a, o._b - _b); }
    GF61 addi(const GF61& o) const { return GF61(_a - o._b, _b + o._a); }
    GF61 subi(const GF61& o) const { return GF61(_a + o._b, _b - o._a); }
    GF61 mul(const GF61& o) const { return GF61(_a * o._a - _b * o._b, _b * o._a + _a * o._b); }
    GF61 mulconj(const GF61& o) const { return GF61(_a * o._a + _b * o._b, _b * o._a - _a * o._b); }
    GF61 sqr() const {
        Z61 t = _a * _b;
        return GF61(_a.sqr() - _b.sqr(), t + t);
    }
    GF61 lshift(uint8_t ls0, uint8_t ls1) const { return GF61(_a.lshift(ls0), _b.lshift(ls1)); }
    GF61 rshift(uint8_t rs0, uint8_t rs1) const { return GF61(_a.rshift(rs0), _b.rshift(rs1)); }
    GF61 pow(uint64_t e) const {
        if (e == 0) return GF61(1u, 0u);
        GF61 r(1u, 0u), y = *this;
        for (uint64_t i = e; i != 1; i /= 2) { if (i & 1ull) r = r.mul(y); y = y.sqr(); }
        return r.mul(y);
    }
    static GF61 root_nth(size_t n) { return GF61(H0, H1).pow(ORDER / n); }
    static uint8_t log2_root_two(size_t n) { return uint8_t(((uint64_t(1) << 60) / n) % 61); }
};

class GF31 {
    Z31 _a, _b;
    static const uint64_t ORDER = uint64_t(1) << 32;
    static const uint32_t H0 = 7735u, H1 = 748621u;
public:
    GF31() : _a(0u), _b(0u) {}
    GF31(const Z31& a, const Z31& b): _a(a), _b(b) {}
    GF31(uint64_t a, uint64_t b): _a(a), _b(b) {}
    uint32_t s0() const { return _a.get(); }
    uint32_t s1() const { return _b.get(); }
    GF31 operator+(const GF31& o) const { return GF31(_a + o._a, _b + o._b); }
    GF31 operator-(const GF31& o) const { return GF31(_a - o._a, _b - o._b); }
    GF31 addconj(const GF31& o) const { return GF31(_a + o._a, _b - o._b); }
    GF31 subconj(const GF31& o) const { return GF31(_a - o._a, _b + o._b); }
    GF31 sub_conj(const GF31& o) const { return GF31(_a - o._a, o._b - _b); }
    GF31 addi(const GF31& o) const { return GF31(_a - o._b, _b + o._a); }
    GF31 subi(const GF31& o) const { return GF31(_a + o._b, _b - o._a); }
    GF31 mul(const GF31& o) const { return GF31(_a * o._a - _b * o._b, _b * o._a + _a * o._b); }
    GF31 mulconj(const GF31& o) const { return GF31(_a * o._a + _b * o._b, _b * o._a - _a * o._b); }
    GF31 sqr() const {
        Z31 t = _a * _b;
        return GF31(_a.sqr() - _b.sqr(), t + t);
    }
    GF31 lshift(uint8_t ls0, uint8_t ls1) const { return GF31(_a.lshift(ls0), _b.lshift(ls1)); }
    GF31 rshift(uint8_t rs0, uint8_t rs1) const { return GF31(_a.rshift(rs0), _b.rshift(rs1)); }
    GF31 pow(uint64_t e) const {
        if (e == 0) return GF31(1u, 0u);
        GF31 r(1u, 0u), y = *this;
        for (uint64_t i = e; i != 1; i /= 2) { if (i & 1ull) r = r.mul(y); y = y.sqr(); }
        return r.mul(y);
    }
    static GF31 root_nth(size_t n) { return GF31(H0, H1).pow(ORDER / n); }
    static uint8_t log2_root_two(size_t n) { return uint8_t(((uint64_t(1) << 30) / n) % 31); }
};

class GF61_31 {
public:
    GF61 g61;
    GF31 g31;
    GF61_31() : g61(), g31() {}
    explicit GF61_31(uint32_t n) : g61(n, 0u), g31(n, 0u) {}
    GF61_31(uint64_t a0, uint64_t a1) : g61(a0, a1), g31(a0, a1) {}
    GF61_31(const GF61& a, const GF31& b): g61(a), g31(b) {}
    GF61_31 operator+(const GF61_31& o) const { return GF61_31(g61 + o.g61, g31 + o.g31); }
    GF61_31 operator-(const GF61_31& o) const { return GF61_31(g61 - o.g61, g31 - o.g31); }
    GF61_31 addconj(const GF61_31& o) const { return GF61_31(g61.addconj(o.g61), g31.addconj(o.g31)); }
    GF61_31 subconj(const GF61_31& o) const { return GF61_31(g61.subconj(o.g61), g31.subconj(o.g31)); }
    GF61_31 sub_conj(const GF61_31& o) const { return GF61_31(g61.sub_conj(o.g61), g31.sub_conj(o.g31)); }
    GF61_31 addi(const GF61_31& o) const { return GF61_31(g61.addi(o.g61), g31.addi(o.g31)); }
    GF61_31 subi(const GF61_31& o) const { return GF61_31(g61.subi(o.g61), g31.subi(o.g31)); }
    GF61_31 mul(const GF61_31& o) const { return GF61_31(g61.mul(o.g61), g31.mul(o.g31)); }
    GF61_31 mulconj(const GF61_31& o) const { return GF61_31(g61.mulconj(o.g61), g31.mulconj(o.g31)); }
    GF61_31 sqr() const { return GF61_31(g61.sqr(), g31.sqr()); }
    GF61_31 pow(uint64_t e) const { return GF61_31(g61.pow(e), g31.pow(e)); }
    static GF61_31 root_nth(size_t n) { return GF61_31(GF61::root_nth(n), GF31::root_nth(n)); }
};

static inline uint64_t pack_u32x2(const uint32_t lo, const uint32_t hi) {
    return uint64_t(lo) | (uint64_t(hi) << 32);
}

static inline void pack_gf_words(std::vector<uint64_t>& dst, const size_t idx, const GF61_31& v) {
    dst[3 * idx + 0] = v.g61.s0();
    dst[3 * idx + 1] = v.g61.s1();
    dst[3 * idx + 2] = pack_u32x2(v.g31.s0(), v.g31.s1());
}

static inline GF61_31 unpack_gf_words(const std::vector<uint64_t>& src, const size_t idx) {
    const uint64_t p = src[3 * idx + 2];
    return GF61_31(GF61(src[3 * idx + 0], src[3 * idx + 1]), GF31(uint32_t(p), uint32_t(p >> 32)));
}

struct IBWeight { uint8_t w61, w31; };

static size_t bitrev(size_t i, size_t n) {
    size_t r = 0;
    for (; n > 1; n >>= 1, i >>= 1) r = (r << 1) | (i & 1);
    return r;
}

static uint8_t transformsize(const uint32_t q) {
    uint8_t ln = 2; uint32_t w = 0;
    do {
        ++ln;
        w = q >> ln;
    } while (ln + 2 * (w + 1) >= 61 + 31);
    return ln;
}

const char* KC = R"CLC(
typedef ulong  u64;
typedef uint   u32;
typedef struct { ulong lo, hi; } u128;
typedef struct { u64 s0, s1; u32 t0, t1; } GF;
typedef struct { ulong l0_lo, l0_hi, l1_lo, l1_hi; } RawPair;
typedef struct { uchar w61, w31; } IW;

#define P61 (((u64)1<<61)-1)
#define P31 (((u32)1<<31)-1)

inline u64 pack_u32x2_dev(const u32 lo, const u32 hi) { return ((u64)lo) | (((u64)hi) << 32); }

inline u128 make_u128(ulong lo, ulong hi){ u128 r = { lo, hi }; return r; }
inline u128 add_u128(u128 a, u128 b){ ulong lo = a.lo + b.lo; ulong carry = (lo < a.lo); return make_u128(lo, a.hi + b.hi + carry); }
inline u128 shl_u128(u128 a, uint w){
    if (w == 0) return a;
    if (w < 64) return make_u128(a.lo << w, (a.hi << w) | (a.lo >> (64 - w)));
    return make_u128(0ul, a.lo << (w - 64));
}
inline u128 rshift_u128(u128 a, uint w){
    if (w == 0) return a;
    if (w < 64) return make_u128((a.lo >> w) | (a.hi << (64 - w)), a.hi >> w);
    return make_u128(a.hi >> (w - 64), 0ul);
}

inline u128 mul_64x64_128(ulong a, ulong b) {
    ulong a_lo = (uint)a, a_hi = a >> 32;
    ulong b_lo = (uint)b, b_hi = b >> 32;
    ulong lo_lo = a_lo * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_lo = a_hi * b_lo;
    ulong hi_hi = a_hi * b_hi;
    ulong cross = lo_hi + hi_lo;
    ulong cross_lo = cross << 32;
    ulong cross_hi = cross >> 32;
    ulong lo = lo_lo + cross_lo;
    ulong carry = (lo < lo_lo) ? 1ul : 0ul;
    ulong hi = hi_hi + cross_hi + carry;
    return make_u128(lo, hi);
}

inline u64 add61(u64 a,u64 b){ u64 t = a + b; return t - (t >= P61 ? P61 : 0); }
inline u64 sub61(u64 a,u64 b){ u64 t = a - b; return t + (a < b ? P61 : 0); }
inline u64 mul61(u64 a, u64 b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    u64 p0 = (u64)a0 * (u64)b0;
    u64 p1 = (u64)a0 * (u64)b1 + (u64)a1 * (u64)b0;
    u64 p2 = (u64)a1 * (u64)b1;

    u64 lo = p0 + (p1 << 32);
    u64 carry = (lo < p0) ? 1ul : 0ul;
    u64 hi = p2 + (p1 >> 32) + carry;

    u64 r = (lo & P61) + (lo >> 61) + (hi << 3);
    r = (r & P61) + (r >> 61);
    r = (r & P61) + (r >> 61);
    return (r >= P61) ? (r - P61) : r;
}
inline uint mod_u61_small(uint s) {
    s -= (s >= 61u) ? 61u : 0u;
    s -= (s >= 61u) ? 61u : 0u;
    return s;
}
inline u64 lshift_mod61(u64 x, uint s) {
    s = mod_u61_small(s); if (s == 0) return x;
    u64 lo = (x << s) & P61;
    u64 hi = (x >> (61 - s));
    return add61(lo, hi);
}
inline u64 rshift_mod61(u64 x, uint s) { s = mod_u61_small(s); return (s == 0) ? x : lshift_mod61(x, 61u - s); }

inline u32 add31(u32 a,u32 b){ u32 t = a + b; return t - (t >= P31 ? P31 : 0); }
inline u32 sub31(u32 a,u32 b){ u32 t = a - b; return t + (a < b ? P31 : 0); }
inline u32 mul31(u32 a, u32 b) {
    u64 t = (u64)a * b;
    u32 lo = (u32)(t & P31);
    u32 hi = (u32)(t >> 31);
    return add31(lo, hi);
}
inline uint mod_u31_small(uint s) {
    s -= (s >= 31u) ? 31u : 0u;
    s -= (s >= 31u) ? 31u : 0u;
    s -= (s >= 31u) ? 31u : 0u;
    return s;
}
inline u32 lshift_mod31(u32 x, uint s) {
    s = mod_u31_small(s); if (s == 0) return x;
    u32 lo = (x << s) & P31;
    u32 hi = (x >> (31 - s));
    return add31(lo, hi);
}
inline u32 rshift_mod31(u32 x, uint s) { s = mod_u31_small(s); return (s == 0) ? x : lshift_mod31(x, 31u - s); }

inline GF gf_add(GF a,GF b){ GF r = { add61(a.s0,b.s0), add61(a.s1,b.s1), add31(a.t0,b.t0), add31(a.t1,b.t1) }; return r; }
inline GF gf_sub(GF a,GF b){ GF r = { sub61(a.s0,b.s0), sub61(a.s1,b.s1), sub31(a.t0,b.t0), sub31(a.t1,b.t1) }; return r; }
inline GF gf_addconj(GF a, GF b){ GF r = { add61(a.s0,b.s0), sub61(a.s1,b.s1), add31(a.t0,b.t0), sub31(a.t1,b.t1) }; return r; }
inline GF gf_subconj(GF a, GF b){ GF r = { sub61(a.s0,b.s0), add61(a.s1,b.s1), sub31(a.t0,b.t0), add31(a.t1,b.t1) }; return r; }
inline GF gf_sub_conj(GF a, GF b){ GF r = { sub61(a.s0,b.s0), sub61(b.s1,a.s1), sub31(a.t0,b.t0), sub31(b.t1,a.t1) }; return r; }
inline GF gf_addi(GF a, GF b){ GF r = { sub61(a.s0,b.s1), add61(a.s1,b.s0), sub31(a.t0,b.t1), add31(a.t1,b.t0) }; return r; }
inline GF gf_subi(GF a, GF b){ GF r = { add61(a.s0,b.s1), sub61(a.s1,b.s0), add31(a.t0,b.t1), sub31(a.t1,b.t0) }; return r; }
inline GF gf_mul(GF a,GF b){
    const u64 ac61 = mul61(a.s0, b.s0);
    const u64 bd61 = mul61(a.s1, b.s1);
    const u64 k61  = mul61(add61(a.s0, a.s1), add61(b.s0, b.s1));

    const u32 ac31 = mul31(a.t0, b.t0);
    const u32 bd31 = mul31(a.t1, b.t1);
    const u32 k31  = mul31(add31(a.t0, a.t1), add31(b.t0, b.t1));

    GF r = {
        sub61(ac61, bd61),
        sub61(sub61(k61, ac61), bd61),
        sub31(ac31, bd31),
        sub31(sub31(k31, ac31), bd31)
    };
    return r;
}
inline GF gf_mulconj(GF a, GF b){
    const u64 ac61 = mul61(a.s0, b.s0);
    const u64 bd61 = mul61(a.s1, b.s1);
    const u64 k61  = mul61(add61(a.s0, a.s1), sub61(b.s0, b.s1));

    const u32 ac31 = mul31(a.t0, b.t0);
    const u32 bd31 = mul31(a.t1, b.t1);
    const u32 k31  = mul31(add31(a.t0, a.t1), sub31(b.t0, b.t1));

    GF r = {
      add61(ac61, bd61),
      add61(sub61(k61, ac61), bd61),
      add31(ac31, bd31),
      add31(sub31(k31, ac31), bd31)
    };
    return r;
}
inline GF gf_sqr(GF a){
    u64 t61 = mul61(a.s0, a.s1);
    u32 t31 = mul31(a.t0, a.t1);
    GF r = {
        sub61(mul61(a.s0, a.s0), mul61(a.s1, a.s1)),
        add61(t61, t61),
        sub31(mul31(a.t0, a.t0), mul31(a.t1, a.t1)),
        add31(t31, t31)
    };
    return r;
}
inline GF lshift_GF(GF z, uint ls0, uint ls1, uint lt0, uint lt1) {
    GF r = { lshift_mod61(z.s0, ls0), lshift_mod61(z.s1, ls1), lshift_mod31(z.t0, lt0), lshift_mod31(z.t1, lt1) }; return r;
}
inline GF lshift_GF_small31_words(const u64 s0, const u64 s1, const uint ls0, const uint ls1, const uint lt0, const uint lt1) {
    GF r = { lshift_mod61(s0, ls0), lshift_mod61(s1, ls1), lshift_mod31((u32)s0, lt0), lshift_mod31((u32)s1, lt1) };
    return r;
}
inline GF rshift_GF(GF z, uint rs0, uint rs1, uint rt0, uint rt1) {
    GF r = { rshift_mod61(z.s0, rs0), rshift_mod61(z.s1, rs1), rshift_mod31(z.t0, rt0), rshift_mod31(z.t1, rt1) }; return r;
}

inline ulong wload_raw(__global const ulong* restrict wbuf, int idx) { return wbuf[idx]; }
inline GF wload_gf(__global const ulong* restrict wbuf, int idx) {
    ulong p = wload_raw(wbuf, 3 * idx + 2);
    GF r = { wload_raw(wbuf, 3 * idx + 0), wload_raw(wbuf, 3 * idx + 1), (u32)p, (u32)(p >> 32) };
    return r;
}

#define WGLOAD(i) wload_gf(w, (i))

inline void forward4_stage_local(__local GF* restrict scratch, const int base, const int stride, const int lane,
                                 __global const ulong* restrict w, const int tw, const int n) {
    GF u0 = scratch[base + lane];
    GF u1 = gf_mul(scratch[base + stride + lane],     WGLOAD(2 * tw));
    GF u2 = gf_mul(scratch[base + 2 * stride + lane], WGLOAD(tw));
    GF u3 = gf_mul(scratch[base + 3 * stride + lane], WGLOAD((n >> 1) + tw));
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    scratch[base + lane]              = gf_add(v0, v1);
    scratch[base + stride + lane]     = gf_sub(v0, v1);
    scratch[base + 2 * stride + lane] = gf_addi(v2, v3);
    scratch[base + 3 * stride + lane] = gf_subi(v2, v3);
}

inline void backward4_stage_local(__local GF* restrict scratch, const int base, const int stride, const int lane,
                                  __global const ulong* restrict w, const int tw, const int n) {
    GF u0 = scratch[base + lane];
    GF u1 = scratch[base + stride + lane];
    GF u2 = scratch[base + 2 * stride + lane];
    GF u3 = scratch[base + 3 * stride + lane];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    scratch[base + lane]              = gf_add(v0, v2);
    scratch[base + 2 * stride + lane] = gf_mulconj(gf_sub(v0, v2), WGLOAD(tw));
    scratch[base + stride + lane]     = gf_mulconj(gf_addi(v1, v3), WGLOAD(2 * tw));
    scratch[base + 3 * stride + lane] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + tw));
}

inline void forward4_stage_local_w3(__local GF* restrict scratch, const int base, const int stride, const int lane,
                                    const GF w2, const GF w1, const GF wh) {
    GF u0 = scratch[base + lane];
    GF u1 = gf_mul(scratch[base + stride + lane],     w2);
    GF u2 = gf_mul(scratch[base + 2 * stride + lane], w1);
    GF u3 = gf_mul(scratch[base + 3 * stride + lane], wh);
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    scratch[base + lane]              = gf_add(v0, v1);
    scratch[base + stride + lane]     = gf_sub(v0, v1);
    scratch[base + 2 * stride + lane] = gf_addi(v2, v3);
    scratch[base + 3 * stride + lane] = gf_subi(v2, v3);
}

inline void backward4_stage_local_w3(__local GF* restrict scratch, const int base, const int stride, const int lane,
                                     const GF w2, const GF w1, const GF wh) {
    GF u0 = scratch[base + lane];
    GF u1 = scratch[base + stride + lane];
    GF u2 = scratch[base + 2 * stride + lane];
    GF u3 = scratch[base + 3 * stride + lane];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    scratch[base + lane]              = gf_add(v0, v2);
    scratch[base + 2 * stride + lane] = gf_mulconj(gf_sub(v0, v2), w1);
    scratch[base + stride + lane]     = gf_mulconj(gf_addi(v1, v3), w2);
    scratch[base + 3 * stride + lane] = gf_mulconj(gf_subi(v1, v3), wh);
}

__kernel void weight(__global GF* restrict z, __global const IW* restrict w) {
    uint i = get_global_id(0);
    z[i] = lshift_GF(z[i], w[2*i].w61, w[2*i+1].w61, w[2*i].w31, w[2*i+1].w31);
}

__kernel void weight_forward4_first(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int m, int n) {
    int i = (int)get_global_id(0);
    if (i >= m) return;
    GF u0 = lshift_GF(z[i],       iw[2*(uint)i].w61,       iw[2*(uint)i+1].w61,       iw[2*(uint)i].w31,       iw[2*(uint)i+1].w31);
    GF u1 = lshift_GF(z[m + i],   iw[2*(uint)(m+i)].w61,   iw[2*(uint)(m+i)+1].w61,   iw[2*(uint)(m+i)].w31,   iw[2*(uint)(m+i)+1].w31);
    GF u2 = lshift_GF(z[2*m + i], iw[2*(uint)(2*m+i)].w61, iw[2*(uint)(2*m+i)+1].w61, iw[2*(uint)(2*m+i)].w31, iw[2*(uint)(2*m+i)+1].w31);
    GF u3 = lshift_GF(z[3*m + i], iw[2*(uint)(3*m+i)].w61, iw[2*(uint)(3*m+i)+1].w61, iw[2*(uint)(3*m+i)].w31, iw[2*(uint)(3*m+i)+1].w31);
    u1 = gf_mul(u1, WGLOAD(2));
    u2 = gf_mul(u2, WGLOAD(1));
    u3 = gf_mul(u3, WGLOAD((n >> 1) + 1));
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    z[i]       = gf_add(v0, v1);
    z[m + i]   = gf_sub(v0, v1);
    z[2*m + i] = gf_addi(v2, v3);
    z[3*m + i] = gf_subi(v2, v3);
}


__kernel void weight_small31_refresh(__global GF* restrict z, __global const IW* restrict w) {
    uint i = get_global_id(0);
    __global const ulong* restrict zraw = (__global const ulong*)z;
    const uint base = 3u * i;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    z[i] = lshift_GF_small31_words(s0, s1, w[2*i].w61, w[2*i+1].w61, w[2*i].w31, w[2*i+1].w31);
}

__kernel void weight_forward4_first_small31_refresh(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int m, int n) {
    int i = (int)get_global_id(0);
    if (i >= m) return;
    __global const ulong* restrict zraw = (__global const ulong*)z;
    const uint b0 = 3u * (uint)i;
    const uint b1 = 3u * (uint)(m + i);
    const uint b2 = 3u * (uint)(2*m + i);
    const uint b3 = 3u * (uint)(3*m + i);
    GF u0 = lshift_GF_small31_words(zraw[b0 + 0u], zraw[b0 + 1u], iw[2*(uint)i].w61,       iw[2*(uint)i+1].w61,       iw[2*(uint)i].w31,       iw[2*(uint)i+1].w31);
    GF u1 = lshift_GF_small31_words(zraw[b1 + 0u], zraw[b1 + 1u], iw[2*(uint)(m+i)].w61,   iw[2*(uint)(m+i)+1].w61,   iw[2*(uint)(m+i)].w31,   iw[2*(uint)(m+i)+1].w31);
    GF u2 = lshift_GF_small31_words(zraw[b2 + 0u], zraw[b2 + 1u], iw[2*(uint)(2*m+i)].w61, iw[2*(uint)(2*m+i)+1].w61, iw[2*(uint)(2*m+i)].w31, iw[2*(uint)(2*m+i)+1].w31);
    GF u3 = lshift_GF_small31_words(zraw[b3 + 0u], zraw[b3 + 1u], iw[2*(uint)(3*m+i)].w61, iw[2*(uint)(3*m+i)+1].w61, iw[2*(uint)(3*m+i)].w31, iw[2*(uint)(3*m+i)+1].w31);
    u1 = gf_mul(u1, WGLOAD(2));
    u2 = gf_mul(u2, WGLOAD(1));
    u3 = gf_mul(u3, WGLOAD((n >> 1) + 1));
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    z[i]       = gf_add(v0, v1);
    z[m + i]   = gf_sub(v0, v1);
    z[2*m + i] = gf_addi(v2, v3);
    z[3*m + i] = gf_subi(v2, v3);
}

__kernel void forward2(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j = (int)get_global_id(0);
    if (j >= n4) return;
    GF u0 = z[2*j + 0];
    GF u1 = gf_mul(z[2*j + 1], WGLOAD(n4 + j));
    z[2*j + 0] = gf_add(u0, u1);
    z[2*j + 1] = gf_sub(u0, u1);
}

__kernel void backward2(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j = (int)get_global_id(0);
    if (j >= n4) return;
    GF u0 = z[2*j + 0];
    GF u1 = z[2*j + 1];
    z[2*j + 0] = gf_add(u0, u1);
    z[2*j + 1] = gf_mulconj(gf_sub(u0, u1), WGLOAD(n4 + j));
}


__kernel void forward2_x2(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j0 = (int)(get_global_id(0) * 2);
    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        GF u0 = z[2*j + 0];
        GF u1 = gf_mul(z[2*j + 1], WGLOAD(n4 + j));
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_sub(u0, u1);
    }
}

__kernel void forward2_x4(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j0 = (int)(get_global_id(0) * 4);
    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        GF u0 = z[2*j + 0];
        GF u1 = gf_mul(z[2*j + 1], WGLOAD(n4 + j));
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_sub(u0, u1);
    }
}

__kernel void backward2_x2(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j0 = (int)(get_global_id(0) * 2);
    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        GF u0 = z[2*j + 0];
        GF u1 = z[2*j + 1];
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_mulconj(gf_sub(u0, u1), WGLOAD(n4 + j));
    }
}

__kernel void backward2_x4(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int j0 = (int)(get_global_id(0) * 4);
    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        GF u0 = z[2*j + 0];
        GF u1 = z[2*j + 1];
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_mulconj(gf_sub(u0, u1), WGLOAD(n4 + j));
    }
}


__kernel void fused_center2_x2(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int blk = (int)(get_global_id(0) * 2);
    int j0 = blk;
    if (j0 >= n4) return;

    GF r[4];
    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        int base = 2 * t;
        GF a = z[2*j + 0];
        GF b = z[2*j + 1];
        GF tw = WGLOAD(n4 + j);
        GF u1 = gf_mul(b, tw);
        r[base + 0] = gf_add(a, u1);
        r[base + 1] = gf_sub(a, u1);
    }

    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        int k_local = 2 * t;
        int k = 2 * j;
        int mk = (k != 0) ? ((3u << (31 - clz((uint)k))) - k - 1) : 0;
        int mk_local = mk - 2 * j0;
        GF zk = r[k_local];
        GF zmk = r[mk_local];
        GF u0 = gf_addconj(zk, zmk);
        GF u1 = gf_subconj(zk, zmk);
        GF v0 = gf_sub(gf_sqr(u0), gf_mul(gf_sqr(u1), WGLOAD(n4 + j)));
        GF v1 = gf_mul(u0, gf_add(u1, u1));
        r[k_local] = gf_add(v0, v1);
        if (k == 0) r[1] = gf_sqr(gf_add(r[1], r[1]));
        else r[mk_local] = gf_sub_conj(v0, v1);
    }

    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        int base = 2 * t;
        GF tw = WGLOAD(n4 + j);
        GF u0 = r[base + 0];
        GF u1 = r[base + 1];
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_mulconj(gf_sub(u0, u1), tw);
    }
}

__kernel void fused_center2_x4(__global GF* restrict z, __global const ulong* restrict w, const int n4) {
    int blk = (int)(get_global_id(0) * 4);
    int j0 = blk;
    if (j0 >= n4) return;

    GF r[8];
    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        int base = 2 * t;
        GF a = z[2*j + 0];
        GF b = z[2*j + 1];
        GF tw = WGLOAD(n4 + j);
        GF u1 = gf_mul(b, tw);
        r[base + 0] = gf_add(a, u1);
        r[base + 1] = gf_sub(a, u1);
    }

    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        int k_local = 2 * t;
        int k = 2 * j;
        int mk = (k != 0) ? ((3u << (31 - clz((uint)k))) - k - 1) : 0;
        int mk_local = mk - 2 * j0;
        GF zk = r[k_local];
        GF zmk = r[mk_local];
        GF u0 = gf_addconj(zk, zmk);
        GF u1 = gf_subconj(zk, zmk);
        GF v0 = gf_sub(gf_sqr(u0), gf_mul(gf_sqr(u1), WGLOAD(n4 + j)));
        GF v1 = gf_mul(u0, gf_add(u1, u1));
        r[k_local] = gf_add(v0, v1);
        if (k == 0) r[1] = gf_sqr(gf_add(r[1], r[1]));
        else r[mk_local] = gf_sub_conj(v0, v1);
    }

    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        int base = 2 * t;
        GF tw = WGLOAD(n4 + j);
        GF u0 = r[base + 0];
        GF u1 = r[base + 1];
        z[2*j + 0] = gf_add(u0, u1);
        z[2*j + 1] = gf_mulconj(gf_sub(u0, u1), tw);
    }
}

__kernel void forward4(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid = (int)get_global_id(0);
    int j = gid / m;
    int i = gid % m;
    if (j >= s) return;
    int b = j * (m << 2);
    GF u0 = z[b + i];
    GF u1 = gf_mul(z[b + m + i],     WGLOAD(2 * (s + j)));
    GF u2 = gf_mul(z[b + 2*m + i],   WGLOAD(s + j));
    GF u3 = gf_mul(z[b + 3*m + i],   WGLOAD((n >> 1) + s + j));
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    z[b + i]       = gf_add(v0, v1);
    z[b + m + i]   = gf_sub(v0, v1);
    z[b + 2*m + i] = gf_addi(v2, v3);
    z[b + 3*m + i] = gf_subi(v2, v3);
}

__kernel void forward4_local(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    GF u0 = scratch[lid];
    GF u1 = gf_mul(scratch[m + lid],     WGLOAD(2 * (s + j)));
    GF u2 = gf_mul(scratch[2*m + lid],   WGLOAD(s + j));
    GF u3 = gf_mul(scratch[3*m + lid],   WGLOAD((n >> 1) + s + j));
    GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
    z[b + lid]       = gf_add(v0, v1);
    z[b + m + lid]   = gf_sub(v0, v1);
    z[b + 2*m + lid] = gf_addi(v2, v3);
    z[b + 3*m + lid] = gf_subi(v2, v3);
}



__kernel void forward4_local2(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m || (m & 3) != 0) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        GF u0 = scratch[lid];
        GF u1 = gf_mul(scratch[m + lid],     WGLOAD(2 * (s + j)));
        GF u2 = gf_mul(scratch[2*m + lid],   WGLOAD(s + j));
        GF u3 = gf_mul(scratch[3*m + lid],   WGLOAD((n >> 1) + s + j));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        scratch[lid]       = gf_add(v0, v1);
        scratch[m + lid]   = gf_sub(v0, v1);
        scratch[2*m + lid] = gf_addi(v2, v3);
        scratch[3*m + lid] = gf_subi(v2, v3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int m2 = m >> 2;
    int q = lid / m2;
    int i2 = lid - q * m2;
    int base = q * m;
    int s4 = s << 2;
    int j2 = (j << 2) + q;
    {
        GF u0 = scratch[base + i2];
        GF u1 = gf_mul(scratch[base + m2 + i2],       WGLOAD(2 * (s4 + j2)));
        GF u2 = gf_mul(scratch[base + 2*m2 + i2],     WGLOAD(s4 + j2));
        GF u3 = gf_mul(scratch[base + 3*m2 + i2],     WGLOAD((n >> 1) + s4 + j2));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        scratch[base + i2]       = gf_add(v0, v1);
        scratch[base + m2 + i2]  = gf_sub(v0, v1);
        scratch[base + 2*m2 + i2]= gf_addi(v2, v3);
        scratch[base + 3*m2 + i2]= gf_subi(v2, v3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    z[b + lid]       = scratch[lid];
    z[b + m + lid]   = scratch[m + lid];
    z[b + 2*m + lid] = scratch[2*m + lid];
    z[b + 3*m + lid] = scratch[3*m + lid];
}

__kernel void square_half_x2(__global GF* restrict z, __global const ulong* restrict w, const int n) {
    int j0 = (int)(get_global_id(0) * 2);
    int n4 = n >> 2;
    for (int t = 0; t < 2; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        int k = 2 * j;
        int mk = (k != 0) ? ((3u << (31 - clz((uint)k))) - k - 1) : 0;
        GF zk = z[k], zmk = z[mk];
        GF u0 = gf_addconj(zk, zmk);
        GF u1 = gf_subconj(zk, zmk);
        GF v0 = gf_sub(gf_sqr(u0), gf_mul(gf_sqr(u1), WGLOAD(n4 + j)));
        GF v1 = gf_mul(u0, gf_add(u1, u1));
        z[k] = gf_add(v0, v1);
        if (k == 0) z[1] = gf_sqr(gf_add(z[1], z[1]));
        else z[mk] = gf_sub_conj(v0, v1);
    }
}

__kernel void square_half_x4(__global GF* restrict z, __global const ulong* restrict w, const int n) {
    int j0 = (int)(get_global_id(0) * 4);
    int n4 = n >> 2;
    for (int t = 0; t < 4; ++t) {
        int j = j0 + t;
        if (j >= n4) break;
        int k = 2 * j;
        int mk = (k != 0) ? ((3u << (31 - clz((uint)k))) - k - 1) : 0;
        GF zk = z[k], zmk = z[mk];
        GF u0 = gf_addconj(zk, zmk);
        GF u1 = gf_subconj(zk, zmk);
        GF v0 = gf_sub(gf_sqr(u0), gf_mul(gf_sqr(u1), WGLOAD(n4 + j)));
        GF v1 = gf_mul(u0, gf_add(u1, u1));
        z[k] = gf_add(v0, v1);
        if (k == 0) z[1] = gf_sqr(gf_add(z[1], z[1]));
        else z[mk] = gf_sub_conj(v0, v1);
    }
}

__kernel void square_half(__global GF* restrict z, __global const ulong* restrict w, const int n) {
    int j = (int)get_global_id(0);
    int n4 = n >> 2;
    if (j >= n4) return;
    int k = 2 * j;
    int mk = (k != 0) ? ((3u << (31 - clz((uint)k))) - k - 1) : 0;
    GF zk = z[k], zmk = z[mk];
    GF u0 = gf_addconj(zk, zmk);
    GF u1 = gf_subconj(zk, zmk);
    GF v0 = gf_sub(gf_sqr(u0), gf_mul(gf_sqr(u1), WGLOAD(n4 + j)));
    GF v1 = gf_mul(u0, gf_add(u1, u1));
    z[k] = gf_add(v0, v1);
    if (k == 0) z[1] = gf_sqr(gf_add(z[1], z[1]));
    else z[mk] = gf_sub_conj(v0, v1);
}


__kernel void forward4_x2(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid0 = (int)(get_global_id(0) * 2);
    int limit = s * m;
    for (int t = 0; t < 2; ++t) {
        int gid = gid0 + t;
        if (gid >= limit) break;
        int j = gid / m;
        int i = gid - j * m;
        int b = j * (m << 2);
        GF u0 = z[b + i];
        GF u1 = gf_mul(z[b + m + i],     WGLOAD(2 * (s + j)));
        GF u2 = gf_mul(z[b + 2*m + i],   WGLOAD(s + j));
        GF u3 = gf_mul(z[b + 3*m + i],   WGLOAD((n >> 1) + s + j));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        z[b + i]       = gf_add(v0, v1);
        z[b + m + i]   = gf_sub(v0, v1);
        z[b + 2*m + i] = gf_addi(v2, v3);
        z[b + 3*m + i] = gf_subi(v2, v3);
    }
}

__kernel void forward4_x4(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid0 = (int)(get_global_id(0) * 4);
    int limit = s * m;
    for (int t = 0; t < 4; ++t) {
        int gid = gid0 + t;
        if (gid >= limit) break;
        int j = gid / m;
        int i = gid - j * m;
        int b = j * (m << 2);
        GF u0 = z[b + i];
        GF u1 = gf_mul(z[b + m + i],     WGLOAD(2 * (s + j)));
        GF u2 = gf_mul(z[b + 2*m + i],   WGLOAD(s + j));
        GF u3 = gf_mul(z[b + 3*m + i],   WGLOAD((n >> 1) + s + j));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        z[b + i]       = gf_add(v0, v1);
        z[b + m + i]   = gf_sub(v0, v1);
        z[b + 2*m + i] = gf_addi(v2, v3);
        z[b + 3*m + i] = gf_subi(v2, v3);
    }
}

__kernel void backward4_x2(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid0 = (int)(get_global_id(0) * 2);
    int limit = s * m;
    for (int t = 0; t < 2; ++t) {
        int gid = gid0 + t;
        if (gid >= limit) break;
        int j = gid / m;
        int i = gid - j * m;
        int b = j * (m << 2);
        GF u0 = z[b + i], u1 = z[b + m + i], u2 = z[b + 2*m + i], u3 = z[b + 3*m + i];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        z[b + i]       = gf_add(v0, v2);
        z[b + 2*m + i] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
        z[b + m + i]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
        z[b + 3*m + i] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
    }
}

__kernel void backward4_x4(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid0 = (int)(get_global_id(0) * 4);
    int limit = s * m;
    for (int t = 0; t < 4; ++t) {
        int gid = gid0 + t;
        if (gid >= limit) break;
        int j = gid / m;
        int i = gid - j * m;
        int b = j * (m << 2);
        GF u0 = z[b + i], u1 = z[b + m + i], u2 = z[b + 2*m + i], u3 = z[b + 3*m + i];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        z[b + i]       = gf_add(v0, v2);
        z[b + 2*m + i] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
        z[b + m + i]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
        z[b + 3*m + i] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
    }
}

__kernel void backward4(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    int gid = (int)get_global_id(0);
    int j = gid / m;
    int i = gid % m;
    if (j >= s) return;
    int b = j * (m << 2);
    GF u0 = z[b + i], u1 = z[b + m + i], u2 = z[b + 2*m + i], u3 = z[b + 3*m + i];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    z[b + i]       = gf_add(v0, v2);
    z[b + 2*m + i] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
    z[b + m + i]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
    z[b + 3*m + i] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
}

__kernel void backward4_local(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    GF u0 = scratch[lid], u1 = scratch[m + lid], u2 = scratch[2*m + lid], u3 = scratch[3*m + lid];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    z[b + lid]       = gf_add(v0, v2);
    z[b + 2*m + lid] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
    z[b + m + lid]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
    z[b + 3*m + lid] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
}


__kernel void backward4_local2(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j2 = (int)get_group_id(0);
    if (s <= 4 || (m << 2) <= 0) return;
    int m_big = m << 2;
    int s2 = s >> 2;
    if (j2 >= s2 || lid >= m_big) return;
    int q = lid / m;
    int i = lid - q * m;
    int base_global = j2 * (m_big << 2);
    int base_local = q * (m << 2);
    int j = (j2 << 2) + q;
    scratch[base_local + i]         = z[base_global + base_local + i];
    scratch[base_local + m + i]     = z[base_global + base_local + m + i];
    scratch[base_local + 2*m + i]   = z[base_global + base_local + 2*m + i];
    scratch[base_local + 3*m + i]   = z[base_global + base_local + 3*m + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        GF u0 = scratch[base_local + i], u1 = scratch[base_local + m + i], u2 = scratch[base_local + 2*m + i], u3 = scratch[base_local + 3*m + i];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        scratch[base_local + i]       = gf_add(v0, v2);
        scratch[base_local + 2*m + i] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
        scratch[base_local + m + i]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
        scratch[base_local + 3*m + i] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        GF u0 = scratch[lid], u1 = scratch[m_big + lid], u2 = scratch[2*m_big + lid], u3 = scratch[3*m_big + lid];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        scratch[lid]             = gf_add(v0, v2);
        scratch[2*m_big + lid]   = gf_mulconj(gf_sub(v0, v2), WGLOAD(s2 + j2));
        scratch[m_big + lid]     = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s2 + j2)));
        scratch[3*m_big + lid]   = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s2 + j2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    z[base_global + lid]           = scratch[lid];
    z[base_global + m_big + lid]   = scratch[m_big + lid];
    z[base_global + 2*m_big + lid] = scratch[2*m_big + lid];
    z[base_global + 3*m_big + lid] = scratch[3*m_big + lid];
}


__kernel void forward64_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m || (m & 15) != 0) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    forward4_stage_local(scratch, 0, m, lid, w, s + j, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m2 = m >> 2;
    int q4 = lid / m2;
    int i2 = lid - q4 * m2;
    int base4 = q4 * m;
    forward4_stage_local(scratch, base4, m2, i2, w, (s << 2) + ((j << 2) + q4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m4 = m >> 4;
    int q16 = lid / m4;
    int i4 = lid - q16 * m4;
    int base16 = (q16 >> 2) * m + (q16 & 3) * m2;
    forward4_stage_local(scratch, base16, m4, i4, w, (s << 4) + ((j << 4) + q16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[b + lid]       = scratch[lid];
    z[b + m + lid]   = scratch[m + lid];
    z[b + 2*m + lid] = scratch[2*m + lid];
    z[b + 3*m + lid] = scratch[3*m + lid];
}

__kernel void forward256_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m || (m & 63) != 0) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    forward4_stage_local(scratch, 0, m, lid, w, s + j, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m2 = m >> 2;
    int q4 = lid / m2;
    int i2 = lid - q4 * m2;
    int base4 = q4 * m;
    forward4_stage_local(scratch, base4, m2, i2, w, (s << 2) + ((j << 2) + q4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m4 = m >> 4;
    int q16 = lid / m4;
    int i4 = lid - q16 * m4;
    int base16 = (q16 >> 2) * m + (q16 & 3) * m2;
    forward4_stage_local(scratch, base16, m4, i4, w, (s << 4) + ((j << 4) + q16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m8 = m >> 6;
    int q64 = lid / m8;
    int i8 = lid - q64 * m8;
    int base64 = ((q64 >> 4) * m) + (((q64 >> 2) & 3) * m2) + ((q64 & 3) * m4);
    forward4_stage_local(scratch, base64, m8, i8, w, (s << 6) + ((j << 6) + q64), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[b + lid]       = scratch[lid];
    z[b + m + lid]   = scratch[m + lid];
    z[b + 2*m + lid] = scratch[2*m + lid];
    z[b + 3*m + lid] = scratch[3*m + lid];
}


__kernel void forward1024_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int j = (int)get_group_id(0);
    if (j >= s || lid >= m || (m & 255) != 0) return;
    int b = j * (m << 2);
    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    forward4_stage_local(scratch, 0, m, lid, w, s + j, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m2 = m >> 2;
    int q4 = lid / m2;
    int i2 = lid - q4 * m2;
    int base4 = q4 * m;
    forward4_stage_local(scratch, base4, m2, i2, w, (s << 2) + ((j << 2) + q4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m4 = m >> 4;
    int q16 = lid / m4;
    int i4 = lid - q16 * m4;
    int base16 = (q16 >> 2) * m + (q16 & 3) * m2;
    forward4_stage_local(scratch, base16, m4, i4, w, (s << 4) + ((j << 4) + q16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m8 = m >> 6;
    int q64 = lid / m8;
    int i8 = lid - q64 * m8;
    int base64 = ((q64 >> 4) * m) + (((q64 >> 2) & 3) * m2) + ((q64 & 3) * m4);
    forward4_stage_local(scratch, base64, m8, i8, w, (s << 6) + ((j << 6) + q64), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int m16 = m >> 8;
    int q256 = lid / m16;
    int i16 = lid - q256 * m16;
    int base256 = ((q256 >> 6) * m) + (((q256 >> 4) & 3) * m2) + (((q256 >> 2) & 3) * m4) + ((q256 & 3) * m8);
    forward4_stage_local(scratch, base256, m16, i16, w, (s << 8) + ((j << 8) + q256), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[b + lid]       = scratch[lid];
    z[b + m + lid]   = scratch[m + lid];
    z[b + 2*m + lid] = scratch[2*m + lid];
    z[b + 3*m + lid] = scratch[3*m + lid];
}


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void forward1024_stage_m256(__global GF* restrict z, __global const ulong* restrict w, int s, int n, __local GF* scratch) {
    const int lid = (int)get_local_id(0);
    const int j = (int)get_group_id(0);
    const int m = 256;
    const int b = j * (m << 2);

    scratch[lid]       = z[b + lid];
    scratch[m + lid]   = z[b + m + lid];
    scratch[2*m + lid] = z[b + 2*m + lid];
    scratch[3*m + lid] = z[b + 3*m + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    forward4_stage_local(scratch, 0, m, lid, w, s + j, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        const int m2 = 64;
        const int q4 = lid >> 6;
        const int i2 = lid & 63;
        const int base4 = q4 * m;
        forward4_stage_local(scratch, base4, m2, i2, w, (s << 2) + ((j << 2) + q4), n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        const int m4 = 16;
        const int q16 = lid >> 4;
        const int i4 = lid & 15;
        const int base16 = ((q16 >> 2) * m) + ((q16 & 3) << 6);
        forward4_stage_local(scratch, base16, m4, i4, w, (s << 4) + ((j << 4) + q16), n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        const int m8 = 4;
        const int q64 = lid >> 2;
        const int i8 = lid & 3;
        const int base64 = ((q64 >> 4) * m) + (((q64 >> 2) & 3) << 6) + ((q64 & 3) << 4);
        forward4_stage_local(scratch, base64, m8, i8, w, (s << 6) + ((j << 6) + q64), n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    z[b + lid]       = scratch[lid];
    z[b + m + lid]   = scratch[m + lid];
    z[b + 2*m + lid] = scratch[2*m + lid];
    z[b + 3*m + lid] = scratch[3*m + lid];
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void forward256_stage_m64(__global GF* restrict z, __global const ulong* restrict w, int s, int n, __local GF* scratch) {
    const int lid = (int)get_local_id(0);
    const int j = (int)get_group_id(0);
    const int base = j << 8;

    scratch[lid]        = z[base + lid];
    scratch[64 + lid]   = z[base + 64 + lid];
    scratch[128 + lid]  = z[base + 128 + lid];
    scratch[192 + lid]  = z[base + 192 + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    forward4_stage_local(scratch, 0, 64, lid, w, s + j, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int q4 = lid >> 4;
    const int i2 = lid & 15;
    const int base4 = q4 << 6;
    forward4_stage_local(scratch, base4, 16, i2, w, (s << 2) + ((j << 2) + q4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int q16 = lid >> 2;
    const int i4 = lid & 3;
    const int base16 = ((q16 >> 2) << 6) + ((q16 & 3) << 4);
    forward4_stage_local(scratch, base16, 4, i4, w, (s << 4) + ((j << 4) + q16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int q64 = lid;
    const int base64 = ((q64 >> 4) << 6) + (((q64 >> 2) & 3) << 4) + ((q64 & 3) << 2);
    forward4_stage_local(scratch, base64, 1, 0, w, (s << 6) + ((j << 6) + q64), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[base + lid]       = scratch[lid];
    z[base + 64 + lid]  = scratch[64 + lid];
    z[base + 128 + lid] = scratch[128 + lid];
    z[base + 192 + lid] = scratch[192 + lid];
}
__kernel void backward64_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int ls = 16 * m;
    int groups = s >> 4;
    if (g >= groups || lid >= ls) return;

    int block = lid / m;
    int lane0 = lid - block * m;
    int base_block = block * (m << 2);
    int base_global = g * (m << 6);
    scratch[base_block + lane0]         = z[base_global + base_block + lane0];
    scratch[base_block + m + lane0]     = z[base_global + base_block + m + lane0];
    scratch[base_block + 2*m + lane0]   = z[base_global + base_block + 2*m + lane0];
    scratch[base_block + 3*m + lane0]   = z[base_global + base_block + 3*m + lane0];
    barrier(CLK_LOCAL_MEM_FENCE);

    backward4_stage_local(scratch, base_block, m, lane0, w, s + ((g << 4) + block), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride1 = m << 2;
    int block4 = lid / stride1;
    int lane1 = lid - block4 * stride1;
    int base4 = block4 * (stride1 << 2);
    backward4_stage_local(scratch, base4, stride1, lane1, w, (s >> 2) + ((g << 2) + block4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride2 = m << 4;
    backward4_stage_local(scratch, 0, stride2, lid, w, (s >> 4) + g, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[base_global + lid]                = scratch[lid];
    z[base_global + stride2 + lid]      = scratch[stride2 + lid];
    z[base_global + 2*stride2 + lid]    = scratch[2*stride2 + lid];
    z[base_global + 3*stride2 + lid]    = scratch[3*stride2 + lid];
}

__kernel void backward256_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int ls = 64 * m;
    int groups = s >> 6;
    if (g >= groups || lid >= ls) return;

    int block = lid / m;
    int lane0 = lid - block * m;
    int base_block = block * (m << 2);
    int base_global = g * (m << 8);
    scratch[base_block + lane0]         = z[base_global + base_block + lane0];
    scratch[base_block + m + lane0]     = z[base_global + base_block + m + lane0];
    scratch[base_block + 2*m + lane0]   = z[base_global + base_block + 2*m + lane0];
    scratch[base_block + 3*m + lane0]   = z[base_global + base_block + 3*m + lane0];
    barrier(CLK_LOCAL_MEM_FENCE);

    backward4_stage_local(scratch, base_block, m, lane0, w, s + ((g << 6) + block), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride1 = m << 2;
    int block4 = lid / stride1;
    int lane1 = lid - block4 * stride1;
    int base4 = block4 * (stride1 << 2);
    backward4_stage_local(scratch, base4, stride1, lane1, w, (s >> 2) + ((g << 4) + block4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride2 = m << 4;
    int block16 = lid / stride2;
    int lane2 = lid - block16 * stride2;
    int base16 = block16 * (stride2 << 2);
    backward4_stage_local(scratch, base16, stride2, lane2, w, (s >> 4) + ((g << 2) + block16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride3 = m << 6;
    backward4_stage_local(scratch, 0, stride3, lid, w, (s >> 6) + g, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[base_global + lid]                = scratch[lid];
    z[base_global + stride3 + lid]      = scratch[stride3 + lid];
    z[base_global + 2*stride3 + lid]    = scratch[2*stride3 + lid];
    z[base_global + 3*stride3 + lid]    = scratch[3*stride3 + lid];
}

__kernel void backward1024_stage(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int ls = 256 * m;
    int groups = s >> 8;
    if (g >= groups || lid >= ls) return;

    int block = lid / m;
    int lane0 = lid - block * m;
    int base_block = block * (m << 2);
    int base_global = g * (m << 10);
    scratch[base_block + lane0]         = z[base_global + base_block + lane0];
    scratch[base_block + m + lane0]     = z[base_global + base_block + m + lane0];
    scratch[base_block + 2*m + lane0]   = z[base_global + base_block + 2*m + lane0];
    scratch[base_block + 3*m + lane0]   = z[base_global + base_block + 3*m + lane0];
    barrier(CLK_LOCAL_MEM_FENCE);

    backward4_stage_local(scratch, base_block, m, lane0, w, s + ((g << 8) + block), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride1 = m << 2;
    int block4 = lid / stride1;
    int lane1 = lid - block4 * stride1;
    int base4 = block4 * (stride1 << 2);
    backward4_stage_local(scratch, base4, stride1, lane1, w, (s >> 2) + ((g << 6) + block4), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride2 = m << 4;
    int block16 = lid / stride2;
    int lane2 = lid - block16 * stride2;
    int base16 = block16 * (stride2 << 2);
    backward4_stage_local(scratch, base16, stride2, lane2, w, (s >> 4) + ((g << 4) + block16), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride3 = m << 6;
    int block64 = lid / stride3;
    int lane3 = lid - block64 * stride3;
    int base64 = block64 * (stride3 << 2);
    backward4_stage_local(scratch, base64, stride3, lane3, w, (s >> 6) + ((g << 2) + block64), n);
    barrier(CLK_LOCAL_MEM_FENCE);

    int stride4 = m << 8;
    backward4_stage_local(scratch, 0, stride4, lid, w, (s >> 8) + g, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    z[base_global + lid]                = scratch[lid];
    z[base_global + stride4 + lid]      = scratch[stride4 + lid];
    z[base_global + 2*stride4 + lid]    = scratch[2*stride4 + lid];
    z[base_global + 3*stride4 + lid]    = scratch[3*stride4 + lid];
}



__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void forward_pair_large2(__global GF* restrict z, __global const ulong* restrict w, int s, int m, int n) {
    const int lid = (int)get_local_id(0);
    const int gid = (int)get_global_id(0);
    const int m2 = m >> 2;
    const int total = s * m2;
    if (m2 <= 0 || gid >= total) return;

    const int j = gid / m2;
    const int r = gid - j * m2;
    const int base = j * (m << 2);
    const int tw1 = s + j;

    __local GF tw[15];
    if (lid < 3) {
        if (lid == 0) tw[0] = WGLOAD(2 * tw1);
        else if (lid == 1) tw[1] = WGLOAD(tw1);
        else tw[2] = WGLOAD((n >> 1) + tw1);
    }
    if (lid < 12) {
        const int q = lid / 3;
        const int which = lid - q * 3;
        const int tw2 = (s << 2) + ((j << 2) + q);
        if (which == 0) tw[3 + lid] = WGLOAD(2 * tw2);
        else if (which == 1) tw[3 + lid] = WGLOAD(tw2);
        else tw[3 + lid] = WGLOAD((n >> 1) + tw2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const GF w1a = tw[0];
    const GF w1b = tw[1];
    const GF w1c = tw[2];

    for (int q = 0; q < 4; ++q) {
        const int off = base + q * m2 + r;

        GF a0 = z[off];
        GF a1 = gf_mul(z[off + m],     w1a);
        GF a2 = gf_mul(z[off + 2 * m], w1b);
        GF a3 = gf_mul(z[off + 3 * m], w1c);
        GF b0 = gf_add(a0, a2), b1 = gf_add(a1, a3), b2 = gf_sub(a0, a2), b3 = gf_sub(a1, a3);
        GF c0 = gf_add(b0, b1);
        GF c1 = gf_sub(b0, b1);
        GF c2 = gf_addi(b2, b3);
        GF c3 = gf_subi(b2, b3);

        const GF w2a = tw[3 + 3 * q + 0];
        const GF w2b = tw[3 + 3 * q + 1];
        const GF w2c = tw[3 + 3 * q + 2];
        GF d0 = c0;
        GF d1 = gf_mul(c1, w2a);
        GF d2 = gf_mul(c2, w2b);
        GF d3 = gf_mul(c3, w2c);
        GF e0 = gf_add(d0, d2), e1 = gf_add(d1, d3), e2 = gf_sub(d0, d2), e3 = gf_sub(d1, d3);

        const int out = base + q * m + r;
        z[out]          = gf_add(e0, e1);
        z[out + m2]     = gf_sub(e0, e1);
        z[out + 2 * m2] = gf_addi(e2, e3);
        z[out + 3 * m2] = gf_subi(e2, e3);
    }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void forward64_0(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;

    scratch[lid] = lshift_GF(z[i],
                             (uint)iw[2*(uint)i].w61, (uint)iw[2*(uint)i+1].w61,
                             (uint)iw[2*(uint)i].w31, (uint)iw[2*(uint)i+1].w31);
    scratch[64 + lid] = lshift_GF(z[m0 + i],
                                  (uint)iw[2*(uint)(m0 + i)].w61, (uint)iw[2*(uint)(m0 + i)+1].w61,
                                  (uint)iw[2*(uint)(m0 + i)].w31, (uint)iw[2*(uint)(m0 + i)+1].w31);
    scratch[128 + lid] = lshift_GF(z[(m0 << 1) + i],
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w61, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61,
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w31, (uint)iw[2*(uint)((m0 << 1) + i)+1].w31);
    scratch[192 + lid] = lshift_GF(z[3 * m0 + i],
                                   (uint)iw[2*(uint)(3 * m0 + i)].w61, (uint)iw[2*(uint)(3 * m0 + i)+1].w61,
                                   (uint)iw[2*(uint)(3 * m0 + i)].w31, (uint)iw[2*(uint)(3 * m0 + i)+1].w31);
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        GF u0 = scratch[lid];
        GF u1 = gf_mul(scratch[64 + lid], WGLOAD(2));
        GF u2 = gf_mul(scratch[128 + lid], WGLOAD(1));
        GF u3 = gf_mul(scratch[192 + lid], WGLOAD((n >> 1) + 1));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        scratch[lid]       = gf_add(v0, v1);
        scratch[64 + lid]  = gf_sub(v0, v1);
        scratch[128 + lid] = gf_addi(v2, v3);
        scratch[192 + lid] = gf_subi(v2, v3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        forward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        forward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    z[i]            = scratch[lid];
    z[m0 + i]       = scratch[64 + lid];
    z[(m0 << 1) + i]= scratch[128 + lid];
    z[3 * m0 + i]   = scratch[192 + lid];
}


__kernel void forward64_0_small31_refresh(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;
    __global const ulong* restrict zraw = (__global const ulong*)z;
    const uint b0 = 3u * (uint)i;
    const uint b1 = 3u * (uint)(m0 + i);
    const uint b2 = 3u * (uint)((m0 << 1) + i);
    const uint b3 = 3u * (uint)(3 * m0 + i);

    scratch[lid] = lshift_GF_small31_words(zraw[b0 + 0u], zraw[b0 + 1u],
                             (uint)iw[2*(uint)i].w61, (uint)iw[2*(uint)i+1].w61,
                             (uint)iw[2*(uint)i].w31, (uint)iw[2*(uint)i+1].w31);
    scratch[64 + lid] = lshift_GF_small31_words(zraw[b1 + 0u], zraw[b1 + 1u],
                                  (uint)iw[2*(uint)(m0 + i)].w61, (uint)iw[2*(uint)(m0 + i)+1].w61,
                                  (uint)iw[2*(uint)(m0 + i)].w31, (uint)iw[2*(uint)(m0 + i)+1].w31);
    scratch[128 + lid] = lshift_GF_small31_words(zraw[b2 + 0u], zraw[b2 + 1u],
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w61, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61,
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w31, (uint)iw[2*(uint)((m0 << 1) + i)+1].w31);
    scratch[192 + lid] = lshift_GF_small31_words(zraw[b3 + 0u], zraw[b3 + 1u],
                                   (uint)iw[2*(uint)(3 * m0 + i)].w61, (uint)iw[2*(uint)(3 * m0 + i)+1].w61,
                                   (uint)iw[2*(uint)(3 * m0 + i)].w31, (uint)iw[2*(uint)(3 * m0 + i)+1].w31);
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        GF u0 = scratch[lid];
        GF u1 = gf_mul(scratch[64 + lid], WGLOAD(2));
        GF u2 = gf_mul(scratch[128 + lid], WGLOAD(1));
        GF u3 = gf_mul(scratch[192 + lid], WGLOAD((n >> 1) + 1));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        scratch[lid]       = gf_add(v0, v1);
        scratch[64 + lid]  = gf_sub(v0, v1);
        scratch[128 + lid] = gf_addi(v2, v3);
        scratch[192 + lid] = gf_subi(v2, v3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        forward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        forward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    z[i]            = scratch[lid];
    z[m0 + i]       = scratch[64 + lid];
    z[(m0 << 1) + i]= scratch[128 + lid];
    z[3 * m0 + i]   = scratch[192 + lid];
}

__kernel void forward64_0_small31_refresh_compact(__global GF* restrict z, __global const ulong2* restrict zc, __global const ulong* restrict w, __global const IW* restrict iw, int n, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;
    const ulong2 p0 = zc[(uint)i];
    const ulong2 p1 = zc[(uint)(m0 + i)];
    const ulong2 p2 = zc[(uint)((m0 << 1) + i)];
    const ulong2 p3 = zc[(uint)(3 * m0 + i)];

    scratch[lid] = lshift_GF_small31_words(p0.s0, p0.s1,
                             (uint)iw[2*(uint)i].w61, (uint)iw[2*(uint)i+1].w61,
                             (uint)iw[2*(uint)i].w31, (uint)iw[2*(uint)i+1].w31);
    scratch[64 + lid] = lshift_GF_small31_words(p1.s0, p1.s1,
                                  (uint)iw[2*(uint)(m0 + i)].w61, (uint)iw[2*(uint)(m0 + i)+1].w61,
                                  (uint)iw[2*(uint)(m0 + i)].w31, (uint)iw[2*(uint)(m0 + i)+1].w31);
    scratch[128 + lid] = lshift_GF_small31_words(p2.s0, p2.s1,
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w61, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61,
                                   (uint)iw[2*(uint)((m0 << 1) + i)].w31, (uint)iw[2*(uint)((m0 << 1) + i)+1].w31);
    scratch[192 + lid] = lshift_GF_small31_words(p3.s0, p3.s1,
                                   (uint)iw[2*(uint)(3 * m0 + i)].w61, (uint)iw[2*(uint)(3 * m0 + i)+1].w61,
                                   (uint)iw[2*(uint)(3 * m0 + i)].w31, (uint)iw[2*(uint)(3 * m0 + i)+1].w31);
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        GF u0 = scratch[lid];
        GF u1 = gf_mul(scratch[64 + lid], WGLOAD(2));
        GF u2 = gf_mul(scratch[128 + lid], WGLOAD(1));
        GF u3 = gf_mul(scratch[192 + lid], WGLOAD((n >> 1) + 1));
        GF v0 = gf_add(u0, u2), v1 = gf_add(u1, u3), v2 = gf_sub(u0, u2), v3 = gf_sub(u1, u3);
        scratch[lid]       = gf_add(v0, v1);
        scratch[64 + lid]  = gf_sub(v0, v1);
        scratch[128 + lid] = gf_addi(v2, v3);
        scratch[192 + lid] = gf_subi(v2, v3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        forward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        forward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    z[i]            = scratch[lid];
    z[m0 + i]       = scratch[64 + lid];
    z[(m0 << 1) + i]= scratch[128 + lid];
    z[3 * m0 + i]   = scratch[192 + lid];
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void backward64_0(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int n, int ln, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;

    scratch[lid] = z[i];
    scratch[64 + lid] = z[m0 + i];
    scratch[128 + lid] = z[(m0 << 1) + i];
    scratch[192 + lid] = z[3 * m0 + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        backward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        backward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    GF u0 = scratch[lid];
    GF u1 = scratch[64 + lid];
    GF u2 = scratch[128 + lid];
    GF u3 = scratch[192 + lid];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    uint ln1 = (uint)ln + 1u;
    GF z0 = gf_add(v0, v2);
    GF z1 = gf_mulconj(gf_sub(v0, v2), WGLOAD(1));
    GF z2 = gf_mulconj(gf_addi(v1, v3), WGLOAD(2));
    GF z3 = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + 1));

    z[i] = rshift_GF(z0,
                     (uint)iw[2*(uint)i].w61 + ln1, (uint)iw[2*(uint)i+1].w61 + ln1,
                     (uint)iw[2*(uint)i].w31 + ln1, (uint)iw[2*(uint)i+1].w31 + ln1);
    z[m0 + i] = rshift_GF(z2,
                          (uint)iw[2*(uint)(m0 + i)].w61 + ln1, (uint)iw[2*(uint)(m0 + i)+1].w61 + ln1,
                          (uint)iw[2*(uint)(m0 + i)].w31 + ln1, (uint)iw[2*(uint)(m0 + i)+1].w31 + ln1);
    z[(m0 << 1) + i] = rshift_GF(z1,
                                 (uint)iw[2*(uint)((m0 << 1) + i)].w61 + ln1, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61 + ln1,
                                 (uint)iw[2*(uint)((m0 << 1) + i)].w31 + ln1, (uint)iw[2*(uint)((m0 << 1) + i)+1].w31 + ln1);
    z[3 * m0 + i] = rshift_GF(z3,
                              (uint)iw[2*(uint)(3 * m0 + i)].w61 + ln1, (uint)iw[2*(uint)(3 * m0 + i)+1].w61 + ln1,
                              (uint)iw[2*(uint)(3 * m0 + i)].w31 + ln1, (uint)iw[2*(uint)(3 * m0 + i)+1].w31 + ln1);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void backward64_0_small31_defer(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int n, int ln, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;

    scratch[lid] = z[i];
    scratch[64 + lid] = z[m0 + i];
    scratch[128 + lid] = z[(m0 << 1) + i];
    scratch[192 + lid] = z[3 * m0 + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        backward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        backward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    GF u0 = scratch[lid];
    GF u1 = scratch[64 + lid];
    GF u2 = scratch[128 + lid];
    GF u3 = scratch[192 + lid];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    uint ln1 = (uint)ln + 1u;
    GF z0 = gf_add(v0, v2);
    GF z1 = gf_mulconj(gf_sub(v0, v2), WGLOAD(1));
    GF z2 = gf_mulconj(gf_addi(v1, v3), WGLOAD(2));
    GF z3 = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + 1));

    __global ulong* restrict zraw = (__global ulong*)z;
    uint b0 = 3u * (uint)i;
    uint b1 = 3u * (uint)(m0 + i);
    uint b2 = 3u * (uint)((m0 << 1) + i);
    uint b3 = 3u * (uint)(3 * m0 + i);

    zraw[b0 + 0u] = rshift_mod61(z0.s0, (uint)iw[2*(uint)i].w61 + ln1);
    zraw[b0 + 1u] = rshift_mod61(z0.s1, (uint)iw[2*(uint)i+1].w61 + ln1);
    zraw[b1 + 0u] = rshift_mod61(z2.s0, (uint)iw[2*(uint)(m0 + i)].w61 + ln1);
    zraw[b1 + 1u] = rshift_mod61(z2.s1, (uint)iw[2*(uint)(m0 + i)+1].w61 + ln1);
    zraw[b2 + 0u] = rshift_mod61(z1.s0, (uint)iw[2*(uint)((m0 << 1) + i)].w61 + ln1);
    zraw[b2 + 1u] = rshift_mod61(z1.s1, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61 + ln1);
    zraw[b3 + 0u] = rshift_mod61(z3.s0, (uint)iw[2*(uint)(3 * m0 + i)].w61 + ln1);
    zraw[b3 + 1u] = rshift_mod61(z3.s1, (uint)iw[2*(uint)(3 * m0 + i)+1].w61 + ln1);
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void backward64_0_small31_defer_compact(__global GF* restrict z, __global ulong2* restrict zc, __global const ulong* restrict w, __global const IW* restrict iw, int n, int ln, __local GF* scratch) {
    int lid = (int)get_local_id(0);
    int g = (int)get_group_id(0);
    int m0 = n >> 3;
    int i = (g << 6) + lid;
    if (i >= m0) return;

    scratch[lid] = z[i];
    scratch[64 + lid] = z[m0 + i];
    scratch[128 + lid] = z[(m0 << 1) + i];
    scratch[192 + lid] = z[3 * m0 + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 2;
        int r = lid & 3;
        int base = ((q >> 2) << 6) + ((q & 3) << 4);
        backward4_stage_local(scratch, base, 4, r, w, 16 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int q = lid >> 4;
        int r = lid & 15;
        int base = q << 6;
        backward4_stage_local(scratch, base, 16, r, w, 4 + q, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    GF u0 = scratch[lid];
    GF u1 = scratch[64 + lid];
    GF u2 = scratch[128 + lid];
    GF u3 = scratch[192 + lid];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    uint ln1 = (uint)ln + 1u;
    GF z0 = gf_add(v0, v2);
    GF z1 = gf_mulconj(gf_sub(v0, v2), WGLOAD(1));
    GF z2 = gf_mulconj(gf_addi(v1, v3), WGLOAD(2));
    GF z3 = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + 1));

    zc[(uint)i] = (ulong2)(rshift_mod61(z0.s0, (uint)iw[2*(uint)i].w61 + ln1),
                           rshift_mod61(z0.s1, (uint)iw[2*(uint)i+1].w61 + ln1));
    zc[(uint)(m0 + i)] = (ulong2)(rshift_mod61(z2.s0, (uint)iw[2*(uint)(m0 + i)].w61 + ln1),
                                  rshift_mod61(z2.s1, (uint)iw[2*(uint)(m0 + i)+1].w61 + ln1));
    zc[(uint)((m0 << 1) + i)] = (ulong2)(rshift_mod61(z1.s0, (uint)iw[2*(uint)((m0 << 1) + i)].w61 + ln1),
                                         rshift_mod61(z1.s1, (uint)iw[2*(uint)((m0 << 1) + i)+1].w61 + ln1));
    zc[(uint)(3 * m0 + i)] = (ulong2)(rshift_mod61(z3.s0, (uint)iw[2*(uint)(3 * m0 + i)].w61 + ln1),
                                      rshift_mod61(z3.s1, (uint)iw[2*(uint)(3 * m0 + i)+1].w61 + ln1));
}


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void backward16_stage_m64(__global GF* restrict z, __global const ulong* restrict w, int s, int n, __local GF* scratch) {
    const int lid = (int)get_local_id(0);
    const int j2 = (int)get_group_id(0);
    const int m = 64;
    const int m_big = 256;
    const int s2 = s >> 2;
    const int q = lid >> 6;
    const int i = lid & 63;
    const int base_global = j2 * (m_big << 2);
    const int base_local = q * (m << 2);
    const int j = (j2 << 2) + q;

    scratch[base_local + i]       = z[base_global + base_local + i];
    scratch[base_local + m + i]   = z[base_global + base_local + m + i];
    scratch[base_local + 2*m + i] = z[base_global + base_local + 2*m + i];
    scratch[base_local + 3*m + i] = z[base_global + base_local + 3*m + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        GF u0 = scratch[base_local + i], u1 = scratch[base_local + m + i], u2 = scratch[base_local + 2*m + i], u3 = scratch[base_local + 3*m + i];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        scratch[base_local + i]       = gf_add(v0, v2);
        scratch[base_local + 2*m + i] = gf_mulconj(gf_sub(v0, v2), WGLOAD(s + j));
        scratch[base_local + m + i]   = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s + j)));
        scratch[base_local + 3*m + i] = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s + j));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        GF u0 = scratch[lid], u1 = scratch[m_big + lid], u2 = scratch[2*m_big + lid], u3 = scratch[3*m_big + lid];
        GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
        scratch[lid]             = gf_add(v0, v2);
        scratch[2*m_big + lid]   = gf_mulconj(gf_sub(v0, v2), WGLOAD(s2 + j2));
        scratch[m_big + lid]     = gf_mulconj(gf_addi(v1, v3), WGLOAD(2*(s2 + j2)));
        scratch[3*m_big + lid]   = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + s2 + j2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    z[base_global + lid]           = scratch[lid];
    z[base_global + m_big + lid]   = scratch[m_big + lid];
    z[base_global + 2*m_big + lid] = scratch[2*m_big + lid];
    z[base_global + 3*m_big + lid] = scratch[3*m_big + lid];
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void backward4_stage_m256(__global GF* restrict z, __global const ulong* restrict w, int s, int n) {
    const int lid = (int)get_local_id(0);
    const int j = (int)get_group_id(0);
    const int m = 256;
    const int b = j * (m << 2);

    const GF u0 = z[b + lid];
    const GF u1 = z[b + m + lid];
    const GF u2 = z[b + 2*m + lid];
    const GF u3 = z[b + 3*m + lid];
    const GF tw0 = WGLOAD(s + j);
    const GF tw1 = WGLOAD(2 * (s + j));
    const GF tw2 = WGLOAD((n >> 1) + s + j);
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);

    z[b + lid]       = gf_add(v0, v2);
    z[b + 2*m + lid] = gf_mulconj(gf_sub(v0, v2), tw0);
    z[b + m + lid]   = gf_mulconj(gf_addi(v1, v3), tw1);
    z[b + 3*m + lid] = gf_mulconj(gf_subi(v1, v3), tw2);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void backward4_stage_m64(__global GF* restrict z, __global const ulong* restrict w, int s, int n) {
    const int lid = (int)get_local_id(0);
    const int j = (int)get_group_id(0);
    const int base = j << 8;

    const GF u0 = z[base + lid];
    const GF u1 = z[base + 64 + lid];
    const GF u2 = z[base + 128 + lid];
    const GF u3 = z[base + 192 + lid];
    const GF tw0 = WGLOAD(s + j);
    const GF tw1 = WGLOAD(2 * (s + j));
    const GF tw2 = WGLOAD((n >> 1) + s + j);
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);

    z[base + lid]       = gf_add(v0, v2);
    z[base + 128 + lid] = gf_mulconj(gf_sub(v0, v2), tw0);
    z[base + 64 + lid]  = gf_mulconj(gf_addi(v1, v3), tw1);
    z[base + 192 + lid] = gf_mulconj(gf_subi(v1, v3), tw2);
}

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void backward4_stage_m16(__global GF* restrict z, __global const ulong* restrict w, int s, int n) {
    const int lid = (int)get_local_id(0);
    const int j = (int)get_group_id(0);
    const int base = j << 6;

    const GF u0 = z[base + lid];
    const GF u1 = z[base + 16 + lid];
    const GF u2 = z[base + 32 + lid];
    const GF u3 = z[base + 48 + lid];
    const GF tw0 = WGLOAD(s + j);
    const GF tw1 = WGLOAD(2 * (s + j));
    const GF tw2 = WGLOAD((n >> 1) + s + j);
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);

    z[base + lid]      = gf_add(v0, v2);
    z[base + 32 + lid] = gf_mulconj(gf_sub(v0, v2), tw0);
    z[base + 16 + lid] = gf_mulconj(gf_addi(v1, v3), tw1);
    z[base + 48 + lid] = gf_mulconj(gf_subi(v1, v3), tw2);
}

__kernel void backward4_last_unweight(__global GF* restrict z, __global const ulong* restrict w, __global const IW* restrict iw, int m, int n, int ln) {
    int i = (int)get_global_id(0);
    if (i >= m) return;
    int b = 0;
    GF u0 = z[b + i], u1 = z[b + m + i], u2 = z[b + 2*m + i], u3 = z[b + 3*m + i];
    GF v0 = gf_add(u0, u1), v1 = gf_sub(u0, u1), v2 = gf_add(u2, u3), v3 = gf_sub(u3, u2);
    uint ln1 = (uint)ln + 1u;
    GF z0 = gf_add(v0, v2);
    GF z1 = gf_mulconj(gf_sub(v0, v2), WGLOAD(1));
    GF z2 = gf_mulconj(gf_addi(v1, v3), WGLOAD(2));
    GF z3 = gf_mulconj(gf_subi(v1, v3), WGLOAD((n >> 1) + 1));
    z[b + i]       = rshift_GF(z0, (uint)iw[2*(uint)i].w61 + ln1,       (uint)iw[2*(uint)i+1].w61 + ln1,       (uint)iw[2*(uint)i].w31 + ln1,       (uint)iw[2*(uint)i+1].w31 + ln1);
    z[b + m + i]   = rshift_GF(z2, (uint)iw[2*(uint)(m+i)].w61 + ln1,   (uint)iw[2*(uint)(m+i)+1].w61 + ln1,   (uint)iw[2*(uint)(m+i)].w31 + ln1,   (uint)iw[2*(uint)(m+i)+1].w31 + ln1);
    z[b + 2*m + i] = rshift_GF(z1, (uint)iw[2*(uint)(2*m+i)].w61 + ln1, (uint)iw[2*(uint)(2*m+i)+1].w61 + ln1, (uint)iw[2*(uint)(2*m+i)].w31 + ln1, (uint)iw[2*(uint)(2*m+i)+1].w31 + ln1);
    z[b + 3*m + i] = rshift_GF(z3, (uint)iw[2*(uint)(3*m+i)].w61 + ln1, (uint)iw[2*(uint)(3*m+i)+1].w61 + ln1, (uint)iw[2*(uint)(3*m+i)].w31 + ln1, (uint)iw[2*(uint)(3*m+i)+1].w31 + ln1);
}

__kernel void unweight_norm(__global GF* restrict z, __global const IW* restrict w, int ln) {
    uint i = get_global_id(0);
    uint ln1 = (uint)ln + 1u;
    z[i] = rshift_GF(z[i], (uint)w[2*i].w61 + ln1, (uint)w[2*i+1].w61 + ln1,
                          (uint)w[2*i].w31 + ln1, (uint)w[2*i+1].w31 + ln1);
}

inline ulong digit_adc(u128 lhs, uint digit_width, __private u128 *carry) {
    // All digit widths in this engine are strictly below 64 bits.
    // Specialize the carry update for that hot case to avoid the generic
    // rshift_u128/shl_u128/add_u128 sequence on every digit.
    u128 s = add_u128(lhs, *carry);
    bool overflow = (s.hi < lhs.hi) || ((s.hi == lhs.hi) && (s.lo < lhs.lo));
    uint sh = 64u - digit_width;
    carry->lo = (s.lo >> digit_width) | (s.hi << sh);
    carry->hi = (s.hi >> digit_width) + (overflow ? (1ul << sh) : 0ul);
    ulong mask = ((ulong)1 << digit_width) - 1ul;
    return s.lo & mask;
}

inline ulong digit_adc_two_width(u128 lhs,
                                 const uint narrow_w,
                                 const ulong mask_narrow,
                                 const ulong mask_wide,
                                 __private u128 *carry,
                                 const uint is_wide) {
    u128 s = add_u128(lhs, *carry);
    bool overflow = (s.hi < lhs.hi) || ((s.hi == lhs.hi) && (s.lo < lhs.lo));
    if (is_wide != 0u) {
        const uint w = narrow_w + 1u;
        const uint sh = 63u - narrow_w;
        carry->lo = (s.lo >> w) | (s.hi << sh);
        carry->hi = (s.hi >> w) + (overflow ? (1ul << sh) : 0ul);
        return s.lo & mask_wide;
    } else {
        const uint w = narrow_w;
        const uint sh = 64u - narrow_w;
        carry->lo = (s.lo >> w) | (s.hi << sh);
        carry->hi = (s.hi >> w) + (overflow ? (1ul << sh) : 0ul);
        return s.lo & mask_narrow;
    }
}

inline u128 garner_reconstruct_from_u61_u31(const u64 s, const u32 n31) {
    u64 u = sub61(s, (u64)n31);
    u = add61(u, lshift_mod61(u, 31));

    const ulong lo_shift = u << 31;
    const ulong hi_shift = u >> 33;
    const ulong lo_sub = lo_shift - u;
    const ulong borrow = (lo_shift < u) ? 1ul : 0ul;
    const ulong hi_sub = hi_shift - borrow;
    const ulong lo_fin = lo_sub + (ulong)n31;
    const ulong carry = (lo_fin < lo_sub) ? 1ul : 0ul;
    return make_u128(lo_fin, hi_sub + carry);
}

inline void garner_GF(const GF x, __private u128 *l0, __private u128 *l1) {
    *l0 = garner_reconstruct_from_u61_u31(x.s0, x.t0);
    *l1 = garner_reconstruct_from_u61_u31(x.s1, x.t1);
}

inline void garner_raw_small31_words(const ulong s0,
                                     const ulong s1,
                                     const ulong p,
                                     __private u128 *l0,
                                     __private u128 *l1) {
    *l0 = garner_reconstruct_from_u61_u31(s0, (u32)p);
    *l1 = garner_reconstruct_from_u61_u31(s1, (u32)(p >> 32));
}

inline void garner_raw_small31_lowwords(const ulong s0,
                                        const ulong s1,
                                        __private u128 *l0,
                                        __private u128 *l1) {
    *l0 = garner_reconstruct_from_u61_u31(s0, (u32)s0);
    *l1 = garner_reconstruct_from_u61_u31(s1, (u32)s1);
}

inline ulong digit_sbc(ulong lhs, uint w, __private uint *borrow) {
    uint b = *borrow;
    bool br = (lhs < (ulong)b);
    ulong r = lhs - (ulong)b + (br ? ((ulong)1 << w) : 0ul);
    *borrow = br ? 1u : 0u;
    return r & ((((ulong)1) << w) - 1ul);
}

inline u32 mod31_u64(ulong x) {
    ulong r = (x & (ulong)P31) + (x >> 31);
    r = (r & (ulong)P31) + (r >> 31);
    return (r >= (ulong)P31) ? (u32)(r - (ulong)P31) : (u32)r;
}

typedef struct { uint start, count, bits, pad; } BlockInfo;
typedef struct {
    ulong low_lo, low_hi;   // low LOWK bits of packed block remainder
    ulong q_lo, q_hi;       // carry-out with zero carry-in
    uint  prefix_all_ones;  // all bits above LOWK are ones
    uint  pad;
} BlockState;
typedef struct { ulong lo, hi; } CarryWord;

#define LOWK_BITS 96u

inline void update_lowk(__private u128 *low, uint bitpos, ulong digit, uint w)
{
    if (bitpos >= LOWK_BITS) return;
    uint part = min(w, LOWK_BITS - bitpos);
    ulong mask = (part == 64u) ? ~0ul : ((((ulong)1) << part) - 1ul);
    *low = add_u128(*low, shl_u128(make_u128(digit & mask, 0ul), bitpos));
}

inline uint top_bits_all_ones(ulong digit, uint w, uint keep_low_bits)
{
    // verify that bits [keep_low_bits, w) are all ones
    if (keep_low_bits >= w) return 1u;
    uint hi_bits = w - keep_low_bits;
    ulong hi = digit >> keep_low_bits;
    ulong mask = (hi_bits == 64u) ? ~0ul : ((((ulong)1) << hi_bits) - 1ul);
    return (hi == mask) ? 1u : 0u;
}

__kernel void block_prepare_kernel(__global GF* restrict z,
                                   __global const uchar* restrict digit_width,
                                   __global const BlockInfo* restrict blocks,
                                   __global BlockState* restrict states,
                                   const uint nblocks) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    BlockInfo bi = blocks[b];
    u128 c = make_u128(0ul, 0ul);
    for (uint t = 0; t < bi.count; ++t) {
        uint k = bi.start + t;
        GF zk = z[k];
        u128 L0, L1;
        garner_GF(zk, &L0, &L1);
        uchar2 wd = digit_width2[k];
        uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
        ulong n0 = digit_adc(L0, w0, &c);
        ulong n1 = digit_adc(L1, w1, &c);
        GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
        z[k] = out;
    }
    states[b].low_lo = 0ul; states[b].low_hi = 0ul;
    states[b].q_lo = c.lo; states[b].q_hi = c.hi;
    states[b].prefix_all_ones = 0u;
    states[b].pad = 0u;
}

__kernel void block_scan_kernel(__global const BlockInfo* restrict blocks,
                                __global const BlockState* restrict states,
                                __global CarryWord* restrict carry_in,
                                __global CarryWord* restrict final_carry,
                                const uint nblocks) {
    if (get_global_id(0) != 0) return;
    u128 c = make_u128(0ul, 0ul);
    for (uint b = 0; b < nblocks; ++b) {
        carry_in[b].lo = c.lo; carry_in[b].hi = c.hi;
        BlockInfo bi = blocks[b];
        u128 q = make_u128(states[b].q_lo, states[b].q_hi);
        if (bi.bits <= LOWK_BITS) {
            u128 low = make_u128(states[b].low_lo, states[b].low_hi);
            u128 tmp = add_u128(low, c);
            u128 extra = rshift_u128(tmp, bi.bits);
            c = add_u128(q, extra);
        } else {
            uint extra_bit = 0u;
            if (states[b].prefix_all_ones != 0u) {
                u128 low = make_u128(states[b].low_lo, states[b].low_hi);
                u128 sum = add_u128(low, c);
                if ((sum.hi >> 32) != 0ul) extra_bit = 1u; // overflow beyond low 96 bits
            }
            c = add_u128(q, make_u128((ulong)extra_bit, 0ul));
        }
    }
    final_carry[0].lo = c.lo; final_carry[0].hi = c.hi;
}

__kernel void block_finalize_kernel(__global GF* restrict z,
                                    __global const BlockInfo* restrict blocks,
                                    __global const CarryWord* restrict carry_in,
                                    __global const uchar* restrict digit_width,
                                    const uint nblocks) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    BlockInfo bi = blocks[b];
    u128 c = make_u128(carry_in[b].lo, carry_in[b].hi);
    if ((c.lo | c.hi) == 0ul) return;
    for (uint t = 0; t < bi.count; ++t) {
        uint k = bi.start + t;
        GF zk = z[k];
        uchar2 wd = digit_width2[k];
        uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
        ulong n0 = digit_adc(make_u128(zk.s0, 0ul), w0, &c);
        ulong n1 = digit_adc(make_u128(zk.s1, 0ul), w1, &c);
        GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
        z[k] = out;
        if ((c.lo | c.hi) == 0ul) break;
    }
}

inline GF make_digit_gf(ulong n0, ulong n1) {
    GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
    return out;
}

inline GF make_digit_gf_direct(ulong n0, ulong n1, uint small31) {
    GF out;
    out.s0 = n0; out.s1 = n1;
    if (small31 != 0u) {
        out.t0 = (u32)n0;
        out.t1 = (u32)n1;
    } else {
        out.t0 = mod31_u64(n0);
        out.t1 = mod31_u64(n1);
    }
    return out;
}

inline void normalize_pair_from_garner(__global GF* restrict z,
                                       __global const uchar2* restrict digit_width2,
                                       const uint k,
                                       __private u128 *carry) {
    GF zk = z[k];
    u128 L0, L1;
    garner_GF(zk, &L0, &L1);
    uchar2 wd = digit_width2[k];
    ulong n0 = digit_adc(L0, (uint)wd.s0, carry);
    ulong n1 = digit_adc(L1, (uint)wd.s1, carry);
    z[k] = make_digit_gf(n0, n1);
}

inline void apply_carry_to_pair(__global GF* restrict z,
                                __global const uchar2* restrict digit_width2,
                                const uint k,
                                __private u128 *carry) {
    GF zk = z[k];
    uchar2 wd = digit_width2[k];
    ulong n0 = digit_adc(make_u128(zk.s0, 0ul), (uint)wd.s0, carry);
    ulong n1 = digit_adc(make_u128(zk.s1, 0ul), (uint)wd.s1, carry);
    z[k] = make_digit_gf(n0, n1);
}

inline void normalize_pair_from_garner_direct(__global GF* restrict z,
                                              __global const uchar2* restrict digit_width2,
                                              const uint k,
                                              __private u128 *carry,
                                              const uint narrow_w,
                                              const ulong mask_narrow,
                                              const ulong mask_wide,
                                              const uint small31) {
    GF zk = z[k];
    u128 L0, L1;
    garner_GF(zk, &L0, &L1);
    uchar2 wd = digit_width2[k];
    ulong n0 = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, (uint)(wd.s0 != (uchar)narrow_w));
    ulong n1 = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (uint)(wd.s1 != (uchar)narrow_w));
    z[k] = make_digit_gf_direct(n0, n1, small31);
}

inline uint apply_carry_to_pair_direct(__global GF* restrict z,
                                       __global const uchar2* restrict digit_width2,
                                       const uint k,
                                       __private u128 *carry,
                                       const uint narrow_w,
                                       const ulong mask_narrow,
                                       const ulong mask_wide,
                                       const uint small31) {
    GF zk = z[k];
    uchar2 wd = digit_width2[k];
    ulong n0 = digit_adc_two_width(make_u128(zk.s0, 0ul), narrow_w, mask_narrow, mask_wide, carry, (uint)(wd.s0 != (uchar)narrow_w));
    ulong n1 = digit_adc_two_width(make_u128(zk.s1, 0ul), narrow_w, mask_narrow, mask_wide, carry, (uint)(wd.s1 != (uchar)narrow_w));
    z[k] = make_digit_gf_direct(n0, n1, small31);
    return (carry->lo | carry->hi) == 0ul;
}

__kernel void block_prepare_direct_kernel(__global GF* restrict z,
                                          __global const uchar* restrict digit_width,
                                          __global const BlockInfo* restrict blocks,
                                          __global CarryWord* restrict carry_next,
                                          const uint nblocks,
                                          const uint narrow_w,
                                          const uint small31) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    BlockInfo bi = blocks[b];
    u128 c = make_u128(0ul, 0ul);
    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    uint k = bi.start;
    if (bi.count == 8u) {
        normalize_pair_from_garner_direct(z, digit_width2, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31);
        normalize_pair_from_garner_direct(z, digit_width2, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31);
    } else {
        for (uint t = 0; t < bi.count; ++t) {
            normalize_pair_from_garner_direct(z, digit_width2, k + t, &c, narrow_w, mask_narrow, mask_wide, small31);
        }
    }
    uint dst = (b + 1u < nblocks) ? (b + 1u) : 0u;
    carry_next[dst].lo = c.lo;
    carry_next[dst].hi = c.hi;
}

__kernel void block_apply_carry_direct_kernel(__global GF* restrict z,
                                              __global const BlockInfo* restrict blocks,
                                              __global const CarryWord* restrict carry_in,
                                              __global const uchar* restrict digit_width,
                                              const uint nblocks,
                                              const uint narrow_w,
                                              const uint small31) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    u128 c = make_u128(carry_in[b].lo, carry_in[b].hi);
    if ((c.lo | c.hi) == 0ul) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    BlockInfo bi = blocks[b];
    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    uint k = bi.start;
    if (bi.count == 8u) {
        if (apply_carry_to_pair_direct(z, digit_width2, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        if (apply_carry_to_pair_direct(z, digit_width2, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
    } else {
        for (uint t = 0; t < bi.count; ++t) {
            if (apply_carry_to_pair_direct(z, digit_width2, k + t, &c, narrow_w, mask_narrow, mask_wide, small31) != 0u) return;
        }
    }
}

inline uint pair_wide_flag(const uint block_mask, const uint pair_idx, const uint lane) {
    return (block_mask >> (2u * pair_idx + lane)) & 1u;
}

inline void normalize_pair_from_garner_direct_mask(__global GF* restrict z,
                                                   const uint k,
                                                   __private u128 *carry,
                                                   const uint narrow_w,
                                                   const ulong mask_narrow,
                                                   const ulong mask_wide,
                                                   const uint small31,
                                                   const uint pair_mask) {
    GF zk = z[k];
    u128 L0, L1;
    garner_GF(zk, &L0, &L1);
    ulong n0 = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    z[k] = make_digit_gf_direct(n0, n1, small31);
}

inline uint apply_carry_to_pair_direct_mask(__global GF* restrict z,
                                            const uint k,
                                            __private u128 *carry,
                                            const uint narrow_w,
                                            const ulong mask_narrow,
                                            const ulong mask_wide,
                                            const uint small31,
                                            const uint pair_mask) {
    GF zk = z[k];
    ulong n0 = digit_adc_two_width(make_u128(zk.s0, 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(make_u128(zk.s1, 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    z[k] = make_digit_gf_direct(n0, n1, small31);
    return (carry->lo | carry->hi) == 0ul;
}

inline void normalize_pair_from_garner_direct_mask_raw(__global ulong* restrict zraw,
                                                       const uint k,
                                                       __private u128 *carry,
                                                       const uint narrow_w,
                                                       const ulong mask_narrow,
                                                       const ulong mask_wide,
                                                       const uint small31,
                                                       const uint pair_mask) {
    const uint base = 3u * k;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    u128 L0, L1;
    if (small31 != 0u) {
        garner_raw_small31_lowwords(s0, s1, &L0, &L1);
    } else {
        const ulong p  = zraw[base + 2u];
        const u32 t0 = (u32)p, t1 = (u32)(p >> 32);
        GF zk = { s0, s1, t0, t1 };
        garner_GF(zk, &L0, &L1);
    }
    ulong n0 = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    zraw[base + 0u] = n0;
    zraw[base + 1u] = n1;
    zraw[base + 2u] = (small31 != 0u) ? pack_u32x2_dev((u32)n0, (u32)n1) : pack_u32x2_dev(mod31_u64(n0), mod31_u64(n1));
}

inline uint apply_carry_to_pair_direct_mask_raw(__global ulong* restrict zraw,
                                                const uint k,
                                                __private u128 *carry,
                                                const uint narrow_w,
                                                const ulong mask_narrow,
                                                const ulong mask_wide,
                                                const uint small31,
                                                const uint pair_mask) {
    const uint base = 3u * k;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    ulong n0 = digit_adc_two_width(make_u128(s0, 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(make_u128(s1, 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    zraw[base + 0u] = n0;
    zraw[base + 1u] = n1;
    zraw[base + 2u] = (small31 != 0u) ? pack_u32x2_dev((u32)n0, (u32)n1) : pack_u32x2_dev(mod31_u64(n0), mod31_u64(n1));
    return (carry->lo | carry->hi) == 0ul;
}


inline void normalize_pair_from_garner_direct_mask_to_local(__global const ulong* restrict zraw,
                                                            __local ulong* restrict ls0,
                                                            __local ulong* restrict ls1,
                                                            __local ulong* restrict lp,
                                                            const uint idx,
                                                            const uint k,
                                                            __private u128 *carry,
                                                            const uint narrow_w,
                                                            const ulong mask_narrow,
                                                            const ulong mask_wide,
                                                            const uint small31,
                                                            const uint pair_mask) {
    const uint base = 3u * k;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    const ulong p  = zraw[base + 2u];
    const u32 t0 = (u32)p, t1 = (u32)(p >> 32);
    GF zk = { s0, s1, t0, t1 };
    u128 L0, L1;
    garner_GF(zk, &L0, &L1);
    ulong n0 = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    ls0[idx] = n0;
    ls1[idx] = n1;
    lp[idx] = (small31 != 0u) ? pack_u32x2_dev((u32)n0, (u32)n1) : pack_u32x2_dev(mod31_u64(n0), mod31_u64(n1));
}

inline uint apply_carry_to_pair_direct_mask_local(__local ulong* restrict ls0,
                                                  __local ulong* restrict ls1,
                                                  __local ulong* restrict lp,
                                                  const uint idx,
                                                  __private u128 *carry,
                                                  const uint narrow_w,
                                                  const ulong mask_narrow,
                                                  const ulong mask_wide,
                                                  const uint small31,
                                                  const uint pair_mask) {
    ulong n0 = digit_adc_two_width(make_u128(ls0[idx], 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ulong n1 = digit_adc_two_width(make_u128(ls1[idx], 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    ls0[idx] = n0;
    ls1[idx] = n1;
    lp[idx] = (small31 != 0u) ? pack_u32x2_dev((u32)n0, (u32)n1) : pack_u32x2_dev(mod31_u64(n0), mod31_u64(n1));
    return (carry->lo | carry->hi) == 0ul;
}


inline void normalize_pair_from_garner_direct_mask_to_local_small31(__global const ulong* restrict zraw,
                                                                    __local ulong* restrict ls0,
                                                                    __local ulong* restrict ls1,
                                                                    const uint idx,
                                                                    const uint k,
                                                                    __private u128 *carry,
                                                                    const uint narrow_w,
                                                                    const ulong mask_narrow,
                                                                    const ulong mask_wide,
                                                                    const uint pair_mask) {
    const uint base = 3u * k;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    u128 L0, L1;
    garner_raw_small31_lowwords(s0, s1, &L0, &L1);
    ls0[idx] = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ls1[idx] = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
}

inline uint apply_carry_to_pair_direct_mask_local_small31(__local ulong* restrict ls0,
                                                          __local ulong* restrict ls1,
                                                          const uint idx,
                                                          __private u128 *carry,
                                                          const uint narrow_w,
                                                          const ulong mask_narrow,
                                                          const ulong mask_wide,
                                                          const uint pair_mask) {
    ls0[idx] = digit_adc_two_width(make_u128(ls0[idx], 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ls1[idx] = digit_adc_two_width(make_u128(ls1[idx], 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    return (carry->lo | carry->hi) == 0ul;
}

inline uint apply_carry_to_pair_direct_mask_raw_small31(__global ulong* restrict zraw,
                                                        const uint k,
                                                        __private u128 *carry,
                                                        const uint narrow_w,
                                                        const ulong mask_narrow,
                                                        const ulong mask_wide,
                                                        const uint pair_mask) {
    const uint base = 3u * k;
    const ulong s0 = zraw[base + 0u];
    const ulong s1 = zraw[base + 1u];
    const ulong n0 = digit_adc_two_width(make_u128(s0, 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    const ulong n1 = digit_adc_two_width(make_u128(s1, 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    zraw[base + 0u] = n0;
    zraw[base + 1u] = n1;
    return (carry->lo | carry->hi) == 0ul;
}


inline void normalize_pair_from_garner_direct_mask_to_local_small31_compact(__global const ulong2* restrict zc,
                                                                            __local ulong* restrict ls0,
                                                                            __local ulong* restrict ls1,
                                                                            const uint idx,
                                                                            const uint k,
                                                                            __private u128 *carry,
                                                                            const uint narrow_w,
                                                                            const ulong mask_narrow,
                                                                            const ulong mask_wide,
                                                                            const uint pair_mask) {
    const ulong2 ss = zc[k];
    u128 L0, L1;
    garner_raw_small31_lowwords(ss.s0, ss.s1, &L0, &L1);
    ls0[idx] = digit_adc_two_width(L0, narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    ls1[idx] = digit_adc_two_width(L1, narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
}

inline uint apply_carry_to_pair_direct_mask_raw_small31_compact(__global ulong2* restrict zc,
                                                                const uint k,
                                                                __private u128 *carry,
                                                                const uint narrow_w,
                                                                const ulong mask_narrow,
                                                                const ulong mask_wide,
                                                                const uint pair_mask) {
    const ulong2 ss = zc[k];
    const ulong n0 = digit_adc_two_width(make_u128(ss.s0, 0ul), narrow_w, mask_narrow, mask_wide, carry, pair_mask & 1u);
    const ulong n1 = digit_adc_two_width(make_u128(ss.s1, 0ul), narrow_w, mask_narrow, mask_wide, carry, (pair_mask >> 1) & 1u);
    zc[k] = (ulong2)(n0, n1);
    return (carry->lo | carry->hi) == 0ul;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void block_prepare_direct8_mask_fused64_small31_compact_kernel(__global ulong2* restrict zc,
                                                               __global const ushort* restrict block_wide_mask,
                                                               __global CarryWord* restrict group_tail_carry,
                                                               const uint ngroups,
                                                               const uint narrow_w) {
    const uint g = get_group_id(0);
    const uint lid = get_local_id(0);
    if (g >= ngroups) return;

    __local ulong ls0[64u * 8u];
    __local ulong ls1[64u * 8u];
    __local ulong clo[64u];
    __local ulong chi[64u];
#ifdef DIRECT8_SMALL31_NW
    const uint hot_narrow_w = (uint)DIRECT8_SMALL31_NW;
#else
    const uint hot_narrow_w = narrow_w;
#endif
    const ulong mask_narrow = ((ulong)1 << hot_narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = (g << 6) + lid;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    const uint lbase = lid << 3;

    u128 c = make_u128(0ul, 0ul);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 0u, k + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 1u, k + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 2u, k + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 3u, k + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 4u, k + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 5u, k + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 6u, k + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31_compact(zc, ls0, ls1, lbase + 7u, k + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
    clo[lid] = c.lo;
    chi[lid] = c.hi;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0u) {
        c = make_u128(clo[lid - 1u], chi[lid - 1u]);
        if ((c.lo | c.hi) != 0ul) {
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u) == 0u)
                (void)apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
        }
    }

    zc[k + 0u] = (ulong2)(ls0[lbase + 0u], ls1[lbase + 0u]);
    zc[k + 1u] = (ulong2)(ls0[lbase + 1u], ls1[lbase + 1u]);
    zc[k + 2u] = (ulong2)(ls0[lbase + 2u], ls1[lbase + 2u]);
    zc[k + 3u] = (ulong2)(ls0[lbase + 3u], ls1[lbase + 3u]);
    zc[k + 4u] = (ulong2)(ls0[lbase + 4u], ls1[lbase + 4u]);
    zc[k + 5u] = (ulong2)(ls0[lbase + 5u], ls1[lbase + 5u]);
    zc[k + 6u] = (ulong2)(ls0[lbase + 6u], ls1[lbase + 6u]);
    zc[k + 7u] = (ulong2)(ls0[lbase + 7u], ls1[lbase + 7u]);

    if (lid == 63u) {
        group_tail_carry[g].lo = clo[63u];
        group_tail_carry[g].hi = chi[63u];
    }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void block_apply_group_head_direct8_mask_small31_compact_kernel(__global ulong2* restrict zc,
                                                                __global const ushort* restrict block_wide_mask,
                                                                __global const CarryWord* restrict group_tail_carry,
                                                                const uint ngroups,
                                                                const uint narrow_w) {
    uint g = get_global_id(0);
    if (g >= ngroups) return;
    uint prevg = (g == 0u) ? (ngroups - 1u) : (g - 1u);
    u128 c = make_u128(group_tail_carry[prevg].lo, group_tail_carry[prevg].hi);
    if ((c.lo | c.hi) == 0ul) return;
#ifdef DIRECT8_SMALL31_NW
    const uint hot_narrow_w = (uint)DIRECT8_SMALL31_NW;
#else
    const uint hot_narrow_w = narrow_w;
#endif
    const ulong mask_narrow = ((ulong)1 << hot_narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = g << 6;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u) != 0u) return;
    (void)apply_carry_to_pair_direct_mask_raw_small31_compact(zc, k + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void block_prepare_direct8_mask_fused64_kernel(__global GF* restrict z,
                                               __global const ushort* restrict block_wide_mask,
                                               __global CarryWord* restrict group_tail_carry,
                                               const uint ngroups,
                                               const uint narrow_w,
                                               const uint small31) {
    const uint g = get_group_id(0);
    const uint lid = get_local_id(0);
    if (g >= ngroups) return;

    __global ulong* restrict zraw = (__global ulong*)z;
    __local ulong ls0[64u * 8u];
    __local ulong ls1[64u * 8u];
    __local ulong lp[64u * 8u];
    __local ulong clo[64u];
    __local ulong chi[64u];

    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = (g << 6) + lid;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    const uint lbase = lid << 3;

    u128 c = make_u128(0ul, 0ul);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 0u, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 0) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 1u, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 2) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 2u, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 4) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 3u, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 6) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 4u, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 8) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 5u, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 10) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 6u, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 12) & 3u);
    normalize_pair_from_garner_direct_mask_to_local(zraw, ls0, ls1, lp, lbase + 7u, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 14) & 3u);
    clo[lid] = c.lo;
    chi[lid] = c.hi;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0u) {
        c = make_u128(clo[lid - 1u], chi[lid - 1u]);
        if ((c.lo | c.hi) != 0ul) {
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 0u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 0) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 1u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 2) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 2u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 4) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 3u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 6) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 4u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 8) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 5u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 10) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 6u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 12) & 3u) == 0u)
                (void)apply_carry_to_pair_direct_mask_local(ls0, ls1, lp, lbase + 7u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 14) & 3u);
        }
    }

    zraw[3u * (k + 0u) + 0u] = ls0[lbase + 0u]; zraw[3u * (k + 0u) + 1u] = ls1[lbase + 0u]; zraw[3u * (k + 0u) + 2u] = lp[lbase + 0u];
    zraw[3u * (k + 1u) + 0u] = ls0[lbase + 1u]; zraw[3u * (k + 1u) + 1u] = ls1[lbase + 1u]; zraw[3u * (k + 1u) + 2u] = lp[lbase + 1u];
    zraw[3u * (k + 2u) + 0u] = ls0[lbase + 2u]; zraw[3u * (k + 2u) + 1u] = ls1[lbase + 2u]; zraw[3u * (k + 2u) + 2u] = lp[lbase + 2u];
    zraw[3u * (k + 3u) + 0u] = ls0[lbase + 3u]; zraw[3u * (k + 3u) + 1u] = ls1[lbase + 3u]; zraw[3u * (k + 3u) + 2u] = lp[lbase + 3u];
    zraw[3u * (k + 4u) + 0u] = ls0[lbase + 4u]; zraw[3u * (k + 4u) + 1u] = ls1[lbase + 4u]; zraw[3u * (k + 4u) + 2u] = lp[lbase + 4u];
    zraw[3u * (k + 5u) + 0u] = ls0[lbase + 5u]; zraw[3u * (k + 5u) + 1u] = ls1[lbase + 5u]; zraw[3u * (k + 5u) + 2u] = lp[lbase + 5u];
    zraw[3u * (k + 6u) + 0u] = ls0[lbase + 6u]; zraw[3u * (k + 6u) + 1u] = ls1[lbase + 6u]; zraw[3u * (k + 6u) + 2u] = lp[lbase + 6u];
    zraw[3u * (k + 7u) + 0u] = ls0[lbase + 7u]; zraw[3u * (k + 7u) + 1u] = ls1[lbase + 7u]; zraw[3u * (k + 7u) + 2u] = lp[lbase + 7u];

    if (lid == 63u) {
        group_tail_carry[g].lo = clo[63u];
        group_tail_carry[g].hi = chi[63u];
    }
}



__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void block_prepare_direct8_mask_fused64_small31_kernel(__global GF* restrict z,
                                                       __global const ushort* restrict block_wide_mask,
                                                       __global CarryWord* restrict group_tail_carry,
                                                       const uint ngroups,
                                                       const uint narrow_w) {
    const uint g = get_group_id(0);
    const uint lid = get_local_id(0);
    if (g >= ngroups) return;

    __global ulong* restrict zraw = (__global ulong*)z;
    __local ulong ls0[64u * 8u];
    __local ulong ls1[64u * 8u];
    __local ulong clo[64u];
    __local ulong chi[64u];

#ifdef DIRECT8_SMALL31_NW
    const uint hot_narrow_w = (uint)DIRECT8_SMALL31_NW;
#else
    const uint hot_narrow_w = narrow_w;
#endif
    const ulong mask_narrow = ((ulong)1 << hot_narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = (g << 6) + lid;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    const uint lbase = lid << 3;

    u128 c = make_u128(0ul, 0ul);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 0u, k + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 1u, k + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 2u, k + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 3u, k + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 4u, k + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 5u, k + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 6u, k + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u);
    normalize_pair_from_garner_direct_mask_to_local_small31(zraw, ls0, ls1, lbase + 7u, k + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
    clo[lid] = c.lo;
    chi[lid] = c.hi;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0u) {
        c = make_u128(clo[lid - 1u], chi[lid - 1u]);
        if ((c.lo | c.hi) != 0ul) {
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u) == 0u)
            if (apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u) == 0u)
                (void)apply_carry_to_pair_direct_mask_local_small31(ls0, ls1, lbase + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
        }
    }

    zraw[3u * (k + 0u) + 0u] = ls0[lbase + 0u]; zraw[3u * (k + 0u) + 1u] = ls1[lbase + 0u];
    zraw[3u * (k + 1u) + 0u] = ls0[lbase + 1u]; zraw[3u * (k + 1u) + 1u] = ls1[lbase + 1u];
    zraw[3u * (k + 2u) + 0u] = ls0[lbase + 2u]; zraw[3u * (k + 2u) + 1u] = ls1[lbase + 2u];
    zraw[3u * (k + 3u) + 0u] = ls0[lbase + 3u]; zraw[3u * (k + 3u) + 1u] = ls1[lbase + 3u];
    zraw[3u * (k + 4u) + 0u] = ls0[lbase + 4u]; zraw[3u * (k + 4u) + 1u] = ls1[lbase + 4u];
    zraw[3u * (k + 5u) + 0u] = ls0[lbase + 5u]; zraw[3u * (k + 5u) + 1u] = ls1[lbase + 5u];
    zraw[3u * (k + 6u) + 0u] = ls0[lbase + 6u]; zraw[3u * (k + 6u) + 1u] = ls1[lbase + 6u];
    zraw[3u * (k + 7u) + 0u] = ls0[lbase + 7u]; zraw[3u * (k + 7u) + 1u] = ls1[lbase + 7u];

    if (lid == 63u) {
        group_tail_carry[g].lo = clo[63u];
        group_tail_carry[g].hi = chi[63u];
    }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void block_apply_group_head_direct8_mask_small31_kernel(__global GF* restrict z,
                                                                 __global const ushort* restrict block_wide_mask,
                                                                 __global const CarryWord* restrict group_tail_carry,
                                                                 const uint ngroups,
                                                                 const uint narrow_w) {
    uint g = get_global_id(0);
    if (g >= ngroups) return;
    uint prevg = (g == 0u) ? (ngroups - 1u) : (g - 1u);
    u128 c = make_u128(group_tail_carry[prevg].lo, group_tail_carry[prevg].hi);
    if ((c.lo | c.hi) == 0ul) return;
    __global ulong* restrict zraw = (__global ulong*)z;
#ifdef DIRECT8_SMALL31_NW
    const uint hot_narrow_w = (uint)DIRECT8_SMALL31_NW;
#else
    const uint hot_narrow_w = narrow_w;
#endif
    const ulong mask_narrow = ((ulong)1 << hot_narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = g << 6;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 0u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 0) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 1u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 2) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 2u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 4) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 3u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 6) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 4u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 8) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 5u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 10) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 6u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 12) & 3u) != 0u) return;
    (void)apply_carry_to_pair_direct_mask_raw_small31(zraw, k + 7u, &c, hot_narrow_w, mask_narrow, mask_wide, (bm >> 14) & 3u);
}
__kernel void block_apply_group_head_direct8_mask_kernel(__global GF* restrict z,
                                                         __global const ushort* restrict block_wide_mask,
                                                         __global const CarryWord* restrict group_tail_carry,
                                                         const uint ngroups,
                                                         const uint narrow_w,
                                                         const uint small31) {
    uint g = get_global_id(0);
    if (g >= ngroups) return;
    uint prevg = (g == 0u) ? (ngroups - 1u) : (g - 1u);
    u128 c = make_u128(group_tail_carry[prevg].lo, group_tail_carry[prevg].hi);
    if ((c.lo | c.hi) == 0ul) return;
    __global ulong* restrict zraw = (__global ulong*)z;
    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint b = g << 6;
    const uint k = b << 3;
    const uint bm = block_wide_mask[b];
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 0) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 2) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 4) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 6) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 8) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 10) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 12) & 3u) != 0u) return;
    (void)apply_carry_to_pair_direct_mask_raw(zraw, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 14) & 3u);
}

__kernel void block_prepare_direct8_mask_kernel(__global GF* restrict z,
                                                __global const ushort* restrict block_wide_mask,
                                                __global CarryWord* restrict carry_next,
                                                const uint nblocks,
                                                const uint narrow_w,
                                                const uint small31) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    __global ulong* restrict zraw = (__global ulong*)z;
    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint k = b << 3;
    const uint bm = (uint)block_wide_mask[b];
    u128 c = make_u128(0ul, 0ul);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 0) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 2) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 4) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 6) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 8) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 10) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 12) & 3u);
    normalize_pair_from_garner_direct_mask_raw(zraw, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 14) & 3u);
    uint dst = (b + 1u < nblocks) ? (b + 1u) : 0u;
    carry_next[dst].lo = c.lo;
    carry_next[dst].hi = c.hi;
}

__kernel void block_apply_carry_direct8_mask_kernel(__global GF* restrict z,
                                                    __global const ushort* restrict block_wide_mask,
                                                    __global const CarryWord* restrict carry_in,
                                                    const uint nblocks,
                                                    const uint narrow_w,
                                                    const uint small31) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    u128 c = make_u128(carry_in[b].lo, carry_in[b].hi);
    if ((c.lo | c.hi) == 0ul) return;
    __global ulong* restrict zraw = (__global ulong*)z;
    const ulong mask_narrow = ((ulong)1 << narrow_w) - 1ul;
    const ulong mask_wide = (mask_narrow << 1) | 1ul;
    const uint k = b << 3;
    const uint bm = (uint)block_wide_mask[b];
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 0u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 0) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 1u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 2) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 2u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 4) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 3u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 6) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 4u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 8) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 5u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 10) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 6u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 12) & 3u) != 0u) return;
    if (apply_carry_to_pair_direct_mask_raw(zraw, k + 7u, &c, narrow_w, mask_narrow, mask_wide, small31, (bm >> 14) & 3u) != 0u) return;
}

__kernel void block_apply_prev_carry_kernel(__global GF* restrict z,
                                          __global const BlockInfo* restrict blocks,
                                          __global const BlockState* restrict states,
                                          __global const uchar* restrict digit_width,
                                          const uint nblocks) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    uint prev = (b == 0u) ? (nblocks - 1u) : (b - 1u);
    u128 c = make_u128(states[prev].q_lo, states[prev].q_hi);
    if ((c.lo | c.hi) == 0ul) return;
    BlockInfo bi = blocks[b];
    for (uint t = 0; t < bi.count; ++t) {
        uint k = bi.start + t;
        GF zk = z[k];
        uchar2 wd = digit_width2[k];
        uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
        ulong n0 = digit_adc(make_u128(zk.s0, 0ul), w0, &c);
        ulong n1 = digit_adc(make_u128(zk.s1, 0ul), w1, &c);
        GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
        z[k] = out;
        if ((c.lo | c.hi) == 0ul) break;
    }
}

__kernel void carry_wrap_kernel(__global GF* restrict z,
                                __global const uchar* restrict digit_width,
                                const uint n2,
                                __global const CarryWord* restrict final_carry) {
    if (get_global_id(0) != 0) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    u128 c = make_u128(final_carry[0].lo, final_carry[0].hi);
    while ((c.lo | c.hi) != 0ul) {
        for (uint k = 0; k < n2; ++k) {
            u128 L0 = make_u128(z[k].s0, 0ul), L1 = make_u128(z[k].s1, 0ul);
            uchar2 wd = digit_width2[k];
            uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
            ulong n0 = digit_adc(L0, w0, &c);
            ulong n1 = digit_adc(L1, w1, &c);
            GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
            z[k] = out;
            if ((c.lo | c.hi) == 0ul) break;
        }
    }
}

__kernel void init_block_carry_kernel(__global const BlockState* restrict states,
                                      __global CarryWord* restrict carry,
                                      const uint nblocks) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    uint dst = (b + 1u < nblocks) ? (b + 1u) : 0u;
    carry[dst].lo = states[b].q_lo;
    carry[dst].hi = states[b].q_hi;
}

__kernel void block_carry_phase_kernel(__global GF* restrict z,
                                       __global const BlockInfo* restrict blocks,
                                       __global CarryWord* restrict carry,
                                       __global const uchar* restrict digit_width,
                                       const uint nblocks,
                                       const uint phase) {
    uint b = get_global_id(0);
    if (b >= nblocks) return;
    if ((b & 1u) != (phase & 1u)) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    u128 c = make_u128(carry[b].lo, carry[b].hi);
    if ((c.lo | c.hi) == 0ul) return;
    carry[b].lo = 0ul;
    carry[b].hi = 0ul;
    BlockInfo bi = blocks[b];
    for (uint t = 0; t < bi.count; ++t) {
        uint k = bi.start + t;
        GF zk = z[k];
        uchar2 wd = digit_width2[k];
        uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
        ulong n0 = digit_adc(make_u128(zk.s0, 0ul), w0, &c);
        ulong n1 = digit_adc(make_u128(zk.s1, 0ul), w1, &c);
        GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
        z[k] = out;
        if ((c.lo | c.hi) == 0ul) break;
    }
    if ((c.lo | c.hi) != 0ul) {
        uint next = (b + 1u < nblocks) ? (b + 1u) : 0u;
        u128 nxt = make_u128(carry[next].lo, carry[next].hi);
        nxt = add_u128(nxt, c);
        carry[next].lo = nxt.lo;
        carry[next].hi = nxt.hi;
    }
}

__kernel void block_carry_drain_kernel(__global GF* restrict z,
                                       __global const BlockInfo* restrict blocks,
                                       __global CarryWord* restrict carry,
                                       __global const uchar* restrict digit_width,
                                       const uint nblocks) {
    if (get_global_id(0) != 0) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    uint any = 1u;
    while (any != 0u) {
        any = 0u;
        for (uint b = 0; b < nblocks; ++b) {
            u128 c = make_u128(carry[b].lo, carry[b].hi);
            if ((c.lo | c.hi) == 0ul) continue;
            carry[b].lo = 0ul;
            carry[b].hi = 0ul;
            any = 1u;
            BlockInfo bi = blocks[b];
            for (uint t = 0; t < bi.count; ++t) {
                uint k = bi.start + t;
                GF zk = z[k];
                uchar2 wd = digit_width2[k];
                uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
                ulong n0 = digit_adc(make_u128(zk.s0, 0ul), w0, &c);
                ulong n1 = digit_adc(make_u128(zk.s1, 0ul), w1, &c);
                GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
                z[k] = out;
                if ((c.lo | c.hi) == 0ul) break;
            }
            if ((c.lo | c.hi) != 0ul) {
                uint next = (b + 1u < nblocks) ? (b + 1u) : 0u;
                u128 nxt = make_u128(carry[next].lo, carry[next].hi);
                nxt = add_u128(nxt, c);
                carry[next].lo = nxt.lo;
                carry[next].hi = nxt.hi;
            }
        }
    }
}

__kernel void sub_kernel(__global GF* restrict z,
                         __global const uchar* restrict digit_width,
                         const uint n2,
                         const uint a) {
    if (get_global_id(0) != 0) return;
    __global const uchar2* restrict digit_width2 = (__global const uchar2*)digit_width;
    uint borrow = a;
    while (borrow != 0u) {
        for (uint k = 0; k < n2; ++k) {
            uchar2 wd = digit_width2[k];
            uint w0 = (uint)wd.s0, w1 = (uint)wd.s1;
            ulong n0 = digit_sbc(z[k].s0, w0, &borrow);
            ulong n1 = digit_sbc(z[k].s1, w1, &borrow);
            GF out = { n0, n1, mod31_u64(n0), mod31_u64(n1) };
            z[k] = out;
            if (borrow == 0u) break;
        }
    }
}

)CLC";

static bool verify_is_zero(const std::vector<GF61_31>& host_z) {
    for (const auto& v : host_z) {
        if (v.g61.s0() != 0 || v.g61.s1() != 0) return false;
    }
    return true;
}

static bool verify_is_Mp(const std::vector<GF61_31>& host_z, const std::vector<uint8_t>& digit_width) {
    for (size_t k = 0; k < host_z.size(); ++k) {
        uint64_t mask0 = (uint64_t(1) << digit_width[2*k]) - 1ull;
        uint64_t mask1 = (uint64_t(1) << digit_width[2*k+1]) - 1ull;
        if (host_z[k].g61.s0() != mask0 || host_z[k].g61.s1() != mask1) return false;
    }
    return true;
}

static uint64_t digit_sbc_host(uint64_t lhs, uint8_t w, uint32_t & borrow) {
    const bool b = (lhs < borrow);
    const uint64_t r = lhs - borrow + (b ? (uint64_t(1) << w) : 0ull);
    borrow = b ? 1u : 0u;
    return r;
}

static std::vector<GF61_31> subtract_a(const std::vector<GF61_31>& host_z, const std::vector<uint8_t>& digit_width, uint32_t a) {
    std::vector<GF61_31> tmp = host_z;
    uint32_t borrow = a;
    while (borrow != 0u) {
        for (size_t k = 0; k < tmp.size(); ++k) {
            uint64_t s0 = tmp[k].g61.s0();
            uint64_t s1 = tmp[k].g61.s1();
            s0 = digit_sbc_host(s0, digit_width[2*k], borrow);
            s1 = digit_sbc_host(s1, digit_width[2*k + 1], borrow);
            tmp[k] = GF61_31(s0, s1);
            if (borrow == 0u) break;
        }
    }
    return tmp;
}

static bool verify_equals(const std::vector<GF61_31>& host_z, const std::vector<uint8_t>& digit_width, uint32_t a) {
    auto tmp = subtract_a(host_z, digit_width, a);
    return verify_is_zero(tmp);
}

static bool verify_prp_residue_9(const std::vector<GF61_31>& host_z, const std::vector<uint8_t>& digit_width) {
    if (verify_equals(host_z, digit_width, 9u)) return true;
    auto z_minus_9 = subtract_a(host_z, digit_width, 9u);
    return verify_is_Mp(z_minus_9, digit_width);
}

static void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::cerr << what << " failed with error " << err << std::endl;
        std::exit(1);
    }
}

enum class TestMode { LL, PRP };

static TestMode parse_mode(const char* s) {
    if (s == nullptr) return TestMode::LL;
    std::string m(s);
    for (char& c : m) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (m == "ll") return TestMode::LL;
    if (m == "prp") return TestMode::PRP;
    std::cerr << "Unknown mode '" << s << "' (expected ll or prp)\n";
    std::exit(1);
}

static size_t round_up(size_t value, size_t group) {
    if (group == 0) return value;
    return ((value + group - 1) / group) * group;
}

static std::string lower_ascii(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

struct AutoTuneChoice {
    uint32_t wg;
    uint32_t carry_wg;
    uint32_t carry_pairs;
    std::string profile;
};

static uint32_t clamp_carry_pairs(uint32_t x, uint32_t h_u32) {
    if (h_u32 == 0u) return 1u;
    x = std::max<uint32_t>(1u, std::min<uint32_t>(x, h_u32));
    uint32_t p = 1u;
    while ((p << 1) != 0u && (p << 1) <= x) p <<= 1;
    return std::min<uint32_t>(p, h_u32);
}

static uint32_t choose_wg_for_device(size_t max_wg, const std::string& dev_lower, const std::string& vendor_lower, cl_uint compute_units) {
    const bool is_amd = vendor_lower.find("amd") != std::string::npos || vendor_lower.find("advanced micro devices") != std::string::npos || dev_lower.find("gfx") != std::string::npos || dev_lower.find("radeon") != std::string::npos;
    const bool is_nvidia = vendor_lower.find("nvidia") != std::string::npos || dev_lower.find("geforce") != std::string::npos || dev_lower.find("rtx") != std::string::npos || dev_lower.find("gtx") != std::string::npos;
    const bool is_apple = vendor_lower.find("apple") != std::string::npos || dev_lower.find("apple") != std::string::npos || dev_lower.find("m1") != std::string::npos || dev_lower.find("m2") != std::string::npos || dev_lower.find("m3") != std::string::npos;
    const bool is_intel = vendor_lower.find("intel") != std::string::npos || dev_lower.find("intel") != std::string::npos;

    size_t wg = 64u;
    if (is_amd) {
        wg = 64u;
        if (compute_units >= 96u && max_wg >= 128u && dev_lower.find("cdna") != std::string::npos) wg = 128u;
    } else if (is_nvidia) {
        wg = (max_wg >= 128u) ? 128u : (max_wg >= 64u ? 64u : 32u);
    } else if (is_apple) {
        wg = (max_wg >= 64u) ? 64u : 32u;
    } else if (is_intel) {
        wg = (max_wg >= 128u) ? 128u : (max_wg >= 64u ? 64u : 32u);
    } else {
        wg = (max_wg >= 64u) ? 64u : std::max<size_t>(1u, max_wg);
    }
    while (wg > max_wg && wg > 1u) wg >>= 1;
    return static_cast<uint32_t>(std::max<size_t>(1u, wg));
}

static AutoTuneChoice choose_auto_tune(uint32_t h_u32, size_t max_wg, cl_ulong local_mem_size,
                                       cl_uint compute_units, const std::string& device_name,
                                       const std::string& vendor_name) {
    const std::string dev_lower = lower_ascii(device_name);
    const std::string vendor_lower = lower_ascii(vendor_name);
    const bool is_amd = vendor_lower.find("amd") != std::string::npos || vendor_lower.find("advanced micro devices") != std::string::npos || dev_lower.find("gfx") != std::string::npos || dev_lower.find("radeon") != std::string::npos;
    const bool is_nvidia = vendor_lower.find("nvidia") != std::string::npos || dev_lower.find("geforce") != std::string::npos || dev_lower.find("rtx") != std::string::npos || dev_lower.find("gtx") != std::string::npos;
    const bool is_apple = vendor_lower.find("apple") != std::string::npos || dev_lower.find("apple") != std::string::npos || dev_lower.find("m1") != std::string::npos || dev_lower.find("m2") != std::string::npos || dev_lower.find("m3") != std::string::npos;
    const bool is_gfx9 = (dev_lower.find("gfx9") != std::string::npos || dev_lower.find("gfx90") != std::string::npos || dev_lower.find("gfx906") != std::string::npos || dev_lower.find("vega") != std::string::npos || dev_lower.find("radeon vii") != std::string::npos);

    AutoTuneChoice r{};
    r.wg = choose_wg_for_device(max_wg, dev_lower, vendor_lower, compute_units);
    r.carry_wg = r.wg;
    r.carry_pairs = 64u;
    r.profile = "generic-auto";

    if (is_amd) {
        r.profile = is_gfx9 ? "amd-wave64-smallblocks" : "amd-wave64-smallblocks";
        // Large transforms benefit from finer carry block granularity: the direct block-carry path
        // is already parallel across blocks, and smaller blocks reduce per-workitem serial digit_adc work.
        // Keep the very small cases conservative to avoid unnecessary carry setup overhead.
        if (h_u32 >= (1u << 20)) r.carry_pairs = 8u;
        else if (h_u32 >= (1u << 18)) r.carry_pairs = 16u;
        else if (h_u32 >= (1u << 16)) r.carry_pairs = 32u;
        else if (h_u32 >= (1u << 14)) r.carry_pairs = 64u;
        else if (h_u32 >= (1u << 12)) r.carry_pairs = 128u;
        else r.carry_pairs = std::min<uint32_t>(256u, std::max<uint32_t>(1u, h_u32));
    } else if (is_nvidia) {
        r.profile = "nvidia-warp32-splitwg";
        // Empirical default tuned from Tesla T4-style results:
        // for medium transforms around 2^18 words, FFT is faster at 64 than 128,
        // while carry still prefers 64-thread groups and fine carry blocks.
        // Keep larger jobs on 128 by default unless the device looks like a smaller
        // 48KB-local-memory NVIDIA part where 64 remains a safer starting point.
        const bool nvidia_small_local = (local_mem_size <= 49152u);
        // Safer default: keep 64 only in the empirically validated T4 window around h=2^18..2^19.
        // Smaller sizes and most other cases stay on 128 by default to avoid correctness/launch regressions.
        if (h_u32 >= (1u << 18) && h_u32 <= (1u << 19) && nvidia_small_local) {
            r.wg = (max_wg >= 64u) ? 64u : (max_wg >= 32u ? 32u : 16u);
        } else {
            r.wg = (max_wg >= 128u) ? 128u : (max_wg >= 64u ? 64u : 32u);
        }
        r.carry_wg = (max_wg >= 64u) ? 64u : (max_wg >= 32u ? 32u : 16u);
        if (h_u32 >= (1u << 20)) r.carry_pairs = 8u;
        else if (h_u32 >= (1u << 18)) r.carry_pairs = 8u;
        else if (h_u32 >= (1u << 16)) r.carry_pairs = 16u;
        else if (h_u32 >= (1u << 14)) r.carry_pairs = 32u;
        else r.carry_pairs = 32u;
    } else if (is_apple) {
        r.profile = "apple-tile";
        if (local_mem_size <= 32768u) r.carry_pairs = (h_u32 >= (1u << 14)) ? 64u : 32u;
        else r.carry_pairs = (h_u32 >= (1u << 18)) ? 128u : ((h_u32 >= (1u << 14)) ? 64u : 32u);
    } else {
        r.profile = "generic-opencl";
        r.carry_pairs = (h_u32 >= (1u << 18)) ? 128u : ((h_u32 >= (1u << 14)) ? 64u : 32u);
    }

    r.carry_pairs = clamp_carry_pairs(r.carry_pairs, h_u32);
    return r;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <p> [ll|prp] [report_every] [-d device_id] [-R report_seconds] [-W fft_wg] [-CW carry_wg] [-B carry_pairs] [-LS local_stage_cap] [-N max_iters] [--show-defaults]\n";
        return 1;
    }

    const uint32_t p = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    TestMode mode = TestMode::LL;
    uint32_t report_every = 1000u;
    double report_seconds = 2.0;
    uint32_t device_index = 0u;
    uint32_t wg_override = 0u;
    uint32_t carry_wg_override = 0u;
    uint32_t carry_pairs_override = 0u;
    uint32_t local_stage_cap_override = 0u;
    uint32_t max_iters_override = 0u;
    bool show_defaults_only = false;
    bool report_every_set = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "ll" || arg == "prp") {
            mode = parse_mode(arg.c_str());
        } else if (arg == "-d") {
            if (i + 1 >= argc) {
                std::cerr << "Missing device id after -d\n";
                return 1;
            }
            device_index = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "-R") {
            if (i + 1 >= argc) {
                std::cerr << "Missing seconds after -R\n";
                return 1;
            }
            report_seconds = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "-W") {
            if (i + 1 >= argc) {
                std::cerr << "Missing work-group size after -W\n";
                return 1;
            }
            wg_override = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "-CW") {
            if (i + 1 >= argc) {
                std::cerr << "Missing carry work-group size after -CW\n";
                return 1;
            }
            carry_wg_override = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "-B") {
            if (i + 1 >= argc) {
                std::cerr << "Missing carry_pairs after -B\n";
                return 1;
            }
            carry_pairs_override = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "-LS") {
            if (i + 1 >= argc) {
                std::cerr << "Missing local stage cap after -LS\n";
                return 1;
            }
            local_stage_cap_override = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "-N") {
            if (i + 1 >= argc) {
                std::cerr << "Missing max iterations after -N\n";
                return 1;
            }
            max_iters_override = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--show-defaults") {
            show_defaults_only = true;
        } else if (!report_every_set) {
            report_every = static_cast<uint32_t>(std::strtoul(arg.c_str(), nullptr, 10));
            report_every_set = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " <p> [ll|prp] [report_every] [-d device_id] [-R report_seconds] [-W fft_wg] [-CW carry_wg] [-B carry_pairs] [-LS local_stage_cap] [-N max_iters] [--show-defaults]\n";
            return 1;
        }
    }

    const uint8_t ln = transformsize(p);
    const size_t n = size_t(1) << ln;
    const size_t h = n >> 1;
    const uint8_t q_n_build = uint8_t(p >> ln);
    const bool direct8_small31_build = (uint32_t(q_n_build) + 1u < 31u);

    std::cout << "p=" << p << ", ln=" << int(ln) << ", transform=" << h << " (n=" << n << ")\n";

    cl_uint pn = 0;
    check(clGetPlatformIDs(0, nullptr, &pn), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> platforms(pn);
    check(clGetPlatformIDs(pn, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    std::vector<cl_device_id> gpu_devices;
    for (cl_platform_id P : platforms) {
        cl_uint dn = 0;
        cl_int derr = clGetDeviceIDs(P, CL_DEVICE_TYPE_GPU, 0, nullptr, &dn);
        if (derr == CL_DEVICE_NOT_FOUND || dn == 0) continue;
        check(derr, "clGetDeviceIDs(count)");
        std::vector<cl_device_id> devs(dn);
        check(clGetDeviceIDs(P, CL_DEVICE_TYPE_GPU, dn, devs.data(), nullptr), "clGetDeviceIDs(list)");
        gpu_devices.insert(gpu_devices.end(), devs.begin(), devs.end());
    }
    if (gpu_devices.empty()) {
        std::cerr << "No GPU OpenCL device found\n";
        return 1;
    }
    if (device_index >= gpu_devices.size()) {
        std::cerr << "Invalid GPU device index " << device_index << " (available 0.." << (gpu_devices.size() - 1) << ")\n";
        return 1;
    }
    cl_device_id D = gpu_devices[device_index];
    char device_name[256] = {0};
    char vendor_name[256] = {0};
    cl_uint compute_units = 0;
    clGetDeviceInfo(D, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(D, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, nullptr);
    std::cout << "device=" << device_index << " (" << device_name << ")\n";

    size_t max_wg = 1;
    cl_ulong local_mem_size = 0;
    check(clGetDeviceInfo(D, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr), "clGetDeviceInfo(MAX_WORK_GROUP_SIZE)");
    check(clGetDeviceInfo(D, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr), "clGetDeviceInfo(LOCAL_MEM_SIZE)");
    check(clGetDeviceInfo(D, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr), "clGetDeviceInfo(MAX_COMPUTE_UNITS)");
    std::string dev_name_str(device_name);
    std::string vendor_name_str(vendor_name);
    const std::string dev_name_lower = lower_ascii(dev_name_str);
    const bool is_gfx9 = (dev_name_lower.find("gfx9") != std::string::npos || dev_name_lower.find("gfx90") != std::string::npos || dev_name_lower.find("gfx906") != std::string::npos || dev_name_lower.find("vega") != std::string::npos || dev_name_lower.find("radeon vii") != std::string::npos);
    const AutoTuneChoice auto_tune = choose_auto_tune(static_cast<uint32_t>(h), max_wg, local_mem_size, compute_units, dev_name_str, vendor_name_str);
    const bool is_nvidia_dev = (lower_ascii(vendor_name_str).find("nvidia") != std::string::npos || dev_name_lower.find("tesla") != std::string::npos || dev_name_lower.find("rtx") != std::string::npos || dev_name_lower.find("gtx") != std::string::npos);
    const bool is_amd_dev = (lower_ascii(vendor_name_str).find("amd") != std::string::npos || lower_ascii(vendor_name_str).find("advanced micro devices") != std::string::npos || dev_name_lower.find("gfx") != std::string::npos || dev_name_lower.find("radeon") != std::string::npos);
    if (!report_every_set) {
        if (is_nvidia_dev) {
            if (h >= (1u << 20)) report_every = 4000u;
            else if (h >= (1u << 18)) report_every = 2000u;
            else report_every = 1000u;
        } else if (is_amd_dev) {
            if (h >= (1u << 20)) report_every = 2000u;
            else report_every = 1000u;
        }
    }
    size_t wg = auto_tune.wg;
    size_t carry_wg = auto_tune.carry_wg;
    if (wg_override != 0u) {
        const size_t o = std::max<size_t>(1u, std::min<size_t>(static_cast<size_t>(wg_override), max_wg));
        wg = o;
        carry_wg = o;
    }
    if (carry_wg_override != 0u) {
        const size_t o = std::max<size_t>(1u, std::min<size_t>(static_cast<size_t>(carry_wg_override), max_wg));
        carry_wg = o;
    }
    if (carry_wg > max_wg) carry_wg = max_wg;
    if (show_defaults_only) {
        const size_t default_local_stage_cap = std::min<size_t>(max_wg, (is_gfx9 || is_nvidia_dev) ? 256u : 128u);
        std::cout << "vendor=" << vendor_name << ", compute_units=" << compute_units
                  << ", max_wg=" << max_wg << ", local_mem=" << local_mem_size << "\n"
                  << "auto_wg=" << auto_tune.wg << ", auto_carry_wg=" << auto_tune.carry_wg
                  << ", auto_carry_pairs=" << auto_tune.carry_pairs << ", auto_profile=" << auto_tune.profile << "\n"
                  << "effective_wg=" << wg << ", effective_carry_wg=" << carry_wg
                  << ", effective_local_stage_cap=" << ((local_stage_cap_override != 0u) ? std::min<size_t>(max_wg, local_stage_cap_override) : default_local_stage_cap)
                  << ", effective_max_iters=" << max_iters_override << "\n";
        return 0;
    }

    cl_int err = CL_SUCCESS;
    cl_context C = clCreateContext(nullptr, 1, &D, nullptr, nullptr, &err); check(err, "clCreateContext");
#if defined(__APPLE__)
    cl_command_queue Q = clCreateCommandQueue(C, D, 0, &err);
#else
    cl_command_queue Q = clCreateCommandQueueWithProperties(C, D, nullptr, &err);
#endif
    check(err, "clCreateCommandQueue");

    std::string build_opts_str = "-cl-std=CL1.2";
    if (direct8_small31_build) build_opts_str += " -DDIRECT8_SMALL31_NW=" + std::to_string(unsigned(q_n_build));
    const char* build_opts = build_opts_str.c_str();
    cl_program PR = clCreateProgramWithSource(C, 1, &KC, nullptr, &err); check(err, "clCreateProgramWithSource");
    err = clBuildProgram(PR, 1, &D, build_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(PR, D, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(PR, D, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        return 1;
    }

    cl_kernel Kw   = clCreateKernel(PR, "weight", &err); check(err, "kernel weight");
    cl_kernel Kws31 = clCreateKernel(PR, "weight_small31_refresh", &err); check(err, "kernel weight_small31_refresh");
    cl_kernel Kwf4 = clCreateKernel(PR, "weight_forward4_first", &err); check(err, "kernel weight_forward4_first");
    cl_kernel Kwf4s31 = clCreateKernel(PR, "weight_forward4_first_small31_refresh", &err); check(err, "kernel weight_forward4_first_small31_refresh");
    cl_kernel Kf2  = clCreateKernel(PR, "forward2", &err); check(err, "kernel forward2");
    cl_kernel Kf2x = clCreateKernel(PR, "forward2_x2", &err); check(err, "kernel forward2_x2");
    cl_kernel Kf2q = clCreateKernel(PR, "forward2_x4", &err); check(err, "kernel forward2_x4");
    cl_kernel Kb2  = clCreateKernel(PR, "backward2", &err); check(err, "kernel backward2");
    cl_kernel Kb2x = clCreateKernel(PR, "backward2_x2", &err); check(err, "kernel backward2_x2");
    cl_kernel Kb2q = clCreateKernel(PR, "backward2_x4", &err); check(err, "kernel backward2_x4");
    cl_kernel Kfz2x = clCreateKernel(PR, "fused_center2_x2", &err); check(err, "kernel fused_center2_x2");
    cl_kernel Kfz2q = clCreateKernel(PR, "fused_center2_x4", &err); check(err, "kernel fused_center2_x4");
    cl_kernel Kf4  = clCreateKernel(PR, "forward4", &err); check(err, "kernel forward4");
    cl_kernel Kf4x = clCreateKernel(PR, "forward4_x2", &err); check(err, "kernel forward4_x2");
    cl_kernel Kf4q = clCreateKernel(PR, "forward4_x4", &err); check(err, "kernel forward4_x4");
    cl_kernel Kf4l = clCreateKernel(PR, "forward4_local", &err); check(err, "kernel forward4_local");
    cl_kernel Kf4l2 = clCreateKernel(PR, "forward4_local2", &err); check(err, "kernel forward4_local2");
    cl_kernel Kf64 = clCreateKernel(PR, "forward64_stage", &err); check(err, "kernel forward64_stage");
    cl_kernel Kfp2 = clCreateKernel(PR, "forward_pair_large2", &err); check(err, "kernel forward_pair_large2");
    cl_kernel Kf640 = clCreateKernel(PR, "forward64_0", &err); check(err, "kernel forward64_0");
    cl_kernel Kf640s31 = clCreateKernel(PR, "forward64_0_small31_refresh", &err); check(err, "kernel forward64_0_small31_refresh");
    cl_kernel Kf640s31c = clCreateKernel(PR, "forward64_0_small31_refresh_compact", &err); check(err, "kernel forward64_0_small31_refresh_compact");
    cl_kernel Kf256 = clCreateKernel(PR, "forward256_stage", &err); check(err, "kernel forward256_stage");
    cl_kernel Kf25664 = clCreateKernel(PR, "forward256_stage_m64", &err); check(err, "kernel forward256_stage_m64");
    cl_kernel Kf1024 = clCreateKernel(PR, "forward1024_stage", &err); check(err, "kernel forward1024_stage");
    cl_kernel Ksq  = clCreateKernel(PR, "square_half", &err); check(err, "kernel square_half");
    cl_kernel Ksqx = clCreateKernel(PR, "square_half_x2", &err); check(err, "kernel square_half_x2");
    cl_kernel Ksqq = clCreateKernel(PR, "square_half_x4", &err); check(err, "kernel square_half_x4");
    cl_kernel Kb4  = clCreateKernel(PR, "backward4", &err); check(err, "kernel backward4");
    cl_kernel Kb4x = clCreateKernel(PR, "backward4_x2", &err); check(err, "kernel backward4_x2");
    cl_kernel Kb4q = clCreateKernel(PR, "backward4_x4", &err); check(err, "kernel backward4_x4");
    cl_kernel Kb4l = clCreateKernel(PR, "backward4_local", &err); check(err, "kernel backward4_local");
    cl_kernel Kb4l2 = clCreateKernel(PR, "backward4_local2", &err); check(err, "kernel backward4_local2");
    cl_kernel Kb64 = clCreateKernel(PR, "backward64_stage", &err); check(err, "kernel backward64_stage");
    cl_kernel Kb640 = clCreateKernel(PR, "backward64_0", &err); check(err, "kernel backward64_0");
    cl_kernel Kb640d = clCreateKernel(PR, "backward64_0_small31_defer", &err); check(err, "kernel backward64_0_small31_defer");
    cl_kernel Kb640dc = clCreateKernel(PR, "backward64_0_small31_defer_compact", &err); check(err, "kernel backward64_0_small31_defer_compact");
    cl_kernel Kb256 = clCreateKernel(PR, "backward256_stage", &err); check(err, "kernel backward256_stage");
    cl_kernel Kb1024 = clCreateKernel(PR, "backward1024_stage", &err); check(err, "kernel backward1024_stage");
    cl_kernel Kb4m64 = clCreateKernel(PR, "backward4_stage_m64", &err); check(err, "kernel backward4_stage_m64");
    cl_kernel Kb4m16 = clCreateKernel(PR, "backward4_stage_m16", &err); check(err, "kernel backward4_stage_m16");
    cl_kernel Kbul = clCreateKernel(PR, "backward4_last_unweight", &err); check(err, "kernel backward4_last_unweight");
    cl_kernel Ku   = clCreateKernel(PR, "unweight_norm", &err); check(err, "kernel unweight_norm");
    cl_kernel Kbp   = clCreateKernel(PR, "block_prepare_kernel", &err); check(err, "kernel block_prepare_kernel");
    cl_kernel Kbpd  = clCreateKernel(PR, "block_prepare_direct_kernel", &err); check(err, "kernel block_prepare_direct_kernel");
    cl_kernel Kbs   = clCreateKernel(PR, "block_scan_kernel", &err); check(err, "kernel block_scan_kernel");
    cl_kernel Kbf   = clCreateKernel(PR, "block_finalize_kernel", &err); check(err, "kernel block_finalize_kernel");
    cl_kernel Kbac  = clCreateKernel(PR, "block_apply_prev_carry_kernel", &err); check(err, "kernel block_apply_prev_carry_kernel");
    cl_kernel Kbacd = clCreateKernel(PR, "block_apply_carry_direct_kernel", &err); check(err, "kernel block_apply_carry_direct_kernel");
    cl_kernel Kbpd8 = clCreateKernel(PR, "block_prepare_direct8_mask_kernel", &err); check(err, "kernel block_prepare_direct8_mask_kernel");
    cl_kernel Kbacd8 = clCreateKernel(PR, "block_apply_carry_direct8_mask_kernel", &err); check(err, "kernel block_apply_carry_direct8_mask_kernel");
    cl_kernel Kbpd8f = clCreateKernel(PR, "block_prepare_direct8_mask_fused64_kernel", &err); check(err, "kernel block_prepare_direct8_mask_fused64_kernel");
    cl_kernel Kbagh8 = clCreateKernel(PR, "block_apply_group_head_direct8_mask_kernel", &err); check(err, "kernel block_apply_group_head_direct8_mask_kernel");
    cl_kernel Kbpd8fs = clCreateKernel(PR, "block_prepare_direct8_mask_fused64_small31_kernel", &err); check(err, "kernel block_prepare_direct8_mask_fused64_small31_kernel");
    cl_kernel Kbagh8s = clCreateKernel(PR, "block_apply_group_head_direct8_mask_small31_kernel", &err); check(err, "kernel block_apply_group_head_direct8_mask_small31_kernel");
    cl_kernel Kbpd8fsc = clCreateKernel(PR, "block_prepare_direct8_mask_fused64_small31_compact_kernel", &err); check(err, "kernel block_prepare_direct8_mask_fused64_small31_compact_kernel");
    cl_kernel Kbagh8sc = clCreateKernel(PR, "block_apply_group_head_direct8_mask_small31_compact_kernel", &err); check(err, "kernel block_apply_group_head_direct8_mask_small31_compact_kernel");
    cl_kernel Kwrap = clCreateKernel(PR, "carry_wrap_kernel", &err); check(err, "kernel carry_wrap_kernel");
    cl_kernel Kbci  = clCreateKernel(PR, "init_block_carry_kernel", &err); check(err, "kernel init_block_carry_kernel");
    cl_kernel Kbcp  = clCreateKernel(PR, "block_carry_phase_kernel", &err); check(err, "kernel block_carry_phase_kernel");
    cl_kernel Kbcp0 = clCreateKernel(PR, "block_carry_phase_kernel", &err); check(err, "kernel block_carry_phase_kernel phase0");
    cl_kernel Kbcp1 = clCreateKernel(PR, "block_carry_phase_kernel", &err); check(err, "kernel block_carry_phase_kernel phase1");
    cl_kernel Kbcd  = clCreateKernel(PR, "block_carry_drain_kernel", &err); check(err, "kernel block_carry_drain_kernel");
    cl_kernel Ksub  = clCreateKernel(PR, "sub_kernel", &err); check(err, "kernel sub_kernel");

    std::vector<GF61_31> z(h);
    std::vector<GF61_31> wv(n);
    std::vector<uint64_t> wraw(3 * n);
    std::vector<IBWeight> w_ib(n);
    std::vector<uint8_t> digit_width(n);

    z[0] = GF61_31((mode == TestMode::LL) ? 4u : 3u);
    for (size_t i = 1; i < h; ++i) z[i] = GF61_31(0u);

    for (size_t s = 1; s <= n / 4; s <<= 1) {
        GF61_31 r_s = GF61_31::root_nth(2 * s);
        for (size_t j = 0; j < s; ++j) wv[s + j] = r_s.pow(bitrev(j, s));
    }
    for (size_t s = 1; s <= n / 4; s <<= 1) {
        for (size_t j = 0; j < s; ++j) wv[n / 2 + s + j] = wv[s + j].mul(wv[2 * (s + j)]);
    }

    for (size_t i = 0; i < n; ++i) pack_gf_words(wraw, i, wv[i]);

    const uint8_t q_n = uint8_t(p >> ln);
    const uint8_t lr2_61 = GF61::log2_root_two(n);
    const uint8_t lr2_31 = GF31::log2_root_two(n);
    uint32_t o = 0;
    for (size_t j = 0; j <= n; ++j) {
        uint64_t qj = uint64_t(p) * uint64_t(j);
        uint32_t ceil_qj_n = uint32_t(((qj - 1) >> ln) + 1);
        if (j > 0) {
            uint8_t c = uint8_t(ceil_qj_n - o);
            if ((c != q_n) && (c != q_n + 1)) {
                std::cerr << "digit width generation error\n";
                return 1;
            }
            digit_width[j - 1] = c;
            if (j < n) {
                const uint32_t r = uint32_t(qj & (n - 1));
                const uint8_t w61 = uint8_t((lr2_61 * (n - r)) % 61);
                const uint8_t w31 = uint8_t((lr2_31 * (n - r)) % 31);
                w_ib[j] = { w61, w31 };
            }
        }
        o = ceil_qj_n;
    }
    w_ib[0] = { 0, 0 };

    struct BlockInfoHost { uint32_t start, count, bits, pad; };
    struct BlockStateHost { uint64_t low_lo, low_hi, q_lo, q_hi; uint32_t prefix_all_ones, pad; };
    struct CarryWordHost { uint64_t lo, hi; };

    const uint32_t h_u32 = static_cast<uint32_t>(h);
    std::vector<BlockInfoHost> blocks;
    uint32_t target_pairs_per_block = auto_tune.carry_pairs;
    if (carry_pairs_override != 0u) target_pairs_per_block = std::max<uint32_t>(1u, carry_pairs_override);
    target_pairs_per_block = clamp_carry_pairs(target_pairs_per_block, h_u32);
    blocks.reserve((h_u32 + target_pairs_per_block - 1u) / target_pairs_per_block);
    for (uint32_t start = 0; start < h_u32;) {
        uint32_t bits = 0;
        uint32_t count = 0;
        while (start + count < h_u32 && count < target_pairs_per_block) {
            uint32_t pair_bits = uint32_t(digit_width[2ull * (start + count)]) + uint32_t(digit_width[2ull * (start + count) + 1]);
            bits += pair_bits;
            ++count;
        }
        if (count == 0) { std::cerr << "block construction error\n"; return 1; }
        blocks.push_back({ start, count, bits, 0u });
        start += count;
    }
    const uint32_t nblocks_u32 = static_cast<uint32_t>(blocks.size());
    bool can_use_direct_block_carry = (nblocks_u32 > 1u);
    for (const auto& bi : blocks) {
        if (bi.bits <= 128u) { can_use_direct_block_carry = false; break; }
    }
    const bool can_use_direct8_mask = can_use_direct_block_carry && target_pairs_per_block == 8u && (h_u32 % 8u) == 0u;
    const bool can_use_group_fused_direct8 = can_use_direct8_mask && carry_wg == 64u && (nblocks_u32 % 64u) == 0u;
    const uint32_t ngroups_direct8 = can_use_group_fused_direct8 ? (nblocks_u32 / 64u) : 0u;
    std::vector<uint16_t> block_wide_mask;
    if (can_use_direct8_mask) {
        block_wide_mask.resize(nblocks_u32);
        for (uint32_t b = 0; b < nblocks_u32; ++b) {
            uint16_t m = 0u;
            const uint32_t start = b * 8u;
            for (uint32_t t = 0; t < 8u; ++t) {
                if (digit_width[2ull * (start + t)] != q_n) m |= (1u << (2u * t));
                if (digit_width[2ull * (start + t) + 1ull] != q_n) m |= (1u << (2u * t + 1u));
            }
            block_wide_mask[b] = m;
        }
    }
    // Conservative safety gate: the direct block-carry fast path is only enabled on
    // very large transforms where it has been performance-tested. Smaller cases use
    // the generic scan/drain path to preserve correctness.
    if (h_u32 < (1u << 16)) can_use_direct_block_carry = false;
    std::cout << "carry_blocks=" << nblocks_u32 << ", avg_pairs_per_block=" << std::fixed << std::setprecision(2) << (double(h_u32) / double(nblocks_u32)) << ", carry_target_pairs=" << target_pairs_per_block << ", tuned_wg=" << wg << ", carry_wg=" << carry_wg << ", local_stage_cap=" << ((local_stage_cap_override != 0u) ? std::min<size_t>(max_wg, static_cast<size_t>(local_stage_cap_override)) : std::min<size_t>(max_wg, (is_gfx9 || is_nvidia_dev) ? 256u : 128u)) << ", auto_profile=" << auto_tune.profile;
    if (wg_override != 0u || carry_pairs_override != 0u) std::cout << " (manual override)";
    std::cout << "\n";

    const bool use_small31_compact_cycle = can_use_group_fused_direct8 && ((uint32_t(q_n) + 1u) < 31u) && (mode == TestMode::PRP);
    std::vector<uint64_t> zc_init;
    if (use_small31_compact_cycle) {
        zc_init.resize(2u * h);
        for (size_t i = 0; i < h; ++i) {
            zc_init[2u * i + 0u] = z[i].g61.s0();
            zc_init[2u * i + 1u] = z[i].g61.s1();
        }
    }

    cl_mem Bz  = clCreateBuffer(C, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(z[0]) * h, z.data(), &err); check(err, "buffer z");
    cl_mem Bzc = nullptr;
    if (use_small31_compact_cycle) {
        Bzc = clCreateBuffer(C, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t) * zc_init.size(), zc_init.data(), &err); check(err, "buffer z_compact");
    }
    cl_mem Bw  = clCreateBuffer(C, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(wraw[0]) * wraw.size(), wraw.data(), &err); check(err, "buffer w");
    cl_mem Bwi = clCreateBuffer(C, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(w_ib[0]) * w_ib.size(), w_ib.data(), &err); check(err, "buffer w_ib");
    cl_mem Bdw = clCreateBuffer(C, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(digit_width[0]) * digit_width.size(), digit_width.data(), &err); check(err, "buffer digit_width");
    cl_mem Bwm = nullptr;
    if (can_use_direct8_mask) {
        Bwm = clCreateBuffer(C, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(block_wide_mask[0]) * block_wide_mask.size(), block_wide_mask.data(), &err); check(err, "buffer block_wide_mask");
    }
    std::vector<BlockStateHost> block_state_init(nblocks_u32);
    std::vector<CarryWordHost> carry_in_init(nblocks_u32);
    std::vector<CarryWordHost> final_carry_init(1);
    cl_mem Bblk   = clCreateBuffer(C, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(BlockInfoHost) * nblocks_u32, blocks.data(), &err); check(err, "buffer blocks");
    cl_mem Bstate = clCreateBuffer(C, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(BlockStateHost) * nblocks_u32, block_state_init.data(), &err); check(err, "buffer block_state");
    cl_mem Bcin   = clCreateBuffer(C, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CarryWordHost) * nblocks_u32, carry_in_init.data(), &err); check(err, "buffer carry_in");
    cl_mem Bfinal = clCreateBuffer(C, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CarryWordHost), final_carry_init.data(), &err); check(err, "buffer final_carry");

    const int n_i = int(n);
    const int ln_i = int(ln);
    const uint32_t sub_val_ll = 2u;

    const size_t gf_local_bytes = 24u;
    size_t local_stage_cap = std::min<size_t>(max_wg, (is_gfx9 || is_nvidia_dev) ? 256u : 128u);
    if (local_stage_cap_override != 0u) local_stage_cap = std::min<size_t>(max_wg, static_cast<size_t>(local_stage_cap_override));
    auto stage_can_use_local = [&](size_t m_stage) -> bool {
        if (m_stage == 0 || m_stage > local_stage_cap) return false;
        const size_t needed = 4u * m_stage * gf_local_bytes;
        return needed <= static_cast<size_t>(local_mem_size);
    };
    auto stage_can_use_local2_fwd = [&](size_t m_stage) -> bool {
        if (m_stage < 4 || (m_stage & 3u) != 0u) return false;
        return stage_can_use_local(m_stage);
    };
    auto stage_can_use_local64_fwd = [&](size_t m_stage) -> bool {
        if (m_stage < 16 || (m_stage & 15u) != 0u) return false;
        return stage_can_use_local(m_stage);
    };
    auto stage_can_use_local256_fwd = [&](size_t m_stage) -> bool {
        if (m_stage < 64 || (m_stage & 63u) != 0u) return false;
        if (m_stage == 0 || m_stage > local_stage_cap) return false;
        const size_t needed = 4u * m_stage * gf_local_bytes + 255u * gf_local_bytes;
        return needed <= static_cast<size_t>(local_mem_size);
    };
    auto stage_can_use_local1024_fwd = [&](size_t m_stage) -> bool {
        (void)m_stage;
        return false;
    };
    auto stage_can_use_local2_bwd = [&](size_t m_stage) -> bool {
        const size_t ls = 4u * m_stage;
        if (m_stage == 0 || ls > local_stage_cap) return false;
        const size_t needed = 16u * m_stage * gf_local_bytes;
        return needed <= static_cast<size_t>(local_mem_size);
    };
    auto stage_can_use_local64_bwd = [&](size_t m_stage) -> bool {
        const size_t ls = 16u * m_stage;
        if (m_stage == 0 || ls > local_stage_cap) return false;
        const size_t needed = 64u * m_stage * gf_local_bytes;
        return needed <= static_cast<size_t>(local_mem_size);
    };
    auto stage_can_use_local256_bwd = [&](size_t m_stage) -> bool {
        const size_t ls = 64u * m_stage;
        if (m_stage == 0 || ls > local_stage_cap) return false;
        const size_t needed = 256u * m_stage * gf_local_bytes + 255u * gf_local_bytes;
        return needed <= static_cast<size_t>(local_mem_size);
    };
    auto stage_can_use_local1024_bwd = [&](size_t m_stage) -> bool {
        (void)m_stage;
        return false;
    };

    auto can_use_forward_pair_large2 = [&](size_t s_stage, size_t m_stage) -> bool {
        (void)s_stage; (void)m_stage;
        return false;
    };

    auto use_x4_path = [&](size_t active, size_t m_stage) -> bool {
        const size_t min_active = is_gfx9 ? (8u * wg) : (4u * wg);
        return (active >= min_active) && (m_stage >= 4) && (m_stage < 512);
    };
    auto use_x2_path = [&](size_t active, size_t m_stage) -> bool {
        const size_t min_active = is_gfx9 ? (4u * wg) : (2u * wg);
        return (active >= min_active) && (m_stage >= 2) && (m_stage < 128);
    };

    auto can_use_chunk64_0 = [&]() -> bool {
        return false;
    };

    check(clSetKernelArg(Kw, 0, sizeof(cl_mem), &Bz), "set arg weight z");
    check(clSetKernelArg(Kw, 1, sizeof(cl_mem), &Bwi), "set arg weight w");

    check(clSetKernelArg(Ksq, 0, sizeof(cl_mem), &Bz), "set arg square_half z");
    check(clSetKernelArg(Ksq, 1, sizeof(cl_mem), &Bw), "set arg square_half w");
    check(clSetKernelArg(Ksq, 2, sizeof(int), &n_i), "set arg square_half n");
    check(clSetKernelArg(Ksqx, 0, sizeof(cl_mem), &Bz), "set arg square_half_x2 z");
    check(clSetKernelArg(Ksqx, 1, sizeof(cl_mem), &Bw), "set arg square_half_x2 w");
    check(clSetKernelArg(Ksqx, 2, sizeof(int), &n_i), "set arg square_half_x2 n");
    check(clSetKernelArg(Ksqq, 0, sizeof(cl_mem), &Bz), "set arg square_half_x4 z");
    check(clSetKernelArg(Ksqq, 1, sizeof(cl_mem), &Bw), "set arg square_half_x4 w");
    check(clSetKernelArg(Ksqq, 2, sizeof(int), &n_i), "set arg square_half_x4 n");

    check(clSetKernelArg(Ku, 0, sizeof(cl_mem), &Bz), "set arg unweight z");
    check(clSetKernelArg(Ku, 1, sizeof(cl_mem), &Bwi), "set arg unweight w");
    check(clSetKernelArg(Ku, 2, sizeof(int), &ln_i), "set arg unweight ln");

    check(clSetKernelArg(Kbp, 0, sizeof(cl_mem), &Bz), "set arg block_prepare z");
    check(clSetKernelArg(Kbp, 1, sizeof(cl_mem), &Bdw), "set arg block_prepare dw");
    check(clSetKernelArg(Kbp, 2, sizeof(cl_mem), &Bblk), "set arg block_prepare blocks");
    check(clSetKernelArg(Kbp, 3, sizeof(cl_mem), &Bstate), "set arg block_prepare state");
    check(clSetKernelArg(Kbp, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_prepare nblocks");

    check(clSetKernelArg(Kbpd, 0, sizeof(cl_mem), &Bz), "set arg block_prepare_direct z");
    check(clSetKernelArg(Kbpd, 1, sizeof(cl_mem), &Bdw), "set arg block_prepare_direct dw");
    check(clSetKernelArg(Kbpd, 2, sizeof(cl_mem), &Bblk), "set arg block_prepare_direct blocks");
    const uint32_t direct_narrow_w = uint32_t(q_n);
    const uint32_t direct_small31 = (uint32_t(q_n) + 1u < 31u) ? 1u : 0u;
    const bool can_use_group_fused_direct8_small31 = can_use_group_fused_direct8 && (direct_small31 != 0u);
    check(clSetKernelArg(Kbpd, 3, sizeof(cl_mem), &Bcin), "set arg block_prepare_direct carry_next");
    check(clSetKernelArg(Kbpd, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_prepare_direct nblocks");
    check(clSetKernelArg(Kbpd, 5, sizeof(uint32_t), &direct_narrow_w), "set arg block_prepare_direct narrow_w");
    check(clSetKernelArg(Kbpd, 6, sizeof(uint32_t), &direct_small31), "set arg block_prepare_direct small31");
    if (can_use_direct8_mask) {
        check(clSetKernelArg(Kbpd8, 0, sizeof(cl_mem), &Bz), "set arg block_prepare_direct8 z");
        check(clSetKernelArg(Kbpd8, 1, sizeof(cl_mem), &Bwm), "set arg block_prepare_direct8 mask");
        check(clSetKernelArg(Kbpd8, 2, sizeof(cl_mem), &Bcin), "set arg block_prepare_direct8 carry_next");
        check(clSetKernelArg(Kbpd8, 3, sizeof(uint32_t), &nblocks_u32), "set arg block_prepare_direct8 nblocks");
        check(clSetKernelArg(Kbpd8, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_prepare_direct8 narrow_w");
        check(clSetKernelArg(Kbpd8, 5, sizeof(uint32_t), &direct_small31), "set arg block_prepare_direct8 small31");
    }
    if (can_use_group_fused_direct8) {
        check(clSetKernelArg(Kbpd8f, 0, sizeof(cl_mem), &Bz), "set arg block_prepare_direct8_fused z");
        check(clSetKernelArg(Kbpd8f, 1, sizeof(cl_mem), &Bwm), "set arg block_prepare_direct8_fused mask");
        check(clSetKernelArg(Kbpd8f, 2, sizeof(cl_mem), &Bcin), "set arg block_prepare_direct8_fused group_tail");
        check(clSetKernelArg(Kbpd8f, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_prepare_direct8_fused ngroups");
        check(clSetKernelArg(Kbpd8f, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_prepare_direct8_fused narrow_w");
        check(clSetKernelArg(Kbpd8f, 5, sizeof(uint32_t), &direct_small31), "set arg block_prepare_direct8_fused small31");

        check(clSetKernelArg(Kbagh8, 0, sizeof(cl_mem), &Bz), "set arg block_apply_group_head_direct8 z");
        check(clSetKernelArg(Kbagh8, 1, sizeof(cl_mem), &Bwm), "set arg block_apply_group_head_direct8 mask");
        check(clSetKernelArg(Kbagh8, 2, sizeof(cl_mem), &Bcin), "set arg block_apply_group_head_direct8 group_tail");
        check(clSetKernelArg(Kbagh8, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_apply_group_head_direct8 ngroups");
        check(clSetKernelArg(Kbagh8, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_apply_group_head_direct8 narrow_w");
        check(clSetKernelArg(Kbagh8, 5, sizeof(uint32_t), &direct_small31), "set arg block_apply_group_head_direct8 small31");
    }
    if (can_use_group_fused_direct8_small31) {
        check(clSetKernelArg(Kbpd8fs, 0, sizeof(cl_mem), &Bz), "set arg block_prepare_direct8_fused_small31 z");
        check(clSetKernelArg(Kbpd8fs, 1, sizeof(cl_mem), &Bwm), "set arg block_prepare_direct8_fused_small31 mask");
        check(clSetKernelArg(Kbpd8fs, 2, sizeof(cl_mem), &Bcin), "set arg block_prepare_direct8_fused_small31 group_tail");
        check(clSetKernelArg(Kbpd8fs, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_prepare_direct8_fused_small31 ngroups");
        check(clSetKernelArg(Kbpd8fs, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_prepare_direct8_fused_small31 narrow_w");

        check(clSetKernelArg(Kbagh8s, 0, sizeof(cl_mem), &Bz), "set arg block_apply_group_head_direct8_small31 z");
        check(clSetKernelArg(Kbagh8s, 1, sizeof(cl_mem), &Bwm), "set arg block_apply_group_head_direct8_small31 mask");
        check(clSetKernelArg(Kbagh8s, 2, sizeof(cl_mem), &Bcin), "set arg block_apply_group_head_direct8_small31 group_tail");
        check(clSetKernelArg(Kbagh8s, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_apply_group_head_direct8_small31 ngroups");
        check(clSetKernelArg(Kbagh8s, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_apply_group_head_direct8_small31 narrow_w");
        if (Bzc) {
            check(clSetKernelArg(Kbpd8fsc, 0, sizeof(cl_mem), &Bzc), "set arg block_prepare_direct8_fused_small31_compact z");
            check(clSetKernelArg(Kbpd8fsc, 1, sizeof(cl_mem), &Bwm), "set arg block_prepare_direct8_fused_small31_compact mask");
            check(clSetKernelArg(Kbpd8fsc, 2, sizeof(cl_mem), &Bcin), "set arg block_prepare_direct8_fused_small31_compact group_tail");
            check(clSetKernelArg(Kbpd8fsc, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_prepare_direct8_fused_small31_compact ngroups");
            check(clSetKernelArg(Kbpd8fsc, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_prepare_direct8_fused_small31_compact narrow_w");
            check(clSetKernelArg(Kbagh8sc, 0, sizeof(cl_mem), &Bzc), "set arg block_apply_group_head_direct8_small31_compact z");
            check(clSetKernelArg(Kbagh8sc, 1, sizeof(cl_mem), &Bwm), "set arg block_apply_group_head_direct8_small31_compact mask");
            check(clSetKernelArg(Kbagh8sc, 2, sizeof(cl_mem), &Bcin), "set arg block_apply_group_head_direct8_small31_compact group_tail");
            check(clSetKernelArg(Kbagh8sc, 3, sizeof(uint32_t), &ngroups_direct8), "set arg block_apply_group_head_direct8_small31_compact ngroups");
            check(clSetKernelArg(Kbagh8sc, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_apply_group_head_direct8_small31_compact narrow_w");
        }
    }

    check(clSetKernelArg(Kbs, 0, sizeof(cl_mem), &Bblk), "set arg block_scan blocks");
    check(clSetKernelArg(Kbs, 1, sizeof(cl_mem), &Bstate), "set arg block_scan state");
    check(clSetKernelArg(Kbs, 2, sizeof(cl_mem), &Bcin), "set arg block_scan carry_in");
    check(clSetKernelArg(Kbs, 3, sizeof(cl_mem), &Bfinal), "set arg block_scan final_carry");
    check(clSetKernelArg(Kbs, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_scan nblocks");

    check(clSetKernelArg(Kbf, 0, sizeof(cl_mem), &Bz), "set arg block_finalize z");
    check(clSetKernelArg(Kbf, 1, sizeof(cl_mem), &Bblk), "set arg block_finalize blocks");
    check(clSetKernelArg(Kbf, 2, sizeof(cl_mem), &Bcin), "set arg block_finalize carry_in");
    check(clSetKernelArg(Kbf, 3, sizeof(cl_mem), &Bdw), "set arg block_finalize dw");
    check(clSetKernelArg(Kbf, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_finalize nblocks");
    check(clSetKernelArg(Kbac, 0, sizeof(cl_mem), &Bz), "set arg block_apply_prev_carry z");
    check(clSetKernelArg(Kbac, 1, sizeof(cl_mem), &Bblk), "set arg block_apply_prev_carry blocks");
    check(clSetKernelArg(Kbac, 2, sizeof(cl_mem), &Bstate), "set arg block_apply_prev_carry states");
    check(clSetKernelArg(Kbac, 3, sizeof(cl_mem), &Bdw), "set arg block_apply_prev_carry dw");
    check(clSetKernelArg(Kbac, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_apply_prev_carry nblocks");
    check(clSetKernelArg(Kbacd, 0, sizeof(cl_mem), &Bz), "set arg block_apply_carry_direct z");
    check(clSetKernelArg(Kbacd, 1, sizeof(cl_mem), &Bblk), "set arg block_apply_carry_direct blocks");
    check(clSetKernelArg(Kbacd, 2, sizeof(cl_mem), &Bcin), "set arg block_apply_carry_direct carry_in");
    check(clSetKernelArg(Kbacd, 3, sizeof(cl_mem), &Bdw), "set arg block_apply_carry_direct dw");
    check(clSetKernelArg(Kbacd, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_apply_carry_direct nblocks");
    check(clSetKernelArg(Kbacd, 5, sizeof(uint32_t), &direct_narrow_w), "set arg block_apply_carry_direct narrow_w");
    check(clSetKernelArg(Kbacd, 6, sizeof(uint32_t), &direct_small31), "set arg block_apply_carry_direct small31");
    if (can_use_direct8_mask) {
        check(clSetKernelArg(Kbacd8, 0, sizeof(cl_mem), &Bz), "set arg block_apply_carry_direct8 z");
        check(clSetKernelArg(Kbacd8, 1, sizeof(cl_mem), &Bwm), "set arg block_apply_carry_direct8 mask");
        check(clSetKernelArg(Kbacd8, 2, sizeof(cl_mem), &Bcin), "set arg block_apply_carry_direct8 carry_in");
        check(clSetKernelArg(Kbacd8, 3, sizeof(uint32_t), &nblocks_u32), "set arg block_apply_carry_direct8 nblocks");
        check(clSetKernelArg(Kbacd8, 4, sizeof(uint32_t), &direct_narrow_w), "set arg block_apply_carry_direct8 narrow_w");
        check(clSetKernelArg(Kbacd8, 5, sizeof(uint32_t), &direct_small31), "set arg block_apply_carry_direct8 small31");
    }

    check(clSetKernelArg(Kwrap, 0, sizeof(cl_mem), &Bz), "set arg carry_wrap z");
    check(clSetKernelArg(Kwrap, 1, sizeof(cl_mem), &Bdw), "set arg carry_wrap dw");
    check(clSetKernelArg(Kwrap, 2, sizeof(uint32_t), &h_u32), "set arg carry_wrap n2");
    check(clSetKernelArg(Kwrap, 3, sizeof(cl_mem), &Bfinal), "set arg carry_wrap final_carry");

    check(clSetKernelArg(Kbci, 0, sizeof(cl_mem), &Bstate), "set arg init_block_carry state");
    check(clSetKernelArg(Kbci, 1, sizeof(cl_mem), &Bcin), "set arg init_block_carry carry");
    check(clSetKernelArg(Kbci, 2, sizeof(uint32_t), &nblocks_u32), "set arg init_block_carry nblocks");

    check(clSetKernelArg(Kbcp, 0, sizeof(cl_mem), &Bz), "set arg block_carry_phase z");
    check(clSetKernelArg(Kbcp, 1, sizeof(cl_mem), &Bblk), "set arg block_carry_phase blocks");
    check(clSetKernelArg(Kbcp, 2, sizeof(cl_mem), &Bcin), "set arg block_carry_phase carry");
    check(clSetKernelArg(Kbcp, 3, sizeof(cl_mem), &Bdw), "set arg block_carry_phase dw");
    check(clSetKernelArg(Kbcp, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_carry_phase nblocks");
    check(clSetKernelArg(Kbcp0, 0, sizeof(cl_mem), &Bz), "set arg block_carry_phase0 z");
    check(clSetKernelArg(Kbcp0, 1, sizeof(cl_mem), &Bblk), "set arg block_carry_phase0 blocks");
    check(clSetKernelArg(Kbcp0, 2, sizeof(cl_mem), &Bcin), "set arg block_carry_phase0 carry");
    check(clSetKernelArg(Kbcp0, 3, sizeof(cl_mem), &Bdw), "set arg block_carry_phase0 dw");
    check(clSetKernelArg(Kbcp0, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_carry_phase0 nblocks");
    { uint32_t phase0 = 0u; check(clSetKernelArg(Kbcp0, 5, sizeof(uint32_t), &phase0), "set arg block_carry_phase0 phase"); }
    check(clSetKernelArg(Kbcp1, 0, sizeof(cl_mem), &Bz), "set arg block_carry_phase1 z");
    check(clSetKernelArg(Kbcp1, 1, sizeof(cl_mem), &Bblk), "set arg block_carry_phase1 blocks");
    check(clSetKernelArg(Kbcp1, 2, sizeof(cl_mem), &Bcin), "set arg block_carry_phase1 carry");
    check(clSetKernelArg(Kbcp1, 3, sizeof(cl_mem), &Bdw), "set arg block_carry_phase1 dw");
    check(clSetKernelArg(Kbcp1, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_carry_phase1 nblocks");
    { uint32_t phase1 = 1u; check(clSetKernelArg(Kbcp1, 5, sizeof(uint32_t), &phase1), "set arg block_carry_phase1 phase"); }

    check(clSetKernelArg(Kbcd, 0, sizeof(cl_mem), &Bz), "set arg block_carry_drain z");
    check(clSetKernelArg(Kbcd, 1, sizeof(cl_mem), &Bblk), "set arg block_carry_drain blocks");
    check(clSetKernelArg(Kbcd, 2, sizeof(cl_mem), &Bcin), "set arg block_carry_drain carry");
    check(clSetKernelArg(Kbcd, 3, sizeof(cl_mem), &Bdw), "set arg block_carry_drain dw");
    check(clSetKernelArg(Kbcd, 4, sizeof(uint32_t), &nblocks_u32), "set arg block_carry_drain nblocks");

    check(clSetKernelArg(Ksub, 0, sizeof(cl_mem), &Bz), "set arg sub z");
    check(clSetKernelArg(Ksub, 1, sizeof(cl_mem), &Bdw), "set arg sub dw");
    check(clSetKernelArg(Ksub, 2, sizeof(uint32_t), &h_u32), "set arg sub n2");
    check(clSetKernelArg(Ksub, 3, sizeof(uint32_t), &sub_val_ll), "set arg sub a");


    struct PlannedLaunch {
        cl_kernel kernel;
        size_t gs;
        size_t ls;
        std::string label;
    };
    std::vector<PlannedLaunch> iter_plan;
    std::vector<PlannedLaunch> post_plan;
    std::vector<cl_kernel> owned_plan_kernels;

    auto add_existing_step = [&](std::vector<PlannedLaunch>& plan, cl_kernel kernel, size_t gs, size_t ls, const std::string& label) {
        plan.push_back(PlannedLaunch{kernel, gs, ls, label});
    };
    auto add_planned_kernel = [&](std::vector<PlannedLaunch>& plan,
                                  const char* kernel_name,
                                  size_t gs,
                                  size_t ls,
                                  auto&& set_args,
                                  const std::string& label) {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(PR, kernel_name, &kerr);
        check(kerr, (std::string("kernel ") + kernel_name + " plan").c_str());
        set_args(k);
        owned_plan_kernels.push_back(k);
        plan.push_back(PlannedLaunch{k, gs, ls, label});
    };
    auto enqueue_plan = [&](const std::vector<PlannedLaunch>& plan) {
        for (const PlannedLaunch& step : plan) {
            const size_t* ls_ptr = (step.ls != 0u) ? &step.ls : nullptr;
            check(clEnqueueNDRangeKernel(Q, step.kernel, 1, nullptr, &step.gs, ls_ptr, 0, nullptr, nullptr), step.label.c_str());
        }
    };

    const size_t gs_h_plan = round_up(h, wg);
    const size_t gs_blocks_plan = round_up(size_t(nblocks_u32), carry_wg);
    const size_t gs_groups_direct8_plan = can_use_group_fused_direct8 ? round_up(size_t(ngroups_direct8), carry_wg) : 0u;
    const size_t gs_serial_plan = 1;
    const bool use_small31_entry_refresh = can_use_group_fused_direct8_small31;

    {
        size_t current_m = n / 4, s = 1;
        bool weighted = false;
        if (current_m > 1 && can_use_chunk64_0()) {
            const size_t gs0 = h / 4;
            const size_t ls0 = 64;
            add_planned_kernel(iter_plan, use_small31_compact_cycle ? "forward64_0_small31_refresh_compact" : (use_small31_entry_refresh ? "forward64_0_small31_refresh" : "forward64_0"), gs0, ls0, [&](cl_kernel k) {
                check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward64_0 z");
                int argi = 1;
                if (use_small31_compact_cycle) check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bzc), "set arg plan forward64_0 zc");
                check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bw), "set arg plan forward64_0 w");
                check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bwi), "set arg plan forward64_0 iw");
                check(clSetKernelArg(k, argi++, sizeof(int), &n_i), "set arg plan forward64_0 n");
                check(clSetKernelArg(k, argi++, 256u * gf_local_bytes, nullptr), "set arg plan forward64_0 scratch");
            }, "enqueue plan forward64_0");
            weighted = true;
            current_m /= 64;
            s *= 64;
        } else if (current_m > 1) {
            const int m0_i = int(current_m / 2);
            const size_t active0 = size_t(m0_i);
            if (stage_can_use_local(active0)) {
                const size_t gs0 = active0;
                add_planned_kernel(iter_plan, use_small31_entry_refresh ? "weight_forward4_first_small31_refresh" : "weight_forward4_first", gs0, active0, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan weight_forward4_first z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan weight_forward4_first w");
                    check(clSetKernelArg(k, 2, sizeof(cl_mem), &Bwi), "set arg plan weight_forward4_first iw");
                    check(clSetKernelArg(k, 3, sizeof(int), &m0_i), "set arg plan weight_forward4_first m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan weight_forward4_first n");
                }, "enqueue plan weight_forward4_first");
                weighted = true;
                current_m /= 4;
                s *= 4;
            }
        }
        if (!weighted) add_existing_step(iter_plan, use_small31_entry_refresh ? Kws31 : Kw, gs_h_plan, wg, "enqueue plan weight");

        for (; current_m > 1; current_m /= 4, s *= 4) {
            int s_i = int(s), m_i = int(current_m / 2);
            const size_t active = s * size_t(m_i);
            if (current_m > 256 && can_use_forward_pair_large2(s, size_t(m_i)) && !stage_can_use_local256_fwd(size_t(m_i)) && !stage_can_use_local64_fwd(size_t(m_i))) {
                const size_t active2 = s * size_t(m_i >> 2);
                const size_t gs = round_up(active2, size_t(64));
                const size_t ls = 64;
                add_planned_kernel(iter_plan, "forward_pair_large2", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward_pair_large2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward_pair_large2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward_pair_large2 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward_pair_large2 m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward_pair_large2 n");
                }, "enqueue plan forward_pair_large2");
                current_m /= 16;
                s *= 16;
                continue;
            } else if (current_m > 256 && stage_can_use_local1024_fwd(size_t(m_i))) {
                if (m_i == 256 && max_wg >= 256 && local_mem_size >= (cl_ulong)(1024u * gf_local_bytes)) {
                    const size_t gs256 = round_up(s, size_t(1)) * 256u;
                    const size_t ls256 = 256u;
                    add_planned_kernel(iter_plan, "forward1024_stage_m256", gs256, ls256, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward1024_stage_m256 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward1024_stage_m256 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward1024_stage_m256 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan forward1024_stage_m256 n");
                        check(clSetKernelArg(k, 4, 1024u * gf_local_bytes, nullptr), "set arg plan forward1024_stage_m256 scratch");
                    }, "enqueue plan forward1024_stage_m256");
                } else {
                    const size_t gs = active;
                    const size_t ls = size_t(m_i);
                    const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                    add_planned_kernel(iter_plan, "forward1024_stage", gs, ls, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward1024_stage z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward1024_stage w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward1024_stage s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward1024_stage m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward1024_stage n");
                        check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan forward1024_stage scratch");
                    }, "enqueue plan forward1024_stage");
                }
                current_m /= 256;
                s *= 256;
                continue;
            } else if (current_m > 64 && stage_can_use_local256_fwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = size_t(m_i);
                if (m_i == 64 && max_wg >= 64 && local_mem_size >= (cl_ulong)(256u * gf_local_bytes)) {
                    const size_t gs64 = round_up(s, size_t(1)) * 64u;
                    const size_t ls64 = 64u;
                    add_planned_kernel(iter_plan, "forward256_stage_m64", gs64, ls64, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward256_stage_m64 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward256_stage_m64 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward256_stage_m64 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan forward256_stage_m64 n");
                        check(clSetKernelArg(k, 4, 256u * gf_local_bytes, nullptr), "set arg plan forward256_stage_m64 scratch");
                    }, "enqueue plan forward256_stage_m64");
                } else {
                    const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                    add_planned_kernel(iter_plan, "forward256_stage", gs, ls, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward256_stage z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward256_stage w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward256_stage s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward256_stage m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward256_stage n");
                        check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan forward256_stage scratch");
                    }, "enqueue plan forward256_stage");
                }
                current_m /= 64;
                s *= 64;
                continue;
            } else if (current_m > 16 && stage_can_use_local64_fwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = size_t(m_i);
                const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "forward64_stage", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward64_stage z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward64_stage w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward64_stage s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward64_stage m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward64_stage n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan forward64_stage scratch");
                }, "enqueue plan forward64_stage");
                current_m /= 16;
                s *= 16;
                continue;
            } else if (current_m > 4 && stage_can_use_local2_fwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = size_t(m_i);
                const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "forward4_local2", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward4_local2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward4_local2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward4_local2 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward4_local2 m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward4_local2 n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan forward4_local2 scratch");
                }, "enqueue plan forward4_local2");
                current_m /= 4;
                s *= 4;
                continue;
            } else if (stage_can_use_local(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = size_t(m_i);
                const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "forward4_local", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward4_local z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward4_local w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward4_local s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward4_local m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward4_local n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan forward4_local scratch");
                }, "enqueue plan forward4_local");
            } else {
                if (use_x4_path(active, size_t(m_i))) {
                    const size_t active4 = (active + 3) / 4;
                    const size_t gs = round_up(active4, wg);
                    add_planned_kernel(iter_plan, "forward4_x4", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward4_x4 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward4_x4 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward4_x4 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward4_x4 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward4_x4 n");
                    }, "enqueue plan forward4_x4");
                } else if (use_x2_path(active, size_t(m_i))) {
                    const size_t active2 = (active + 1) / 2;
                    const size_t gs = round_up(active2, wg);
                    add_planned_kernel(iter_plan, "forward4_x2", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward4_x2 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward4_x2 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward4_x2 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward4_x2 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward4_x2 n");
                    }, "enqueue plan forward4_x2");
                } else {
                    const size_t gs = round_up(active, wg);
                    add_planned_kernel(iter_plan, "forward4", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward4 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward4 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan forward4 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan forward4 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan forward4 n");
                    }, "enqueue plan forward4");
                }
            }
        }

        if (current_m == 1) {
            const int n4_i = int(h / 2);
            const size_t active = h / 2;
            if (use_x4_path(active, active)) {
                const size_t active4 = (active + 3) / 4;
                const size_t gs = round_up(active4, wg);
                add_planned_kernel(iter_plan, "fused_center2_x4", gs, wg, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan fused_center2_x4 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan fused_center2_x4 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &n4_i), "set arg plan fused_center2_x4 n4");
                }, "enqueue plan fused_center2_x4");
            } else if (use_x2_path(active, active)) {
                const size_t active2 = (active + 1) / 2;
                const size_t gs = round_up(active2, wg);
                add_planned_kernel(iter_plan, "fused_center2_x2", gs, wg, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan fused_center2_x2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan fused_center2_x2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &n4_i), "set arg plan fused_center2_x2 n4");
                }, "enqueue plan fused_center2_x2");
            } else {
                const size_t gs = round_up(active, wg);
                add_planned_kernel(iter_plan, "forward2", gs, wg, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan forward2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan forward2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &n4_i), "set arg plan forward2 n4");
                }, "enqueue plan forward2");
                add_existing_step(iter_plan, Ksq, gs, wg, "enqueue plan square_half");
                add_planned_kernel(iter_plan, "backward2", gs, wg, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &n4_i), "set arg plan backward2 n4");
                }, "enqueue plan backward2");
            }
        } else {
            const size_t active = h / 2;
            if (use_x4_path(active, active)) {
                const size_t active4 = (active + 3) / 4;
                const size_t gs = round_up(active4, wg);
                add_existing_step(iter_plan, Ksqq, gs, wg, "enqueue plan square_half_x4");
            } else if (use_x2_path(active, active)) {
                const size_t active2 = (active + 1) / 2;
                const size_t gs = round_up(active2, wg);
                add_existing_step(iter_plan, Ksqx, gs, wg, "enqueue plan square_half_x2");
            } else {
                const size_t gs = round_up(active, wg);
                add_existing_step(iter_plan, Ksq, gs, wg, "enqueue plan square_half");
            }
        }

        bool unweighted = false;
        for (size_t back_m = (current_m == 1) ? 4 : 2, back_s = s / 4; back_s >= 1; back_m *= 4, back_s /= 4) {
            int s_i = int(back_s), m_i = int(back_m / 2);
            const size_t active = back_s * size_t(m_i);
            if (back_s == 64 && can_use_chunk64_0()) {
                const size_t gs = h / 4;
                const size_t ls = 64;
                add_planned_kernel(iter_plan, use_small31_compact_cycle ? "backward64_0_small31_defer_compact" : (use_small31_entry_refresh ? "backward64_0_small31_defer" : "backward64_0"), gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward64_0 z");
                    int argi = 1;
                    if (use_small31_compact_cycle) check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bzc), "set arg plan backward64_0 zc");
                    check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bw), "set arg plan backward64_0 w");
                    check(clSetKernelArg(k, argi++, sizeof(cl_mem), &Bwi), "set arg plan backward64_0 iw");
                    check(clSetKernelArg(k, argi++, sizeof(int), &n_i), "set arg plan backward64_0 n");
                    check(clSetKernelArg(k, argi++, sizeof(int), &ln_i), "set arg plan backward64_0 ln");
                    check(clSetKernelArg(k, argi++, 256u * gf_local_bytes, nullptr), "set arg plan backward64_0 scratch");
                }, "enqueue plan backward64_0");
                unweighted = true;
                break;
            }
            if (back_s == 1) {
                if (stage_can_use_local(size_t(m_i))) {
                    const size_t gs = size_t(m_i);
                    add_planned_kernel(iter_plan, "backward4_last_unweight", gs, gs, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_last_unweight z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_last_unweight w");
                        check(clSetKernelArg(k, 2, sizeof(cl_mem), &Bwi), "set arg plan backward4_last_unweight iw");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4_last_unweight m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4_last_unweight n");
                        check(clSetKernelArg(k, 5, sizeof(int), &ln_i), "set arg plan backward4_last_unweight ln");
                    }, "enqueue plan backward4_last_unweight");
                    unweighted = true;
                } else {
                    const size_t gs = round_up(active, wg);
                    add_planned_kernel(iter_plan, "backward4", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4 n");
                    }, "enqueue plan backward4");
                }
                break;
            }
            if ((is_gfx9 || is_nvidia_dev) && back_s > 4 && m_i == 64 && max_wg >= 256 && local_mem_size >= (cl_ulong)(1024u * gf_local_bytes)) {
                const size_t gs256 = round_up(back_s >> 2, size_t(1)) * 256u;
                const size_t ls256 = 256u;
                add_planned_kernel(iter_plan, "backward16_stage_m64", gs256, ls256, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward16_stage_m64 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward16_stage_m64 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward16_stage_m64 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan backward16_stage_m64 n");
                    check(clSetKernelArg(k, 4, 1024u * gf_local_bytes, nullptr), "set arg plan backward16_stage_m64 scratch");
                }, "enqueue plan backward16_stage_m64");
                back_m *= 16;
                back_s /= 16;
                continue;
            }
            if (m_i == 64 && max_wg >= 64 && local_mem_size >= (cl_ulong)(256u * gf_local_bytes)) {
                const size_t gs64 = round_up(back_s, size_t(1)) * 64u;
                const size_t ls64 = 64u;
                add_planned_kernel(iter_plan, "backward4_stage_m64", gs64, ls64, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_stage_m64 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_stage_m64 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_stage_m64 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan backward4_stage_m64 n");
                }, "enqueue plan backward4_stage_m64");
                continue;
            }
            if (m_i == 16 && max_wg >= 16) {
                const size_t gs16 = round_up(back_s, size_t(1)) * 16u;
                const size_t ls16 = 16u;
                add_planned_kernel(iter_plan, "backward4_stage_m16", gs16, ls16, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_stage_m16 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_stage_m16 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_stage_m16 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan backward4_stage_m16 n");
                }, "enqueue plan backward4_stage_m16");
                continue;
            }
            if (back_s > 256 && stage_can_use_local1024_bwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = 256u * size_t(m_i);
                const size_t scratch = 1024u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "backward1024_stage", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward1024_stage z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward1024_stage w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward1024_stage s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward1024_stage m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward1024_stage n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan backward1024_stage scratch");
                }, "enqueue plan backward1024_stage");
                back_m *= 256;
                back_s /= 256;
                continue;
            }
            if (back_s > 64 && stage_can_use_local256_bwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = 64u * size_t(m_i);
                const size_t scratch = 256u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "backward256_stage", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward256_stage z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward256_stage w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward256_stage s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward256_stage m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward256_stage n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan backward256_stage scratch");
                }, "enqueue plan backward256_stage");
                back_m *= 64;
                back_s /= 64;
                continue;
            } else if (back_s > 16 && stage_can_use_local64_bwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = 16u * size_t(m_i);
                const size_t scratch = 64u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "backward64_stage", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward64_stage z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward64_stage w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward64_stage s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward64_stage m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward64_stage n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan backward64_stage scratch");
                }, "enqueue plan backward64_stage");
                back_m *= 16;
                back_s /= 16;
                continue;
            } else if (back_s > 4 && stage_can_use_local2_bwd(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = 4u * size_t(m_i);
                const size_t scratch = 16u * size_t(m_i) * gf_local_bytes;
                add_planned_kernel(iter_plan, "backward4_local2", gs, ls, [&](cl_kernel k) {
                    check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_local2 z");
                    check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_local2 w");
                    check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_local2 s");
                    check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4_local2 m");
                    check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4_local2 n");
                    check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan backward4_local2 scratch");
                }, "enqueue plan backward4_local2");
                back_m *= 4;
                back_s /= 4;
                continue;
            } else if (stage_can_use_local(size_t(m_i))) {
                const size_t gs = active;
                const size_t ls = size_t(m_i);
                if (m_i == 256 && max_wg >= 256) {
                    const size_t gs256 = round_up(back_s, size_t(1)) * 256u;
                    const size_t ls256 = 256u;
                    add_planned_kernel(iter_plan, "backward4_stage_m256", gs256, ls256, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_stage_m256 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_stage_m256 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_stage_m256 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &n_i), "set arg plan backward4_stage_m256 n");
                    }, "enqueue plan backward4_stage_m256");
                } else {
                    const size_t scratch = 4u * size_t(m_i) * gf_local_bytes;
                    add_planned_kernel(iter_plan, "backward4_local", gs, ls, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_local z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_local w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_local s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4_local m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4_local n");
                        check(clSetKernelArg(k, 5, scratch, nullptr), "set arg plan backward4_local scratch");
                    }, "enqueue plan backward4_local");
                }
            } else {
                if (use_x4_path(active, size_t(m_i))) {
                    const size_t active4 = (active + 3) / 4;
                    const size_t gs = round_up(active4, wg);
                    add_planned_kernel(iter_plan, "backward4_x4", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_x4 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_x4 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_x4 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4_x4 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4_x4 n");
                    }, "enqueue plan backward4_x4");
                } else if (use_x2_path(active, size_t(m_i))) {
                    const size_t active2 = (active + 1) / 2;
                    const size_t gs = round_up(active2, wg);
                    add_planned_kernel(iter_plan, "backward4_x2", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4_x2 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4_x2 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4_x2 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4_x2 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4_x2 n");
                    }, "enqueue plan backward4_x2");
                } else {
                    const size_t gs = round_up(active, wg);
                    add_planned_kernel(iter_plan, "backward4", gs, wg, [&](cl_kernel k) {
                        check(clSetKernelArg(k, 0, sizeof(cl_mem), &Bz), "set arg plan backward4 z");
                        check(clSetKernelArg(k, 1, sizeof(cl_mem), &Bw), "set arg plan backward4 w");
                        check(clSetKernelArg(k, 2, sizeof(int), &s_i), "set arg plan backward4 s");
                        check(clSetKernelArg(k, 3, sizeof(int), &m_i), "set arg plan backward4 m");
                        check(clSetKernelArg(k, 4, sizeof(int), &n_i), "set arg plan backward4 n");
                    }, "enqueue plan backward4");
                }
            }
        }

        if (!unweighted) add_existing_step(iter_plan, Ku, gs_h_plan, wg, "enqueue plan unweight_norm");
    }

    if (can_use_direct_block_carry) {
        if (can_use_group_fused_direct8_small31) {
            if (use_small31_compact_cycle) {
                add_existing_step(post_plan, Kbpd8fsc, gs_groups_direct8_plan, carry_wg, "enqueue plan block_prepare_direct8_mask_fused64_small31_compact_kernel");
                add_existing_step(post_plan, Kbagh8sc, gs_groups_direct8_plan, carry_wg, "enqueue plan block_apply_group_head_direct8_mask_small31_compact_kernel");
            } else {
                add_existing_step(post_plan, Kbpd8fs, gs_groups_direct8_plan, carry_wg, "enqueue plan block_prepare_direct8_mask_fused64_small31_kernel");
                add_existing_step(post_plan, Kbagh8s, gs_groups_direct8_plan, carry_wg, "enqueue plan block_apply_group_head_direct8_mask_small31_kernel");
            }
        } else if (can_use_group_fused_direct8) {
            add_existing_step(post_plan, Kbpd8f, gs_groups_direct8_plan, carry_wg, "enqueue plan block_prepare_direct8_mask_fused64_kernel");
            add_existing_step(post_plan, Kbagh8, gs_groups_direct8_plan, carry_wg, "enqueue plan block_apply_group_head_direct8_mask_kernel");
        } else if (can_use_direct8_mask) {
            add_existing_step(post_plan, Kbpd8, gs_blocks_plan, carry_wg, "enqueue plan block_prepare_direct8_mask_kernel");
            add_existing_step(post_plan, Kbacd8, gs_blocks_plan, carry_wg, "enqueue plan block_apply_carry_direct8_mask_kernel");
        } else {
            add_existing_step(post_plan, Kbpd, gs_blocks_plan, carry_wg, "enqueue plan block_prepare_direct_kernel");
            add_existing_step(post_plan, Kbacd, gs_blocks_plan, carry_wg, "enqueue plan block_apply_carry_direct_kernel");
        }
    } else {
        add_existing_step(post_plan, Kbp, gs_blocks_plan, carry_wg, "enqueue plan block_prepare_kernel");
        add_existing_step(post_plan, Kbci, gs_blocks_plan, carry_wg, "enqueue plan init_block_carry_kernel");
        if ((nblocks_u32 & 1u) == 0u && nblocks_u32 >= 2u) {
            const uint32_t phase_rounds = is_gfx9 ? 2u : 1u;
            for (uint32_t r = 0; r < phase_rounds; ++r) {
                add_existing_step(post_plan, Kbcp0, gs_blocks_plan, carry_wg, "enqueue plan block_carry_phase even");
                add_existing_step(post_plan, Kbcp1, gs_blocks_plan, carry_wg, "enqueue plan block_carry_phase odd");
            }
        }
        add_existing_step(post_plan, Kbcd, gs_serial_plan, 0u, "enqueue plan block_carry_drain_kernel");
    }
    if (mode == TestMode::LL) add_existing_step(post_plan, Ksub, gs_serial_plan, 0u, "enqueue plan sub_kernel");

    auto t0 = std::chrono::steady_clock::now();
    auto last_report_clock = t0;
    uint32_t total_iters = (mode == TestMode::LL) ? ((p >= 2) ? (p - 2) : 0u) : p;
    if (max_iters_override != 0u) total_iters = std::min<uint32_t>(total_iters, max_iters_override);
    auto maybe_report = [&](uint32_t iter, bool force) {
        auto now = std::chrono::steady_clock::now();
        const double since_last = std::chrono::duration<double>(now - last_report_clock).count();
        const bool by_iters = (report_every != 0u && (iter == 0u || (iter % report_every) == 0u));
        const bool by_time = (report_seconds > 0.0 && since_last >= report_seconds);
        if (!force && !by_iters && !by_time) return;
        check(clFinish(Q), force ? "clFinish progress(force)" : "clFinish progress");
        now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double pct = (total_iters != 0) ? (100.0 * double(iter) / double(total_iters)) : 100.0;
        double itps = (elapsed > 0.0) ? (double(iter) / elapsed) : 0.0;
        double eta = (itps > 0.0) ? ((double(total_iters - iter)) / itps) : std::numeric_limits<double>::infinity();
        std::cout << "iter " << iter << "/" << total_iters
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                  << ", elapsed " << std::setprecision(2) << elapsed << " s"
                  << ", it/s " << std::setprecision(1) << itps;
        if (std::isfinite(eta)) std::cout << ", ETA " << std::setprecision(1) << eta << " s";
        std::cout << "\n";
        last_report_clock = now;
    };

    const uint32_t batch_iters = std::max<uint32_t>(1u, report_every != 0u ? report_every : 1000u);
    if (max_iters_override != 0u) std::cout << "benchmark_iters=" << total_iters << "\n";
    maybe_report(0u, true);
    uint32_t iter = 0u;
    while (iter < total_iters) {
        const uint32_t chunk_end = std::min<uint32_t>(total_iters, iter + batch_iters);
        for (; iter < chunk_end; ++iter) {
            enqueue_plan(iter_plan);
            enqueue_plan(post_plan);
        }
        maybe_report(chunk_end, true);
    }

    check(clFinish(Q), "clFinish final");
    std::vector<GF61_31> host_z(h);
    if (use_small31_compact_cycle) {
        std::vector<uint64_t> host_zc(2u * h);
        check(clEnqueueReadBuffer(Q, Bzc, CL_TRUE, 0, sizeof(uint64_t) * host_zc.size(), host_zc.data(), 0, nullptr, nullptr), "read z_compact");
        for (size_t i = 0; i < h; ++i) host_z[i] = GF61_31(host_zc[2u * i + 0u], host_zc[2u * i + 1u]);
    } else {
        check(clEnqueueReadBuffer(Q, Bz, CL_TRUE, 0, sizeof(GF61_31) * h, host_z.data(), 0, nullptr, nullptr), "read z");
    }

    const bool is_probable_prime = (mode == TestMode::LL)
        ? (verify_is_zero(host_z) || verify_is_Mp(host_z, digit_width))
        : verify_prp_residue_9(host_z, digit_width);

    auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double itps = (elapsed > 0.0) ? (double(total_iters) / elapsed) : 0.0;

    if (is_probable_prime) {
        std::cout << "2^" << p << " - 1 is "
                  << ((mode == TestMode::LL) ? "prime" : "a probable prime")
                  << ", elapsed " << std::fixed << std::setprecision(2) << elapsed
                  << " s, it/s " << std::setprecision(1) << itps << "\n";
    } else {
        std::cout << "2^" << p << " - 1 is composite"
                  << ", elapsed " << std::fixed << std::setprecision(2) << elapsed
                  << " s, it/s " << std::setprecision(1) << itps << "\n";
    }

    clReleaseMemObject(Bz);
    if (Bzc) clReleaseMemObject(Bzc);
    clReleaseMemObject(Bw);
    clReleaseMemObject(Bwi);
    if (Bwm) clReleaseMemObject(Bwm);
    clReleaseMemObject(Bdw);
    clReleaseMemObject(Bblk);
    clReleaseMemObject(Bstate);
    clReleaseMemObject(Bcin);
    clReleaseMemObject(Bfinal);

    for (cl_kernel k : owned_plan_kernels) clReleaseKernel(k);

    clReleaseKernel(Kw);
    clReleaseKernel(Kws31);
    clReleaseKernel(Kwf4);
    clReleaseKernel(Kwf4s31);
    clReleaseKernel(Kf2);
    clReleaseKernel(Kf2x);
    clReleaseKernel(Kf2q);
    clReleaseKernel(Kb2);
    clReleaseKernel(Kb2x);
    clReleaseKernel(Kb2q);
    clReleaseKernel(Kf4);
    clReleaseKernel(Kf4x);
    clReleaseKernel(Kf4q);
    clReleaseKernel(Kf4l);
    clReleaseKernel(Kf4l2);
    clReleaseKernel(Kf64);
    clReleaseKernel(Kfp2);
    clReleaseKernel(Kf640);
    clReleaseKernel(Kf640s31);
    clReleaseKernel(Kf640s31c);
    clReleaseKernel(Kf256);
    clReleaseKernel(Kf1024);
    clReleaseKernel(Ksq);
    clReleaseKernel(Ksqx);
    clReleaseKernel(Ksqq);
    clReleaseKernel(Kb4);
    clReleaseKernel(Kb4x);
    clReleaseKernel(Kb4q);
    clReleaseKernel(Kb4l);
    clReleaseKernel(Kb4l2);
    clReleaseKernel(Kb64);
    clReleaseKernel(Kb640);
    clReleaseKernel(Kb640d);
    clReleaseKernel(Kb640dc);
    clReleaseKernel(Kb256);
    clReleaseKernel(Kb1024);
    clReleaseKernel(Kb4m64);
    clReleaseKernel(Kb4m16);
    clReleaseKernel(Kbul);
    clReleaseKernel(Ku);
    clReleaseKernel(Kbp);
    clReleaseKernel(Kbs);
    clReleaseKernel(Kbf);
    clReleaseKernel(Kwrap);
    clReleaseKernel(Kbpd);
    clReleaseKernel(Kbac);
    clReleaseKernel(Kbacd);
    clReleaseKernel(Kbpd8);
    clReleaseKernel(Kbacd8);
    clReleaseKernel(Kbpd8f);
    clReleaseKernel(Kbagh8);
    clReleaseKernel(Kbpd8fs);
    clReleaseKernel(Kbagh8s);
    clReleaseKernel(Kbpd8fsc);
    clReleaseKernel(Kbagh8sc);
    clReleaseKernel(Kbci);
    clReleaseKernel(Kbcp);
    clReleaseKernel(Kbcd);
    clReleaseKernel(Ksub);
    clReleaseProgram(PR);
    clReleaseCommandQueue(Q);
    clReleaseContext(C);
    return 0;
}
