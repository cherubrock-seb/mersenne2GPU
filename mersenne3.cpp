/*
 * Copyright 2025, cherubrock
 *
 * This program is an OpenCL implementation inspired by "mersenne2.cpp" 
 * originally written by Yves Gallot (Copyright 2025).
 *
 * The original "mersenne2.cpp" was released as free source code. This version 
 * inherits the same spirit: it is free to use, modify, and redistribute.
 *
 * If you make improvements, please consider giving feedback to the author.
 * This code is provided in the hope that it will be useful.
 */

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#include <iostream>
#include <vector>
#include <cstdint>

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
        uint64_t lo = uint64_t(t) & Z61_p;
        uint64_t hi = uint64_t(t >> 61);
        return add(lo, hi);
    }
public:
    Z61() : _n(0) {}
    explicit Z61(uint64_t n): _n(n & Z61_p) {}
    uint64_t get() const { return _n; }
    Z61 operator+(const Z61& o) const { return Z61(add(_n, o._n)); }
    Z61 operator-(const Z61& o) const { return Z61(sub(_n, o._n)); }
    Z61 operator*(const Z61& o) const { return Z61(mul(_n, o._n)); }
    Z61 sqr() const { return Z61(mul(_n, _n)); }
    Z61 lshift(int s) const {
        int ss = s % 61;
        if (!ss) return *this;
        __uint128_t t = __uint128_t(_n) << ss;
        uint64_t lo = uint64_t(t) & Z61_p;
        uint64_t hi = uint64_t(t >> 61);
        return Z61(add(lo, hi));
    }
    Z61 rshift(int s) const { return lshift(61 - (s % 61)); }
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
        uint32_t lo = uint32_t(t & Z31_p);
        uint32_t hi = uint32_t(t >> 31);
        return add(lo, hi);
    }
public:
    Z31() : _n(0) {}
    explicit Z31(uint32_t n): _n(n & Z31_p) {}
    uint32_t get() const { return _n; }
    Z31 operator+(const Z31& o) const { return Z31(add(_n, o._n)); }
    Z31 operator-(const Z31& o) const { return Z31(sub(_n, o._n)); }
    Z31 operator*(const Z31& o) const { return Z31(mul(_n, o._n)); }
    Z31 sqr() const { return Z31(mul(_n, _n)); }
    Z31 lshift(int s) const {
        int ss = s % 31;
        if (!ss) return *this;
        uint64_t t = uint64_t(_n) << ss;
        uint32_t lo = uint32_t(t & Z31_p);
        uint32_t hi = uint32_t(t >> 31);
        return Z31(add(lo, hi));
    }
    Z31 rshift(int s) const { return lshift(31 - (s % 31)); }
};

class GF61 {
    Z61 _a, _b;
    static const uint64_t ORDER = uint64_t(1) << 62;
    static const uint64_t H0 = 264036120304204ull, H1 = 4677669021635377ull;
public:
    GF61() {}
    GF61(const Z61& a, const Z61& b): _a(a), _b(b) {}
    uint64_t s0() const { return _a.get(); }
    uint64_t s1() const { return _b.get(); }
    GF61 operator+(const GF61& o) const { return GF61(_a + o._a, _b + o._b); }
    GF61 operator-(const GF61& o) const { return GF61(_a - o._a, _b - o._b); }
    GF61 mul(const GF61& o) const { return GF61(_a * o._a - _b * o._b, _b * o._a + _a * o._b); }
    GF61 sqr() const {
        Z61 t = _a * _b;
        return GF61(_a.sqr() - _b.sqr(), t + t);
    }
    GF61 conj() const { return GF61(_a, Z61(Z61_p) - _b); }
    GF61 pow(uint64_t e) const {
        GF61 r(Z61(1), Z61(0)), x = *this;
        while (e) {
            if (e & 1) r = r.mul(x);
            x = x.sqr();
            e >>= 1;
        }
        return r;
    }
    static GF61 root_one(size_t n) {
        return GF61(Z61(H0), Z61(H1)).pow(ORDER / n);
    }
    static uint8_t log2_root_two(const size_t n) { return uint8_t(((uint64_t(1) << 60) / n) % 61); }

};

class GF31 {
    Z31 _a, _b;
    static const uint64_t ORDER = uint64_t(1) << 32;
    static const uint32_t H0 = 7735u, H1 = 748621u;
public:
    GF31() {}
    GF31(const Z31& a, const Z31& b): _a(a), _b(b) {}
    uint32_t s0() const { return _a.get(); }
    uint32_t s1() const { return _b.get(); }
    GF31 operator+(const GF31& o) const { return GF31(_a + o._a, _b + o._b); }
    GF31 operator-(const GF31& o) const { return GF31(_a - o._a, _b - o._b); }
    GF31 mul(const GF31& o) const { return GF31(_a * o._a - _b * o._b, _b * o._a + _a * o._b); }
    GF31 sqr() const {
        Z31 t = _a * _b;
        return GF31(_a.sqr() - _b.sqr(), t + t);
    }
    GF31 conj() const { return GF31(_a, Z31(Z31_p) - _b); }
    GF31 pow(uint64_t e) const {
        GF31 r(Z31(1), Z31(0)), x = *this;
        while (e) {
            if (e & 1) r = r.mul(x);
            x = x.sqr();
            e >>= 1;
        }
        return r;
    }
    static GF31 root_one(size_t n) {
        return GF31(Z31(H0), Z31(H1)).pow(ORDER / n);
    }
    static uint8_t log2_root_two(const size_t n) { return uint8_t(((uint64_t(1) << 30) / n) % 31); }

};

class GF61_31 {
    
public:
    GF61 g61;
    GF31 g31;
    GF61_31() {}
    GF61_31(uint64_t a0, uint64_t a1)
        : g61(Z61(a0), Z61(a1)), g31(Z31(a0), Z31(a1)) {}
    GF61_31(const GF61& a, const GF31& b): g61(a), g31(b) {}
    GF61_31 operator+(const GF61_31& o) const {
        return GF61_31(g61 + o.g61, g31 + o.g31);
    }
    GF61_31 operator-(const GF61_31& o) const {
        return GF61_31(g61 - o.g61, g31 - o.g31);
    }
    GF61_31 mul(const GF61_31& o) const {
        return GF61_31(g61.mul(o.g61), g31.mul(o.g31));
    }
    GF61_31 sqr() const {
        return GF61_31(g61.sqr(), g31.sqr());
    }
    GF61_31 pow(uint64_t e) const {
        return GF61_31(g61.pow(e), g31.pow(e));
    }
    static GF61_31 root_one(size_t n) {
        return GF61_31(GF61::root_one(n), GF31::root_one(n));
    }
    
};

struct IBWeight { int w61, w31; };

static size_t bitrev(size_t i, size_t n) {
    size_t r = 0;
    for (; n > 1; n >>= 1, i >>= 1)
        r = (r << 1) | (i & 1);
    return r;
}

const char* KC = R"CLC(
typedef struct {
    ulong lo,hi;
} u128_fake;
#pragma pack(push,1)
typedef ulong  u64;
typedef uint   u32;
//typedef unsigned __int128 u128;
typedef u128_fake u128;

typedef struct {
    u64 s0, s1;
    u32 t0, t1;
} GF;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct { int w61, w31; } IW;
#pragma pack(pop)
#define P61 (((u64)1<<61)-1)
#define P31 (((u32)1<<31)-1)
inline u128 mul_64x64_128(ulong a, ulong b) {
    ulong a_lo = (uint)a;
    ulong a_hi = a >> 32;
    ulong b_lo = (uint)b;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_lo = a_hi * b_lo;
    ulong hi_hi = a_hi * b_hi;

    ulong cross = lo_hi + hi_lo;
    ulong cross_lo = cross << 32;
    ulong cross_hi = cross >> 32;

    ulong lo = lo_lo + cross_lo;
    ulong carry = (lo < lo_lo) ? 1 : 0;
    ulong hi = hi_hi + cross_hi + carry;

    return (u128){ .hi = hi, .lo = lo };
}

u64 add61(u64 a,u64 b){u64 t=a+b;return t-(t>=P61?P61:0);} 
u64 sub61(u64 a,u64 b){u64 t=a-b;return t+(a<b?P61:0);} 
inline u64 mul61(u64 a, u64 b) {
    u128 t = mul_64x64_128(a, b);
    u64 lo = t.lo & P61;
    u64 hi = (t.lo >> 61) | (t.hi << 3); // (64 - 61 = 3)
    return add61(lo, hi);
}



u32 add31(u32 a,u32 b){u32 t=a+b;return t-(t>=P31?P31:0);} 
u32 sub31(u32 a,u32 b){u32 t=a-b;return t+(a<b?P31:0);} 
inline u32 mul31(u32 a, u32 b) {
    u64 t = (u64)a * b;
    u32 lo = (u32)(t & P31);
    u32 hi = (u32)(t >> 31);
    return add31(lo, hi);
}
GF gf_add(GF a,GF b){return (GF){
    add61(a.s0,b.s0),add61(a.s1,b.s1),
    add31(a.t0,b.t0),add31(a.t1,b.t1)};
}

GF gf_sub(GF a,GF b){return (GF){
    sub61(a.s0,b.s0),sub61(a.s1,b.s1),
    sub31(a.t0,b.t0),sub31(a.t1,b.t1)};
}
GF gf_mul(GF a,GF b){
    u64 x0=mul61(a.s0,b.s0), x1=mul61(a.s1,b.s1), y0=mul61(a.s1,b.s0), y1=mul61(a.s0,b.s1); 
    u32 X0=mul31(a.t0,b.t0), X1=mul31(a.t1,b.t1), Y0=mul31(a.t1,b.t0), Y1=mul31(a.t0,b.t1); 
    return (GF){sub61(x0,x1),add61(y0,y1),sub31(X0,X1),add31(Y0,Y1)};
}
inline u128 make_u128(ulong lo, ulong hi){ return (u128){lo,hi}; }
inline u128 add_u128(u128 a, u128 b){
    ulong lo = a.lo + b.lo;
    ulong carry = (lo < a.lo);
    return (u128){ lo, a.hi + b.hi + carry };
}
inline u128 shl_u128(u128 a, uint w){
    return (u128){ a.lo << w, (a.hi << w) | (a.lo >> (64-w)) };
}
inline u128 rshift_u128(u128 a, uint w) {
    if (w == 0) return a;
    if (w < 64) {
        ulong lo = (a.lo >> w) | (a.hi << (64 - w));
        ulong hi =  a.hi >> w;
        return (u128){ .hi = hi, .lo = lo };
    } else {
        ulong lo = a.hi >> (w - 64);
        return (u128){ .hi = 0, .lo = lo };
    }
}

inline ulong lshift_mod61(ulong x, uint s) {
    s %= 61;
    if (s == 0) return x;
    ulong lo = (x << s) & P61;
    ulong hi = (x >> (61 - s));
    return add61(lo, hi);
}


inline ulong rshift_mod61(ulong x, uint s) {
    s %= 61;
    if (s == 0) return x;
    return lshift_mod61(x, 61 - s);
}

inline u32 lshift_mod31(u32 x, uint s) {
    s %= 31;
    if (s == 0) return x;
    u32 lo = (x << s) & P31;
    u32 hi = (x >> (31 - s));
    return add31(lo, hi);
}

inline u32 rshift_mod31(u32 x, uint s) {
    s %= 31;
    if (s == 0) return x;
    return lshift_mod31(x, 31 - s);
}



inline GF rshift_GF(GF z, uint rs0, uint rs1, uint rt0, uint rt1)
{

    return (GF){
        rshift_mod61(z.s0, rs0),
        rshift_mod61(z.s1, rs1),
        rshift_mod31(z.t0, rt0),
        rshift_mod31(z.t1, rt1)
    };
}
inline GF lshift_GF(GF z,
                    uint ls0, uint ls1,
                    uint lt0, uint lt1)
{
    return (GF){
        lshift_mod61(z.s0, ls0),
        lshift_mod61(z.s1, ls1),
        lshift_mod31(z.t0, lt0),
        lshift_mod31(z.t1, lt1)
    };
}


GF gf_addi(GF a, GF b){
    return (GF){
        sub61(a.s0, b.s1),    
        add61(a.s1, b.s0),    
        sub31(a.t0, b.t1),    
        add31(a.t1, b.t0)     
    };
}
GF gf_subi(GF a, GF b){
    return (GF){
        add61(a.s0, b.s1),    
        sub61(a.s1, b.s0),    
        add31(a.t0, b.t1),    
        sub31(a.t1, b.t0)     
    };
}

__kernel void weight(__global GF* z, __global const IW* w)
{
   
    int i = get_global_id(0);
     //printf("i=%d | w61=%u %u | w31=%u %u\n",i,w[2*i + 0].w61, w[2*i + 1].w61,w[2*i + 0].w31, w[2*i + 1].w31);
       
    uint rs0 = w[2*i + 0].w61;
    uint rs1 = w[2*i + 1].w61;
    uint rt0 = w[2*i + 0].w31;
    uint rt1 = w[2*i + 1].w31;
    //printf("i=%d | w61=%u %u | w31=%u %u | rs0=%u rs1=%u rt0=%u rt1=%u\n",i,w[2*i + 0].w61, w[2*i + 1].w61,w[2*i + 0].w31, w[2*i + 1].w31,rs0, rs1, rt0, rt1);
    GF zi = z[i];
    //printf(" Before: s0=%llu s1=%llu | t0=%u t1=%u\n",zi.s0, zi.s1, zi.t0, zi.t1);
    z[i] = lshift_GF(zi, rs0, rs1, rt0, rt1);
    zi = z[i];
    //printf(" After : s0=%llu s1=%llu | t0=%u t1=%u\n",zi.s0, zi.s1, zi.t0, zi.t1);


}


__kernel void forward4(__global GF* z,
                       __global const GF* w,
                       int s,
                       int m,
                       int n)
{
    int gid = get_global_id(0);
    int j   = gid / m;
    int i   = gid % m;

    if (gid == 0) {
        ////printf("forward4 params: s=%d\n", s);
        ////printf("forward4 params: m=%d\n", m);
        ////printf("forward4 params: n=%d\n", n);
    }

    if (j < s) {
        int b = j * (m << 2);

        GF u0 = z[b + i];
        GF u1 = gf_mul(z[b + m + i],     w[2 * (s + j)]);
        GF u2 = gf_mul(z[b + 2 * m + i], w[s + j]);
        GF u3 = gf_mul(z[b + 3 * m + i], w[(n >> 1) + s + j]);

        ////printf("gid=%d j=%d i=%d\n", gid, j, i);

        /* u0 */
        ////printf(" u0.s0=%u\n", (unsigned int)u0.s0);
        ////printf(" u0.s1=%u\n", (unsigned int)u0.s1);
        ////printf(" u0.t0=%u\n",           u0.t0);
        ////printf(" u0.t1=%u\n",           u0.t1);

        /* u1 */
        ////printf(" u1.s0=%u\n", (unsigned int)u1.s0);
        ////printf(" u1.s1=%u\n", (unsigned int)u1.s1);
        ////printf(" u1.t0=%u\n",           u1.t0);
        ////printf(" u1.t1=%u\n",           u1.t1);

        /* u2 */
        ////printf(" u2.s0=%u\n", (unsigned int)u2.s0);
        ////printf(" u2.s1=%u\n", (unsigned int)u2.s1);
        ////printf(" u2.t0=%u\n",           u2.t0);
        ////printf(" u2.t1=%u\n",           u2.t1);

        /* u3 */
        ////printf(" u3.s0=%u\n", (unsigned int)u3.s0);
        ////printf(" u3.s1=%u\n", (unsigned int)u3.s1);
        ////printf(" u3.t0=%u\n",           u3.t0);
        ////printf(" u3.t1=%u\n",           u3.t1);

        /* Calcul des v’s */
        GF v0 = gf_add(u0, u2);
        GF v1 = gf_add(u1, u3);
        GF v2 = gf_sub(u0, u2);
        GF v3 = gf_sub(u1, u3);

        /* v0 */
        ////printf(" v0.s0=%u\n", (unsigned int)v0.s0);
        ////printf(" v0.s1=%u\n", (unsigned int)v0.s1);
        ////printf(" v0.t0=%u\n",           v0.t0);
        ////printf(" v0.t1=%u\n",           v0.t1);

        /* v1 */
        ////printf(" v1.s0=%u\n", (unsigned int)v1.s0);
        ////printf(" v1.s1=%u\n", (unsigned int)v1.s1);
        ////printf(" v1.t0=%u\n",           v1.t0);
        ////printf(" v1.t1=%u\n",           v1.t1);

        /* v2 */
        ////printf(" v2.s0=%u\n", (unsigned int)v2.s0);
        ////printf(" v2.s1=%u\n", (unsigned int)v2.s1);
        ////printf(" v2.t0=%u\n",           v2.t0);
        ////printf(" v2.t1=%u\n",           v2.t1);

        /* v3 */
        ////printf(" v3.s0=%u\n", (unsigned int)v3.s0);
        ////printf(" v3.s1=%u\n", (unsigned int)v3.s1);
        ////printf(" v3.t0=%u\n",           v3.t0);
        ////printf(" v3.t1=%u\n",           v3.t1);

        GF z0 = gf_add(v0, v1);
        GF z1 = gf_sub(v0, v1);
        GF z2 = gf_addi(v2, v3);
        GF z3 = gf_subi(v2, v3);

        /* write z0 */
        ////printf(" write z[%d].s0=%u\n", b + i,    (unsigned int)z0.s0);
        ////printf(" write z[%d].s1=%u\n", b + i,    (unsigned int)z0.s1);
        ////printf(" write z[%d].t0=%u\n", b + i,               z0.t0);
        ////printf(" write z[%d].t1=%u\n", b + i,               z0.t1);
        z[b + i] = z0;

        /* write z1 */
        ////printf(" write z[%d].s0=%u\n", b + m + i, (unsigned int)z1.s0);
        ////printf(" write z[%d].s1=%u\n", b + m + i, (unsigned int)z1.s1);
        ////printf(" write z[%d].t0=%u\n", b + m + i,           z1.t0);
        ////printf(" write z[%d].t1=%u\n", b + m + i,           z1.t1);
        z[b + m + i] = z1;

        /* write z2 */
        ////printf(" write z[%d].s0=%u\n", b + 2*m + i, (unsigned int)z2.s0);
        ////printf(" write z[%d].s1=%u\n", b + 2*m + i, (unsigned int)z2.s1);
        ////printf(" write z[%d].t0=%u\n", b + 2*m + i,           z2.t0);
        ////printf(" write z[%d].t1=%u\n", b + 2*m + i,           z2.t1);
        z[b + 2*m + i] = z2;

        /* write z3 */
        ////printf(" write z[%d].s0=%u\n", b + 3*m + i, (unsigned int)z3.s0);
        ////printf(" write z[%d].s1=%u\n", b + 3*m + i, (unsigned int)z3.s1);
        ////printf(" write z[%d].t0=%u\n", b + 3*m + i,           z3.t0);
        ////printf(" write z[%d].t1=%u\n", b + 3*m + i,           z3.t1);
        z[b + 3*m + i] = z3;
    }
}



// GF61_31::addconj
inline GF gf_addconj(GF a, GF b){
    return (GF){
      add61(a.s0, b.s0), sub61(a.s1, b.s1),
      add31(a.t0, b.t0), sub31(a.t1, b.t1)
    };
}
// GF61_31::subconj
inline GF gf_subconj(GF a, GF b){
    return (GF){
      sub61(a.s0, b.s0), add61(a.s1, b.s1),
      sub31(a.t0, b.t0), add31(a.t1, b.t1)
    };
}

// GF61_31::mulconj
inline GF gf_mulconj(GF a, GF b){
    // croisement GF61
    u64 x = add61(mul61(a.s0, b.s0), mul61(a.s1, b.s1));
    u64 y = sub61(mul61(a.s1, b.s0), mul61(a.s0, b.s1));
    // croisement GF31
    u32 X = add31(mul31(a.t0, b.t0), mul31(a.t1, b.t1));
    u32 Y = sub31(mul31(a.t1, b.t0), mul31(a.t0, b.t1));
    return (GF){ x, y, X, Y };
}


inline GF gf_conj(GF a){
    u64 inv_s1 = P61 - a.s1;
    u64 s1 = (inv_s1 >= P61 ? inv_s1 - P61 : inv_s1);  // si inv_s1 == P61 → 0
    u32 inv_t1 = P31 - a.t1;
    u32 t1 = (inv_t1 >= P31 ? inv_t1 - P31 : inv_t1);
    return (GF){
        a.s0,
        s1,
        a.t0,
        t1
    };
}

// forward2 : DFT radix-2
__kernel void forward2(__global GF* z,
                       __global const GF* w,
                       const int n4)
{
    int j = get_global_id(0);
    GF u0 = z[2*j + 0];
    GF u1 = gf_mul(z[2*j + 1], w[n4 + j]);
    z[2*j + 0] = gf_add(u0, u1);
    z[2*j + 1] = gf_sub(u0, u1);
}

// backward2 : inverse DFT radix-2
__kernel void backward2(__global GF* z,
                        __global const GF* w,
                        const int n4)
{
    int j = get_global_id(0);
    GF u0 = z[2*j + 0];
    GF u1 = z[2*j + 1];
    z[2*j + 0] = gf_add(u0, u1);
    z[2*j + 1] = gf_mulconj( gf_sub(u0, u1), w[n4 + j] );
}


//forward2, sqr, backward2
__kernel void forward_2_sqr_backward2(__global GF* z,
                            __global const GF* w_base,
                            const uint n)
{
    const uint n2 = n >> 1;
    __global GF* zp = z;
    __global const GF* wp = w_base + n;

    {
        GF Z0 = gf_add(zp[0], zp[0]);

        u64 a0 = Z0.s0, a1 = Z0.s1;
        u64 X0  = add61(a0, a1), Xm0 = sub61(a0, a1);
        u64 X0sq  = mul61(X0, X0), Xm0sq = mul61(Xm0, Xm0);
        u64 r0 = add61(X0sq, Xm0sq), r1 = sub61(X0sq, Xm0sq);

        u32 b0 = Z0.t0, b1 = Z0.t1;
        u32 Y0   = add31(b0, b1), Ym0 = sub31(b0, b1);
        u32 Y0sq = mul31(Y0, Y0), Ym0sq= mul31(Ym0, Ym0);
        u32 rt0  = add31(Y0sq, Ym0sq), rt1 = sub31(Y0sq, Ym0sq);

        zp[0] = (GF){ r0, r1, rt0, rt1 };
    }

    {
        GF Z1   = gf_add(zp[1], zp[1]);
        GF Z1sq = gf_mul(Z1, Z1);
        zp[1]    = gf_add(Z1sq, Z1sq);
    }

    const uint msz = 32 - clz(n2);
    for (uint k = 2; k < n2; k += 2) {
        GF wk = wp[k>>1];
        uint msb = 31 - clz(k);
        uint mk = (3u << msb) - k - 1;

        GF zk   = zp[k],    zmk = zp[mk];
        GF Zek  = gf_addconj( zk,    zmk );
        GF Zok  = gf_mul(    gf_subconj(zk, zmk), wk );

        GF Zk   = gf_subi( Zek, Zok ),    Zmk = gf_addi( Zek, Zok );

        GF Zk2  = gf_mul( Zk,  Zk ),      Zmk2 = gf_mul( Zmk, Zmk );

        GF Zek2 = gf_add( Zk2,  Zmk2 );
        GF Zok2 = gf_mulconj( gf_sub(Zk2, Zmk2), wk );

        zp[k]   = gf_addi( Zek2, Zok2 );
        GF tmp  = gf_subi( Zek2, Zok2 );
        zp[mk]  = gf_conj( tmp );
    }
}


__kernel void backward4(__global GF* z,
                        __global const GF* w,
                        int s,
                        int m,
                        int n)
{
    int gid = get_global_id(0);
    int j   = gid / m;
    int i   = gid % m;

    if (j < s) {
        int b = j * (m << 2);

        GF u0 = z[b +   i];
        GF u1 = z[b +   m + i];
        GF u2 = z[b + 2*m + i];
        GF u3 = z[b + 3*m + i];

        GF v0 = gf_add(  u0,   u1);
        GF v1 = gf_sub(  u0,   u1);
        GF v2 = gf_add(  u2,   u3);
        GF v3 = gf_sub(  u3,   u2);

        GF z0 = gf_add(  v0,   v2);
        GF z1 = gf_mulconj(gf_sub(  v0,   v2), w[s + j]);
        GF z2 = gf_mulconj(gf_addi( v1,   v3), w[2*(s + j)]);
        GF z3 = gf_mulconj(gf_subi( v1,   v3), w[(n>>1) + s + j]);

        z[b +   i]     = z0;
        z[b + 2*m + i] = z1;
        z[b +   m + i] = z2;
        z[b + 3*m + i] = z3;
    }
}


__kernel void unweight_norm(__global GF* z, __global const IW* w, int ln)
{
    int i = get_global_id(0);
     ////printf("i=%d | w61=%u %u | w31=%u %u\n",i,       w[2*i + 0].w61, w[2*i + 1].w61, w[2*i + 0].w31, w[2*i + 1].w31);
    uint ln2 = ln + 2;
    uint rs0 = w[2*i + 0].w61 + ln2;
    uint rs1 = w[2*i + 1].w61 + ln2;
    uint rt0 = w[2*i + 0].w31 + ln2;
    uint rt1 = w[2*i + 1].w31 + ln2;
    ////printf("i=%d | ln=%d | w61=%u %u | w31=%u %u | rs0=%u rs1=%u rt0=%u rt1=%u\n",i, ln,    w[2*i + 0].w61, w[2*i + 1].w61,    w[2*i + 0].w31, w[2*i + 1].w31,rs0, rs1, rt0, rt1);
    GF zi = z[i];
    ////printf(" Before: s0=%llu s1=%llu | t0=%u t1=%u\n",zi.s0, zi.s1, zi.t0, zi.t1);
    z[i] = rshift_GF(zi, rs0, rs1, rt0, rt1);
    zi = z[i];
    ////printf(" After : s0=%llu s1=%llu | t0=%u t1=%u\n",zi.s0, zi.s1, zi.t0, zi.t1);
}


inline ulong digit_adc(u128 lhs, uint w, __private u128 *carry){
    u128 s = add_u128(lhs, *carry);
    ulong mask = ((ulong)1 << w) - 1;
    ulong res  = s.lo & mask;
    *carry     = rshift_u128(s, w);
    return res;
}

inline void garner_GF(const GF x, __private u128 *l0, __private u128 *l1){
    u32 n31_0 = x.t0, n31_1 = x.t1;
    u64 u0 = sub61(x.s0, n31_0), u1 = sub61(x.s1, n31_1);
    
    u64 lo0 = (u0 << 31) & P61, hi0 = u0 >> (61-31);
    u64 lo1 = (u1 << 31) & P61, hi1 = u1 >> (61-31);

    u0 = add61(u0, add61(lo0, hi0));
    u1 = add61(u1, add61(lo1, hi1));

    *l0 = make_u128((ulong)n31_0 + ((ulong)u0 << 31) - u0, 0);
    *l1 = make_u128((ulong)n31_1 + ((ulong)u1 << 31) - u1, 0);
}
__kernel void carry(__global GF* z,
                    __global const uint* digit_width,
                    const uint n2)
{
    __private u128 c = make_u128(0, 0);
   //printf("Carry start\n");

    for (uint k = 0; k < n2; ++k) {
        u128 L0, L1;
        garner_GF(z[k], &L0, &L1);
        uint w0 = digit_width[2 * k], w1 = digit_width[2 * k + 1];
        ulong n0 = digit_adc(L0, w0, &c);
        ulong n1 = digit_adc(L1, w1, &c);
        z[k].s0 = n0; z[k].s1 = n1;
        z[k].t0 = (uint)n0; z[k].t1 = (uint)n1;

        //printf("k=%u | L0=(%llu,%llu) L1=(%llu,%llu) | w0=%u w1=%u | n0=%lu n1=%lu | c=(%llu,%llu)\n",k, L0.lo, L0.hi, L1.lo, L1.hi, w0, w1, n0, n1, c.lo, c.hi);
    }

    while ((c.lo | c.hi) != 0) {
        //printf("Carry propagation loop | c=(%llu,%llu)\n", c.lo, c.hi);

        for (uint k = 0; k < n2; ++k) {
            u128 L0 = make_u128(z[k].s0, 0);
            u128 L1 = make_u128(z[k].s1, 0);
            uint w0 = digit_width[2 * k], w1 = digit_width[2 * k + 1];
            ulong n0 = digit_adc(L0, w0, &c);
            ulong n1 = digit_adc(L1, w1, &c);
            z[k].s0 = n0; z[k].s1 = n1;
            z[k].t0 = (uint)n0; z[k].t1 = (uint)n1;

            //printf("k=%u | R0=%lu R1=%lu | w0=%u w1=%u | c=(%llu,%llu)\n",k, n0, n1, w0, w1, c.lo, c.hi);

            if ((c.lo | c.hi) == 0) {
                //printf("Carry cleared at k=%u\n", k);
                break;
            }
        }
    }

    ////printf("Carry end\n");
}
// Subtraction with borrow across a w-bit digit
inline ulong digit_sbc(ulong lhs, uint w, __private uint *borrow) {
    uint b = *borrow;
    ulong mask = ((ulong)1 << w) - 1ul;
    // soustraction de l’ancien borrow
    ulong t = lhs - (ulong)b;
    // nouveau borrow si underflow
    *borrow = (lhs < (ulong)b) ? 1u : 0u;
    // on ne garde que les w bits de poids faible
    return t & mask;
}

// sub_kernel : soustrait `a` à tout z[0..n2-1], en propageant borrow
__kernel void sub_kernel(__global GF*        z,
                         __global const uint* digit_width,
                         const uint           n2,
                         const uint           a)
{
    uint borrow = a;
    while (borrow != 0u) {
        for (uint k = 0; k < n2; ++k) {
            // récupère les deux parties 61-bits
            ulong s0 = z[k].s0;
            ulong s1 = z[k].s1;
            // soustractions digitaires
            ulong n0 = digit_sbc(s0, digit_width[2*k],   &borrow);
            ulong n1 = digit_sbc(s1, digit_width[2*k+1], &borrow);
            // mise à jour
            z[k].s0 = n0;
            z[k].s1 = n1;
            z[k].t0 = (uint)n0;
            z[k].t1 = (uint)n1;
            if (borrow == 0u) break;
        }
    }
}

// is_zero_kernel : vérifie que tous les s0 sont 0
__kernel void is_zero_kernel(__global const GF* z,
                             const uint          n2,
                             __global int*       flag_zero)
{
    for (uint k = 0; k < n2; ++k) {
        if (z[k].s0 != 0ul) {
            *flag_zero = 0;
            return;
        }
    }
}

// is_Mp_kernel : vérifie que chaque (s0,s1)==(2^w0−1, 2^w1−1)
__kernel void is_Mp_kernel(__global const GF*   z,
                           __global const uint* digit_width,
                           const uint           n2,
                           __global int*        flag_mp)
{
    for (uint k = 0; k < n2; ++k) {
        uint w0 = digit_width[2*k];
        uint w1 = digit_width[2*k+1];
        ulong mask0 = ((ulong)1 << w0) - 1ul;
        ulong mask1 = ((ulong)1 << w1) - 1ul;
        if (z[k].s0 != mask0 || z[k].s1 != mask1) {
            *flag_mp = 0;
            return;
        }
    }
}







)CLC";

void debug_read(cl_command_queue Q, cl_mem buf, size_t h, const std::string& stage) {
    size_t cnt = std::min<size_t>(4, h);
    #pragma pack(push,1)
    struct GF_host { uint64_t s0, s1; uint32_t t0, t1; };
    #pragma pack(pop)
    std::vector<GF_host> tmp(cnt);
    clEnqueueReadBuffer(Q, buf, CL_TRUE, 0,
        sizeof(GF_host) * cnt, tmp.data(), 0, nullptr, nullptr);
    std::cout << "Stage=" << stage;
    for(size_t i = 0; i < cnt; ++i) {
        std::cout << " | z[" << i << "] = ("
                  << tmp[i].s0 << "," << tmp[i].s1 << ")/("
                  << tmp[i].t0 << "," << tmp[i].t1 << ")";
    }
    std::cout << "\n";
}

static constexpr uint64_t digit_sbc(const uint64_t lhs, const uint8_t digit_width, uint32_t & carry)
{
    const bool borrow = (lhs < carry);
    const uint64_t r = lhs - carry + (borrow ? (uint64_t(1) << digit_width) : 0);
    carry = borrow ? 1 : 0;
    return r;
}

static uint64_t digit_sbc_host(uint64_t lhs, uint8_t w, uint32_t & borrow)
{
    bool  b = (lhs < borrow);
    uint64_t r = lhs - borrow + (b ? (uint64_t(1) << w) : 0);
    borrow = b ? 1u : 0u;
    return r;
}

bool verify_equals(const std::vector<GF61_31>& host_z,
                   const std::vector<int>& digit_width,
                   uint32_t a)
{
    size_t n2 = host_z.size();
    std::vector<GF61_31> tmp = host_z;

    uint32_t borrow = a;
    while (borrow != 0) {
        for (size_t k = 0; k < n2; ++k) {
            uint64_t s0 = tmp[k].g61.s0();
            uint64_t s1 = tmp[k].g61.s1();
            s0 = digit_sbc_host(s0, digit_width[2*k],   borrow);
            s1 = digit_sbc_host(s1, digit_width[2*k+1], borrow);
            tmp[k] = GF61_31(s0, s1);
            if (borrow == 0) break;
        }
    }
    for (auto &v : tmp) {
        if (v.g61.s0() != 0) return false;
    }
    
    return true;
}

bool verify_is_Mp(const std::vector<GF61_31>& host_z,
                  const std::vector<int>& digit_width)
{
    size_t n2 = host_z.size();
    for (size_t k = 0; k < n2; ++k) {
        uint64_t s0 = host_z[k].g61.s0();
        uint64_t s1 = host_z[k].g61.s1();
        uint64_t mask0 = ((uint64_t)1 << digit_width[2*k]) - 1;
        uint64_t mask1 = ((uint64_t)1 << digit_width[2*k+1]) - 1;
        if (s0 != mask0 || s1 != mask1) return false;
    }
    return true;
}


int main(int argc, char* argv[]){
    cl_platform_id P; cl_uint pn;
    clGetPlatformIDs(1,&P,&pn);
    cl_device_id D; cl_uint dn;
    clGetDeviceIDs(P,CL_DEVICE_TYPE_GPU,1,&D,&dn);
    cl_context C=clCreateContext(nullptr,1,&D,nullptr,nullptr,nullptr);
    //cl_command_queue Q=clCreateCommandQueueWithProperties(C,D,nullptr,nullptr);
    cl_int err2 = CL_SUCCESS;
    cl_command_queue Q = clCreateCommandQueue(C, D, CL_QUEUE_PROFILING_ENABLE, &err2);

    cl_program PR=clCreateProgramWithSource(C,1,&KC,nullptr,nullptr);
    cl_int err = clBuildProgram(PR,1,&D,nullptr,nullptr,nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(PR, D, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(PR, D, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        return 1;
    }

    cl_kernel Kw=clCreateKernel(PR,"weight",nullptr);
    cl_kernel Kf=clCreateKernel(PR,"forward4",nullptr);
    cl_kernel Kf2 = clCreateKernel(PR, "forward2",   nullptr);
    cl_kernel Ks  = clCreateKernel(PR, "forward_2_sqr_backward2", nullptr);
    cl_kernel Kb2 = clCreateKernel(PR, "backward2",  nullptr);
    cl_kernel Ksub    = clCreateKernel(PR, "sub_kernel",    nullptr);
    cl_kernel Kisz    = clCreateKernel(PR, "is_zero_kernel",nullptr);
    cl_kernel KisMp   = clCreateKernel(PR, "is_Mp_kernel",  nullptr);

    cl_kernel Kb=clCreateKernel(PR,"backward4",nullptr);
    cl_kernel Ku=clCreateKernel(PR,"unweight_norm",nullptr);
    cl_kernel Kc=clCreateKernel(PR,"carry",nullptr);
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <start_p>\n";
        return 1;
    }
    uint32_t start_p = static_cast<uint32_t>(std::atoi(argv[1]));
    for(uint32_t p=start_p;p<=start_p;p+=2){
        
        /*bool isp=true;
        for(uint32_t d=3;d*d<=p;d+=2) if(p%d==0){isp=false;break;}
        if(!isp) continue;*/

        int ln = 2, w;
        do {
            ++ln;
            w = int(p >> ln);
        } while (ln + 2 * (w + 1) >= 92);

        if ((ln & 1) == 0) {
            ++ln;
        }

        size_t h = size_t{1} << (ln - 1);  
        size_t n = h << 1;                 
        std::cout << "ln = " << ln 
                << ", h = " << h 
                << " (power of 4), n = " << n 
                << std::endl;


        

        std::cout << "TRANSFORM SIZE = " << n <<std::endl;

        std::vector<GF61_31> z(h), wv(5*h/2);
        std::vector<IBWeight> iw_fwd(n), iw_inv(n);
        std::vector<int> dw(n);

        z[0]=GF61_31(3,0);
        for(size_t i=1;i<h;++i) z[i]=GF61_31(0,0);


        for(size_t s=1;s<=h/2;s<<=1){
            GF61_31 R=GF61_31::root_one(2*s);
            for(size_t j=0;j<s;++j) wv[s+j]=R.pow(bitrev(j,s));
        }
        for(size_t s=1;s<=h/2;s<<=1)
            for(size_t j=0;j<s;++j)
                wv[h+s+j]=wv[s+j].mul(wv[2*(s+j)]);
        GF61_31 RN=GF61_31::root_one(n);
        for(size_t j=0,m=h/2;j<m;++j)
            wv[2*h+j]=RN.pow(bitrev(j,m));

        uint32_t o = 0;
        const uint8_t q_n = uint8_t(p / n);
        for (size_t j = 1; j <= n; ++j) {
            uint64_t qj       = uint64_t(p) * j;
            uint32_t ceil_qj  = uint32_t((qj - 1) / n + 1);
            int      c        = int(ceil_qj - o);
            // c doit être q_n ou q_n+1
            dw[j - 1]         = c;
            o                 = ceil_qj;
        }

        //printf("digit_width = { ");
        for (size_t j = 0; j < n; ++j) {
            //printf("%u ", dw[j]); 
        }
        //printf("}\n");

        //dw[n-1] = 0;

        iw_fwd[0]={0,0};
        for(size_t i=1;i<n;++i){
            uint64_t qj=uint64_t(p)*i;
            int c=int(((qj-1)/n+1)-((qj-1)/n));
            iw_fwd[i]={c,c};
        }
/*        int lr2_61 = int(((uint64_t)1 << 60) / n % 61);
        int lr2_31 = int(((uint64_t)1 << 30) / n % 31);
        for (size_t i = 0; i < n; ++i) {
            uint64_t qj = uint64_t(p) * i;
            uint32_t r = uint32_t(qj % n);
            iw_inv[i].w61 = int((lr2_61 * (n - r)) % 61);
            iw_inv[i].w31 = int((lr2_31 * (n - r)) % 31);
        }
        iw_inv[0].w61 = 0;
        iw_inv[0].w31 = 0;*/
std::vector<IBWeight> w_ib(n);
const uint8_t lr2_61 = GF61::log2_root_two(n);
const uint8_t lr2_31 = GF31::log2_root_two(n);
for (size_t j = 0; j < n; ++j) {
    uint64_t qj = uint64_t(p) * j;
    uint32_t r  = uint32_t(qj % n);
    const uint8_t w61 = uint8_t((lr2_61 * (n - r)) % 61);
    const uint8_t w31 = uint8_t((lr2_31 * (n - r)) % 31);
    w_ib[j] = { w61, w31 };
}

w_ib[0] = { 0, 0 };
/*for(int j = 0; j < 8; ++j)
  std::cout << "host w_ib["<<j<<"] = ("
            << int(w_ib[j].w61) << "," << int(w_ib[j].w31) << ")\n";*/
cl_int err;
cl_mem wib_buf = clCreateBuffer(
    C,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof(IBWeight) * n,
    w_ib.data(),
    &err
);
if (err != CL_SUCCESS) { }
        int flag=1;
        cl_mem Bz=clCreateBuffer(C,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(z[0])*h,z.data(),nullptr);
        cl_mem Bw=clCreateBuffer(C,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(wv[0])*wv.size(),wv.data(),nullptr);
        cl_mem Bfwd=clCreateBuffer(C,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(iw_fwd[0])*n,iw_fwd.data(),nullptr);
        cl_mem Binv=clCreateBuffer(C,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(iw_inv[0])*n,iw_inv.data(),nullptr);
        cl_mem BdW = clCreateBuffer(C,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(dw[0]) * dw.size(),
                    dw.data(), nullptr);
        cl_mem Bf=clCreateBuffer(C,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(flag),&flag,nullptr);
        for (uint32_t iter = 0; iter < p; ++iter) {
            { // weight
                size_t gs=h;
                clSetKernelArg(Kw,0,sizeof(Bz),&Bz);
                clSetKernelArg(Kw, 1, sizeof(cl_mem),&wib_buf);
                clEnqueueNDRangeKernel(Q,Kw,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
                //clFinish(Q);
                //debug_read(Q,Bz,h,">>>weight");
            }

            { // forward radix-4
                size_t m=h/4, s=1;
                for(; m>=1; m/=4, s*=4){
                    size_t gs=s*m;
                    //std::cout << "m=" << m << std::endl;
                    //debug_read(Q,Bz,h,"Bz");
                    //debug_read(Q,Bw,h,"Bw");
                    clSetKernelArg(Kf,0,sizeof(Bz),&Bz);
                    clSetKernelArg(Kf,1,sizeof(Bw),&Bw);
                    clSetKernelArg(Kf,2,sizeof(int),&s);
                    clSetKernelArg(Kf,3,sizeof(int),&m);
                    clSetKernelArg(Kf,4,sizeof(int),&n);
                    clEnqueueNDRangeKernel(Q,Kf,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
                    //clFinish(Q);
                    
                    //debug_read(Q,Bz,h,">>>forward4");
                }
               
            }
            
            // === début sqr2() radix-2 ===
/*            const int n4 = int(h / 2);

            // 1) forward2
            {
                size_t gs1 = size_t(h/2);
                clSetKernelArg(Kf2, 0, sizeof(Bz), &Bz);
                clSetKernelArg(Kf2, 1, sizeof(Bw), &Bw);
                clSetKernelArg(Kf2, 2, sizeof(n4), &n4);
                clEnqueueNDRangeKernel(Q, Kf2, 1, nullptr, &gs1, nullptr, 0, nullptr, nullptr);
                //clFinish(Q);
                //debug_read(Q, Bz, h, "forward (sqr2)");
            }
*/
            {
                size_t gs2 = h;
                clSetKernelArg(Ks, 0, sizeof(cl_mem), &Bz);
                clSetKernelArg(Ks, 1, sizeof(cl_mem), &Bw);
                int nin = int(n);
                clSetKernelArg(Ks, 2, sizeof(int), &nin);
                clEnqueueNDRangeKernel(Q, Ks, 1, nullptr, &gs2, nullptr, 0, nullptr, nullptr);
                //clFinish(Q);
                //debug_read(Q, Bz, h, ">>>pointwise_sqr");
            }
/*
            // 3) backward2
            {
                size_t gs3 = size_t(h/2); 
                clSetKernelArg(Kb2, 0, sizeof(Bz), &Bz);
                clSetKernelArg(Kb2, 1, sizeof(Bw), &Bw);
                clSetKernelArg(Kb2, 2, sizeof(n), &n4);
                clEnqueueNDRangeKernel(Q, Kb2, 1, nullptr, &gs3, nullptr, 0, nullptr, nullptr);
                //clFinish(Q);
                //debug_read(Q, Bz, h, "backward (sqr2)");
            }
            // === fin sqr2() radix-2 ===

*/
            { // inverse radix-4
                size_t m0 = 1, s0 = h/4;
                for(size_t m=m0, s=s0; s>=1; m*=4, s/=4){
                    size_t gs=s*m;
                    //std::cout << "gs=" << gs << std::endl;
                    //debug_read(Q,Bz,h,"Bz");
                    //debug_read(Q,Bw,h,"Bw");
                    clSetKernelArg(Kb,0,sizeof(Bz),&Bz);
                    clSetKernelArg(Kb,1,sizeof(Bw),&Bw);
                    clSetKernelArg(Kb,2,sizeof(int),&s);
                    clSetKernelArg(Kb,3,sizeof(int),&m);
                    clSetKernelArg(Kb,4,sizeof(int),&n);
                    clEnqueueNDRangeKernel(Q,Kb,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
                    //clFinish(Q);
                    //debug_read(Q,Bz,h,">>>backward4");
                }
                
            }

            { // unweight and normalize
                size_t gs=h;
                clSetKernelArg(Ku,0,sizeof(Bz),&Bz);
                clSetKernelArg(Ku, 1, sizeof(cl_mem), &wib_buf);
                clSetKernelArg(Ku,2,sizeof(int),&ln);
                clEnqueueNDRangeKernel(Q,Ku,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
                //clFinish(Q);
                //debug_read(Q,Bz,h,">>>unweight_norm");
            }

            {
                size_t gs = 1;
                clSetKernelArg(Kc, 0, sizeof(Bz),   &Bz);
                clSetKernelArg(Kc, 1, sizeof(BdW),   &BdW);
                clSetKernelArg(Kc, 2, sizeof(cl_uint), &h);
                clEnqueueNDRangeKernel(Q, Kc, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
                //clFinish(Q);  
                //debug_read(Q, Bz, h, ">>>carry");
                
            }
            if(iter%800==0){
                clFinish(Q);
            }
        }
        //std::cout << "Loop is done check result in progress\n";
        clFinish(Q);
        std::cout << "Loop is done check result in progress\n";
        
        //debug_read(Q, Bz, h, ">>>end"); 
        std::vector<GF61_31> host_z(h);
        cl_int err2 = clEnqueueReadBuffer(
            Q,
            Bz,
            CL_TRUE,
            0,
            sizeof(GF61_31)*h,
            host_z.data(),
            0, nullptr, nullptr
        );

        /*for(size_t i = 0; i < h; ++i) {
            std::cout << " | z[" << i << "] = ("
                    << host_z[i].g61.s0() << "," << host_z[i].g61.s1() << ")/("
                    << host_z[i].g31.s0() << "," << host_z[i].g31.s1() << ")";
        }*/
        if (err2 != CL_SUCCESS) {
            std::cerr << "Erreur clEnqueueReadBuffer pour Bz : " << err2 << "\n";
            std::exit(1);
        }

        if (verify_equals(host_z, dw, 9)) {
            std::cout << "2^" << p << " - 1" << " probably prime (residue is 9)\n";
        }
        else if (verify_is_Mp(host_z, dw)) {
            std::cout << "2^" << p << " - 1" << " probably prime (residue is 9)\n";
        }
        else {
            std::cout << "2^" << p << " - 1" <<  " is composite (residue is not 9)\n";
        }
        std::cout<<"p="<<p<<"\n";
        

        clReleaseMemObject(Bz);
        clReleaseMemObject(Bw);
        clReleaseMemObject(Bfwd);
        clReleaseMemObject(Binv);
        clReleaseMemObject(BdW);
        clReleaseMemObject(Bf);
    }

    clReleaseKernel(Kw);
    clReleaseKernel(Kf);
    clReleaseKernel(Kf2);
    clReleaseKernel(Ks);
    clReleaseKernel(Kb2);
    clReleaseKernel(Kb);
    clReleaseKernel(Ku);
    clReleaseKernel(Kc);
    clReleaseProgram(PR);
    clReleaseCommandQueue(Q);
    clReleaseContext(C);
    return 0;
}
