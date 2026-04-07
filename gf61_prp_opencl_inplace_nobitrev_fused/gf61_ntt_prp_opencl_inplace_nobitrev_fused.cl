typedef ulong u64;
typedef uint u32;
typedef uchar u8;
typedef ulong2 GF;

#define P61 (((u64)1 << 61) - 1)

inline u64 add61(u64 a, u64 b) {
    u64 s = a + b;
    s = (s & P61) + (s >> 61);
    if (s >= P61) s -= P61;
    return s;
}

inline u64 sub61(u64 a, u64 b) {
    return (a >= b) ? (a - b) : (P61 - (b - a));
}

inline u64 mul61(u64 a, u64 b) {
    const u32 a0 = (u32)a;
    const u32 a1 = (u32)(a >> 32);
    const u32 b0 = (u32)b;
    const u32 b1 = (u32)(b >> 32);

    const u64 p0 = (u64)a0 * (u64)b0;
    const u64 p1 = (u64)a0 * (u64)b1 + (u64)a1 * (u64)b0;
    const u64 p2 = (u64)a1 * (u64)b1;

    const u64 lo = p0 + (p1 << 32);
    const u64 carry = (lo < p0) ? 1ul : 0ul;
    const u64 hi = p2 + (p1 >> 32) + carry;

    u64 r = (lo & P61) + (lo >> 61) + (hi << 3);
    r = (r & P61) + (r >> 61);
    r = (r & P61) + (r >> 61);
    if (r >= P61) r -= P61;
    return r;
}

inline u64 lshift61(u64 x, u32 s) {
    s %= 61u;
    if (s == 0u) return x;
    u64 lo = (x << s) & P61;
    u64 hi = x >> (61u - s);
    u64 r = lo + hi;
    r = (r & P61) + (r >> 61);
    if (r >= P61) r -= P61;
    return r;
}

inline u64 rshift61(u64 x, u32 s) {
    s %= 61u;
    return (s == 0u) ? x : lshift61(x, 61u - s);
}

inline GF gf_add(GF x, GF y) {
    return (GF)(add61(x.s0, y.s0), add61(x.s1, y.s1));
}

inline GF gf_sub(GF x, GF y) {
    return (GF)(sub61(x.s0, y.s0), sub61(x.s1, y.s1));
}

inline GF gf_mul(GF x, GF y) {
    const u64 ac = mul61(x.s0, y.s0);
    const u64 bd = mul61(x.s1, y.s1);
    const u64 ad = mul61(x.s0, y.s1);
    const u64 bc = mul61(x.s1, y.s0);
    return (GF)(sub61(ac, bd), add61(ad, bc));
}

inline GF gf_sqr(GF x) {
    const u64 aa = mul61(x.s0, x.s0);
    const u64 bb = mul61(x.s1, x.s1);
    const u64 ab = mul61(x.s0, x.s1);
    return (GF)(sub61(aa, bb), add61(ab, ab));
}

__kernel void gf61_weight_first_stage_dif(__global const u64* digits,
                                          __global const u8* shifts,
                                          __global GF* a,
                                          __global const GF* twiddles,
                                          const u32 tw_offset,
                                          const u32 len,
                                          const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = (GF)(lshift61(digits[i0], (u32)shifts[i0]), 0ul);
    const GF v = (GF)(lshift61(digits[i1], (u32)shifts[i1]), 0ul);
    a[i0] = gf_add(u, v);
    a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
}

__kernel void gf61_last_stage_dit_unweight(__global GF* a,
                                           __global const GF* twiddles,
                                           __global const u8* shifts,
                                           __global u64* digits,
                                           const GF scale_factor,
                                           const u32 tw_offset,
                                           const u32 len,
                                           const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = a[i0];
    const GF v = gf_mul(a[i1], twiddles[tw_offset + j]);
    const GF z0 = gf_mul(gf_add(u, v), scale_factor);
    const GF z1 = gf_mul(gf_sub(u, v), scale_factor);
    digits[i0] = rshift61(z0.s0, (u32)shifts[i0]);
    digits[i1] = rshift61(z1.s0, (u32)shifts[i1]);
}

__kernel void gf61_ntt_stage_dif(__global GF* a,
                                 __global const GF* twiddles,
                                 const u32 tw_offset,
                                 const u32 len,
                                 const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = a[i0];
    const GF v = a[i1];
    a[i0] = gf_add(u, v);
    a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
}

__kernel void gf61_ntt_stage_dit(__global GF* a,
                                 __global const GF* twiddles,
                                 const u32 tw_offset,
                                 const u32 len,
                                 const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = a[i0];
    const GF v = gf_mul(a[i1], twiddles[tw_offset + j]);
    a[i0] = gf_add(u, v);
    a[i1] = gf_sub(u, v);
}

__kernel void gf61_pointwise_sqr(__global GF* a, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) a[gid] = gf_sqr(a[gid]);
}

__kernel void gf61_scale(__global GF* a, const GF factor, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) a[gid] = gf_mul(a[gid], factor);
}

__kernel void gf61_mul_small_digits(__global u64* digits, const u32 k, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) digits[gid] *= (u64)k;
}

inline int all_max_digits(__global const u64* digits, __global const u8* widths, const u32 n) {
    for (u32 i = 0; i < n; ++i) {
        const u32 w = (u32)widths[i];
        const u64 mask = ((u64)1 << w) - 1ul;
        if (digits[i] != mask) return 0;
    }
    return 1;
}

__kernel void gf61_carry_normalize_serial(__global u64* digits,
                                          __global const u8* widths,
                                          const u32 n) {
    if (get_global_id(0) != 0) return;

    while (1) {
        u64 carry = 0;
        for (u32 i = 0; i < n; ++i) {
            const u32 w = (u32)widths[i];
            const u64 mask = ((u64)1 << w) - 1ul;
            const u64 total = digits[i] + carry;
            digits[i] = total & mask;
            carry = total >> w;
        }
        if (carry == 0ul) break;
        digits[0] += carry;
    }

    if (all_max_digits(digits, widths, n)) {
        for (u32 i = 0; i < n; ++i) digits[i] = 0ul;
    }
}
