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
                                           const u32 log_n,
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
    const GF z0 = gf_add(u, v);
    const GF z1 = gf_sub(u, v);
    digits[i0] = rshift61(z0.s0, (u32)shifts[i0] + log_n);
    digits[i1] = rshift61(z1.s0, (u32)shifts[i1] + log_n);
}


#define DECL_WEIGHT_FIRST_STAGE(NAME, WG) __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(__global const u64* digits,           __global const u8* shifts,           __global GF* a,           __global const GF* twiddles,           const u32 tw_offset,           const u32 len,           const u32 half_len) {     const u32 gid = (u32)get_global_id(0);     const u32 block = gid / half_len;     const u32 j = gid - block * half_len;     const u32 base = block * len;     const u32 i0 = base + j;     const u32 i1 = i0 + half_len;     const GF u = (GF)(lshift61(digits[i0], (u32)shifts[i0]), 0ul);     const GF v = (GF)(lshift61(digits[i1], (u32)shifts[i1]), 0ul);     a[i0] = gf_add(u, v);     a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]); }

#define DECL_LAST_STAGE_UNWEIGHT(NAME, WG) __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(__global GF* a,           __global const GF* twiddles,           __global const u8* shifts,           __global u64* digits,           const u32 log_n,           const u32 tw_offset,           const u32 len,           const u32 half_len) {     const u32 gid = (u32)get_global_id(0);     const u32 block = gid / half_len;     const u32 j = gid - block * half_len;     const u32 base = block * len;     const u32 i0 = base + j;     const u32 i1 = i0 + half_len;     const GF u = a[i0];     const GF v = gf_mul(a[i1], twiddles[tw_offset + j]);     const GF z0 = gf_add(u, v);     const GF z1 = gf_sub(u, v);     digits[i0] = rshift61(z0.s0, (u32)shifts[i0] + log_n);     digits[i1] = rshift61(z1.s0, (u32)shifts[i1] + log_n); }

DECL_WEIGHT_FIRST_STAGE(gf61_weight_first_stage_dif_wg16, 16u)
DECL_WEIGHT_FIRST_STAGE(gf61_weight_first_stage_dif_wg64, 64u)
DECL_LAST_STAGE_UNWEIGHT(gf61_last_stage_dit_unweight_wg16, 16u)
DECL_LAST_STAGE_UNWEIGHT(gf61_last_stage_dit_unweight_wg64, 64u)

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



__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dif_radix4(__global GF* a,
                               __global const GF* tw_stage1,
                               __global const GF* tw_stage2,
                               const u32 tw_offset1,
                               const u32 tw_offset2,
                               const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 quarter = len >> 2;
    const u32 block = gid / quarter;
    const u32 j = gid - block * quarter;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + quarter;
    const u32 i2 = i1 + quarter;
    const u32 i3 = i2 + quarter;
    __local GF scratch[256];
    GF a0 = a[i0];
    GF a1 = a[i1];
    GF a2 = a[i2];
    GF a3 = a[i3];
    scratch[(lid << 2) + 0u] = a0;
    scratch[(lid << 2) + 1u] = a1;
    scratch[(lid << 2) + 2u] = a2;
    scratch[(lid << 2) + 3u] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    a0 = scratch[(lid << 2) + 0u];
    a1 = scratch[(lid << 2) + 1u];
    a2 = scratch[(lid << 2) + 2u];
    a3 = scratch[(lid << 2) + 3u];
    const GF b0 = gf_add(a0, a2);
    const GF b2 = gf_mul(gf_sub(a0, a2), tw_stage1[tw_offset1 + j]);
    const GF b1 = gf_add(a1, a3);
    const GF b3 = gf_mul(gf_sub(a1, a3), tw_stage1[tw_offset1 + j + quarter]);
    a[i0] = gf_add(b0, b1);
    a[i1] = gf_mul(gf_sub(b0, b1), tw_stage2[tw_offset2 + j]);
    a[i2] = gf_add(b2, b3);
    a[i3] = gf_mul(gf_sub(b2, b3), tw_stage2[tw_offset2 + j]);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dit_radix4(__global GF* a,
                               __global const GF* tw_stage1,
                               __global const GF* tw_stage2,
                               const u32 tw_offset1,
                               const u32 tw_offset2,
                               const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 half_len_local = len >> 1;
    const u32 block = gid / half_len_local;
    const u32 j = gid - block * half_len_local;
    const u32 base = block * (len << 1);
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len_local;
    const u32 i2 = i0 + len;
    const u32 i3 = i2 + half_len_local;
    __local GF scratch[256];
    GF x0 = a[i0];
    GF x1 = a[i1];
    GF x2 = a[i2];
    GF x3 = a[i3];
    scratch[(lid << 2) + 0u] = x0;
    scratch[(lid << 2) + 1u] = x1;
    scratch[(lid << 2) + 2u] = x2;
    scratch[(lid << 2) + 3u] = x3;
    barrier(CLK_LOCAL_MEM_FENCE);
    x0 = scratch[(lid << 2) + 0u];
    x1 = scratch[(lid << 2) + 1u];
    x2 = scratch[(lid << 2) + 2u];
    x3 = scratch[(lid << 2) + 3u];
    const GF y0 = gf_add(x0, gf_mul(x1, tw_stage1[tw_offset1 + j]));
    const GF y1 = gf_sub(x0, gf_mul(x1, tw_stage1[tw_offset1 + j]));
    const GF y2 = gf_add(x2, gf_mul(x3, tw_stage1[tw_offset1 + j]));
    const GF y3 = gf_sub(x2, gf_mul(x3, tw_stage1[tw_offset1 + j]));
    a[i0] = gf_add(y0, gf_mul(y2, tw_stage2[tw_offset2 + j]));
    a[i2] = gf_sub(y0, gf_mul(y2, tw_stage2[tw_offset2 + j]));
    a[i1] = gf_add(y1, gf_mul(y3, tw_stage2[tw_offset2 + j + half_len_local]));
    a[i3] = gf_sub(y1, gf_mul(y3, tw_stage2[tw_offset2 + j + half_len_local]));
}


__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dif_radix8(__global GF* a,
                               __global const GF* tw_stage1,
                               __global const GF* tw_stage2,
                               __global const GF* tw_stage3,
                               const u32 tw_offset1,
                               const u32 tw_offset2,
                               const u32 tw_offset3,
                               const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 eighth = len >> 3;
    const u32 block = gid / eighth;
    const u32 j = gid - block * eighth;
    const u32 base = block * len;
    __local GF scratch[512];
    GF x0 = a[base + j + 0u * eighth];
    GF x1 = a[base + j + 1u * eighth];
    GF x2 = a[base + j + 2u * eighth];
    GF x3 = a[base + j + 3u * eighth];
    GF x4 = a[base + j + 4u * eighth];
    GF x5 = a[base + j + 5u * eighth];
    GF x6 = a[base + j + 6u * eighth];
    GF x7 = a[base + j + 7u * eighth];
    const u32 off = lid << 3;
    scratch[off + 0u] = x0; scratch[off + 1u] = x1; scratch[off + 2u] = x2; scratch[off + 3u] = x3;
    scratch[off + 4u] = x4; scratch[off + 5u] = x5; scratch[off + 6u] = x6; scratch[off + 7u] = x7;
    barrier(CLK_LOCAL_MEM_FENCE);
    x0 = scratch[off + 0u]; x1 = scratch[off + 1u]; x2 = scratch[off + 2u]; x3 = scratch[off + 3u];
    x4 = scratch[off + 4u]; x5 = scratch[off + 5u]; x6 = scratch[off + 6u]; x7 = scratch[off + 7u];

    const GF s10 = gf_add(x0, x4);
    const GF s14 = gf_mul(gf_sub(x0, x4), tw_stage1[tw_offset1 + j]);
    const GF s11 = gf_add(x1, x5);
    const GF s15 = gf_mul(gf_sub(x1, x5), tw_stage1[tw_offset1 + j + eighth]);
    const GF s12 = gf_add(x2, x6);
    const GF s16 = gf_mul(gf_sub(x2, x6), tw_stage1[tw_offset1 + j + (eighth << 1)]);
    const GF s13 = gf_add(x3, x7);
    const GF s17 = gf_mul(gf_sub(x3, x7), tw_stage1[tw_offset1 + j + 3u * eighth]);

    const GF s20 = gf_add(s10, s12);
    const GF s22 = gf_mul(gf_sub(s10, s12), tw_stage2[tw_offset2 + j]);
    const GF s21 = gf_add(s11, s13);
    const GF s23 = gf_mul(gf_sub(s11, s13), tw_stage2[tw_offset2 + j + eighth]);
    const GF s24 = gf_add(s14, s16);
    const GF s26 = gf_mul(gf_sub(s14, s16), tw_stage2[tw_offset2 + j]);
    const GF s25 = gf_add(s15, s17);
    const GF s27 = gf_mul(gf_sub(s15, s17), tw_stage2[tw_offset2 + j + eighth]);

    a[base + j + 0u * eighth] = gf_add(s20, s21);
    a[base + j + 1u * eighth] = gf_mul(gf_sub(s20, s21), tw_stage3[tw_offset3 + j]);
    a[base + j + 2u * eighth] = gf_add(s22, s23);
    a[base + j + 3u * eighth] = gf_mul(gf_sub(s22, s23), tw_stage3[tw_offset3 + j]);
    a[base + j + 4u * eighth] = gf_add(s24, s25);
    a[base + j + 5u * eighth] = gf_mul(gf_sub(s24, s25), tw_stage3[tw_offset3 + j]);
    a[base + j + 6u * eighth] = gf_add(s26, s27);
    a[base + j + 7u * eighth] = gf_mul(gf_sub(s26, s27), tw_stage3[tw_offset3 + j]);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dit_radix8(__global GF* a,
                               __global const GF* tw_stage1,
                               __global const GF* tw_stage2,
                               __global const GF* tw_stage3,
                               const u32 tw_offset1,
                               const u32 tw_offset2,
                               const u32 tw_offset3,
                               const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 half_len_local = len >> 1;
    const u32 block = gid / half_len_local;
    const u32 j = gid - block * half_len_local;
    const u32 base = block * (len << 2);
    __local GF scratch[512];
    GF x0 = a[base + j + 0u * half_len_local];
    GF x1 = a[base + j + 1u * half_len_local];
    GF x2 = a[base + j + 2u * half_len_local];
    GF x3 = a[base + j + 3u * half_len_local];
    GF x4 = a[base + j + 4u * half_len_local];
    GF x5 = a[base + j + 5u * half_len_local];
    GF x6 = a[base + j + 6u * half_len_local];
    GF x7 = a[base + j + 7u * half_len_local];
    const u32 off = lid << 3;
    scratch[off + 0u] = x0; scratch[off + 1u] = x1; scratch[off + 2u] = x2; scratch[off + 3u] = x3;
    scratch[off + 4u] = x4; scratch[off + 5u] = x5; scratch[off + 6u] = x6; scratch[off + 7u] = x7;
    barrier(CLK_LOCAL_MEM_FENCE);
    x0 = scratch[off + 0u]; x1 = scratch[off + 1u]; x2 = scratch[off + 2u]; x3 = scratch[off + 3u];
    x4 = scratch[off + 4u]; x5 = scratch[off + 5u]; x6 = scratch[off + 6u]; x7 = scratch[off + 7u];

    const GF s10 = gf_add(x0, gf_mul(x1, tw_stage1[tw_offset1 + j]));
    const GF s11 = gf_sub(x0, gf_mul(x1, tw_stage1[tw_offset1 + j]));
    const GF s12 = gf_add(x2, gf_mul(x3, tw_stage1[tw_offset1 + j]));
    const GF s13 = gf_sub(x2, gf_mul(x3, tw_stage1[tw_offset1 + j]));
    const GF s14 = gf_add(x4, gf_mul(x5, tw_stage1[tw_offset1 + j]));
    const GF s15 = gf_sub(x4, gf_mul(x5, tw_stage1[tw_offset1 + j]));
    const GF s16 = gf_add(x6, gf_mul(x7, tw_stage1[tw_offset1 + j]));
    const GF s17 = gf_sub(x6, gf_mul(x7, tw_stage1[tw_offset1 + j]));

    const GF s20 = gf_add(s10, gf_mul(s12, tw_stage2[tw_offset2 + j]));
    const GF s22 = gf_sub(s10, gf_mul(s12, tw_stage2[tw_offset2 + j]));
    const GF s21 = gf_add(s11, gf_mul(s13, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s23 = gf_sub(s11, gf_mul(s13, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s24 = gf_add(s14, gf_mul(s16, tw_stage2[tw_offset2 + j]));
    const GF s26 = gf_sub(s14, gf_mul(s16, tw_stage2[tw_offset2 + j]));
    const GF s25 = gf_add(s15, gf_mul(s17, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s27 = gf_sub(s15, gf_mul(s17, tw_stage2[tw_offset2 + j + half_len_local]));

    a[base + j + 0u * half_len_local] = gf_add(s20, gf_mul(s24, tw_stage3[tw_offset3 + j]));
    a[base + j + 1u * half_len_local] = gf_sub(s20, gf_mul(s24, tw_stage3[tw_offset3 + j]));
    a[base + j + 2u * half_len_local] = gf_add(s21, gf_mul(s25, tw_stage3[tw_offset3 + j + half_len_local]));
    a[base + j + 3u * half_len_local] = gf_sub(s21, gf_mul(s25, tw_stage3[tw_offset3 + j + half_len_local]));
    a[base + j + 4u * half_len_local] = gf_add(s22, gf_mul(s26, tw_stage3[tw_offset3 + j + (half_len_local << 1)]));
    a[base + j + 5u * half_len_local] = gf_sub(s22, gf_mul(s26, tw_stage3[tw_offset3 + j + (half_len_local << 1)]));
    a[base + j + 6u * half_len_local] = gf_add(s23, gf_mul(s27, tw_stage3[tw_offset3 + j + 3u * half_len_local]));
    a[base + j + 7u * half_len_local] = gf_sub(s23, gf_mul(s27, tw_stage3[tw_offset3 + j + 3u * half_len_local]));
}



__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dif_radix16(__global GF* a,
                                __global const GF* tw_stage1,
                                __global const GF* tw_stage2,
                                __global const GF* tw_stage3,
                                __global const GF* tw_stage4,
                                const u32 tw_offset1,
                                const u32 tw_offset2,
                                const u32 tw_offset3,
                                const u32 tw_offset4,
                                const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 sixteenth = len >> 4;
    const u32 block = gid / sixteenth;
    const u32 j = gid - block * sixteenth;
    const u32 base = block * len;
    __local GF scratch[1024];
    GF x0  = a[base + j + 0u  * sixteenth];
    GF x1  = a[base + j + 1u  * sixteenth];
    GF x2  = a[base + j + 2u  * sixteenth];
    GF x3  = a[base + j + 3u  * sixteenth];
    GF x4  = a[base + j + 4u  * sixteenth];
    GF x5  = a[base + j + 5u  * sixteenth];
    GF x6  = a[base + j + 6u  * sixteenth];
    GF x7  = a[base + j + 7u  * sixteenth];
    GF x8  = a[base + j + 8u  * sixteenth];
    GF x9  = a[base + j + 9u  * sixteenth];
    GF x10 = a[base + j + 10u * sixteenth];
    GF x11 = a[base + j + 11u * sixteenth];
    GF x12 = a[base + j + 12u * sixteenth];
    GF x13 = a[base + j + 13u * sixteenth];
    GF x14 = a[base + j + 14u * sixteenth];
    GF x15 = a[base + j + 15u * sixteenth];
    const u32 off = lid << 4;
    scratch[off + 0u] = x0; scratch[off + 1u] = x1; scratch[off + 2u] = x2; scratch[off + 3u] = x3;
    scratch[off + 4u] = x4; scratch[off + 5u] = x5; scratch[off + 6u] = x6; scratch[off + 7u] = x7;
    scratch[off + 8u] = x8; scratch[off + 9u] = x9; scratch[off + 10u] = x10; scratch[off + 11u] = x11;
    scratch[off + 12u] = x12; scratch[off + 13u] = x13; scratch[off + 14u] = x14; scratch[off + 15u] = x15;
    barrier(CLK_LOCAL_MEM_FENCE);
    x0 = scratch[off + 0u]; x1 = scratch[off + 1u]; x2 = scratch[off + 2u]; x3 = scratch[off + 3u];
    x4 = scratch[off + 4u]; x5 = scratch[off + 5u]; x6 = scratch[off + 6u]; x7 = scratch[off + 7u];
    x8 = scratch[off + 8u]; x9 = scratch[off + 9u]; x10 = scratch[off + 10u]; x11 = scratch[off + 11u];
    x12 = scratch[off + 12u]; x13 = scratch[off + 13u]; x14 = scratch[off + 14u]; x15 = scratch[off + 15u];

    const GF s10  = gf_add(x0,  x8);
    const GF s18  = gf_mul(gf_sub(x0,  x8),  tw_stage1[tw_offset1 + j + 0u * sixteenth]);
    const GF s11  = gf_add(x1,  x9);
    const GF s19  = gf_mul(gf_sub(x1,  x9),  tw_stage1[tw_offset1 + j + 1u * sixteenth]);
    const GF s12  = gf_add(x2,  x10);
    const GF s1A  = gf_mul(gf_sub(x2,  x10), tw_stage1[tw_offset1 + j + 2u * sixteenth]);
    const GF s13  = gf_add(x3,  x11);
    const GF s1B  = gf_mul(gf_sub(x3,  x11), tw_stage1[tw_offset1 + j + 3u * sixteenth]);
    const GF s14  = gf_add(x4,  x12);
    const GF s1C  = gf_mul(gf_sub(x4,  x12), tw_stage1[tw_offset1 + j + 4u * sixteenth]);
    const GF s15  = gf_add(x5,  x13);
    const GF s1D  = gf_mul(gf_sub(x5,  x13), tw_stage1[tw_offset1 + j + 5u * sixteenth]);
    const GF s16  = gf_add(x6,  x14);
    const GF s1E  = gf_mul(gf_sub(x6,  x14), tw_stage1[tw_offset1 + j + 6u * sixteenth]);
    const GF s17  = gf_add(x7,  x15);
    const GF s1F  = gf_mul(gf_sub(x7,  x15), tw_stage1[tw_offset1 + j + 7u * sixteenth]);

    const GF s20  = gf_add(s10, s14);
    const GF s24  = gf_mul(gf_sub(s10, s14), tw_stage2[tw_offset2 + j + 0u * sixteenth]);
    const GF s21  = gf_add(s11, s15);
    const GF s25  = gf_mul(gf_sub(s11, s15), tw_stage2[tw_offset2 + j + 1u * sixteenth]);
    const GF s22  = gf_add(s12, s16);
    const GF s26  = gf_mul(gf_sub(s12, s16), tw_stage2[tw_offset2 + j + 2u * sixteenth]);
    const GF s23  = gf_add(s13, s17);
    const GF s27  = gf_mul(gf_sub(s13, s17), tw_stage2[tw_offset2 + j + 3u * sixteenth]);
    const GF s28  = gf_add(s18, s1C);
    const GF s2C  = gf_mul(gf_sub(s18, s1C), tw_stage2[tw_offset2 + j + 0u * sixteenth]);
    const GF s29  = gf_add(s19, s1D);
    const GF s2D  = gf_mul(gf_sub(s19, s1D), tw_stage2[tw_offset2 + j + 1u * sixteenth]);
    const GF s2A  = gf_add(s1A, s1E);
    const GF s2E  = gf_mul(gf_sub(s1A, s1E), tw_stage2[tw_offset2 + j + 2u * sixteenth]);
    const GF s2B  = gf_add(s1B, s1F);
    const GF s2F  = gf_mul(gf_sub(s1B, s1F), tw_stage2[tw_offset2 + j + 3u * sixteenth]);

    const GF s30  = gf_add(s20, s22);
    const GF s32  = gf_mul(gf_sub(s20, s22), tw_stage3[tw_offset3 + j + 0u * sixteenth]);
    const GF s31  = gf_add(s21, s23);
    const GF s33  = gf_mul(gf_sub(s21, s23), tw_stage3[tw_offset3 + j + 1u * sixteenth]);
    const GF s34  = gf_add(s24, s26);
    const GF s36  = gf_mul(gf_sub(s24, s26), tw_stage3[tw_offset3 + j + 0u * sixteenth]);
    const GF s35  = gf_add(s25, s27);
    const GF s37  = gf_mul(gf_sub(s25, s27), tw_stage3[tw_offset3 + j + 1u * sixteenth]);
    const GF s38  = gf_add(s28, s2A);
    const GF s3A  = gf_mul(gf_sub(s28, s2A), tw_stage3[tw_offset3 + j + 0u * sixteenth]);
    const GF s39  = gf_add(s29, s2B);
    const GF s3B  = gf_mul(gf_sub(s29, s2B), tw_stage3[tw_offset3 + j + 1u * sixteenth]);
    const GF s3C  = gf_add(s2C, s2E);
    const GF s3E  = gf_mul(gf_sub(s2C, s2E), tw_stage3[tw_offset3 + j + 0u * sixteenth]);
    const GF s3D  = gf_add(s2D, s2F);
    const GF s3F  = gf_mul(gf_sub(s2D, s2F), tw_stage3[tw_offset3 + j + 1u * sixteenth]);

    a[base + j + 0u  * sixteenth] = gf_add(s30, s31);
    a[base + j + 1u  * sixteenth] = gf_mul(gf_sub(s30, s31), tw_stage4[tw_offset4 + j]);
    a[base + j + 2u  * sixteenth] = gf_add(s32, s33);
    a[base + j + 3u  * sixteenth] = gf_mul(gf_sub(s32, s33), tw_stage4[tw_offset4 + j]);
    a[base + j + 4u  * sixteenth] = gf_add(s34, s35);
    a[base + j + 5u  * sixteenth] = gf_mul(gf_sub(s34, s35), tw_stage4[tw_offset4 + j]);
    a[base + j + 6u  * sixteenth] = gf_add(s36, s37);
    a[base + j + 7u  * sixteenth] = gf_mul(gf_sub(s36, s37), tw_stage4[tw_offset4 + j]);
    a[base + j + 8u  * sixteenth] = gf_add(s38, s39);
    a[base + j + 9u  * sixteenth] = gf_mul(gf_sub(s38, s39), tw_stage4[tw_offset4 + j]);
    a[base + j + 10u * sixteenth] = gf_add(s3A, s3B);
    a[base + j + 11u * sixteenth] = gf_mul(gf_sub(s3A, s3B), tw_stage4[tw_offset4 + j]);
    a[base + j + 12u * sixteenth] = gf_add(s3C, s3D);
    a[base + j + 13u * sixteenth] = gf_mul(gf_sub(s3C, s3D), tw_stage4[tw_offset4 + j]);
    a[base + j + 14u * sixteenth] = gf_add(s3E, s3F);
    a[base + j + 15u * sixteenth] = gf_mul(gf_sub(s3E, s3F), tw_stage4[tw_offset4 + j]);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_ntt_stage_dit_radix16(__global GF* a,
                                __global const GF* tw_stage1,
                                __global const GF* tw_stage2,
                                __global const GF* tw_stage3,
                                __global const GF* tw_stage4,
                                const u32 tw_offset1,
                                const u32 tw_offset2,
                                const u32 tw_offset3,
                                const u32 tw_offset4,
                                const u32 len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 half_len_local = len >> 1;
    const u32 block = gid / half_len_local;
    const u32 j = gid - block * half_len_local;
    const u32 base = block * (len << 3);
    __local GF scratch[1024];
    GF x0  = a[base + j + 0u  * half_len_local];
    GF x1  = a[base + j + 1u  * half_len_local];
    GF x2  = a[base + j + 2u  * half_len_local];
    GF x3  = a[base + j + 3u  * half_len_local];
    GF x4  = a[base + j + 4u  * half_len_local];
    GF x5  = a[base + j + 5u  * half_len_local];
    GF x6  = a[base + j + 6u  * half_len_local];
    GF x7  = a[base + j + 7u  * half_len_local];
    GF x8  = a[base + j + 8u  * half_len_local];
    GF x9  = a[base + j + 9u  * half_len_local];
    GF x10 = a[base + j + 10u * half_len_local];
    GF x11 = a[base + j + 11u * half_len_local];
    GF x12 = a[base + j + 12u * half_len_local];
    GF x13 = a[base + j + 13u * half_len_local];
    GF x14 = a[base + j + 14u * half_len_local];
    GF x15 = a[base + j + 15u * half_len_local];
    const u32 off = lid << 4;
    scratch[off + 0u] = x0; scratch[off + 1u] = x1; scratch[off + 2u] = x2; scratch[off + 3u] = x3;
    scratch[off + 4u] = x4; scratch[off + 5u] = x5; scratch[off + 6u] = x6; scratch[off + 7u] = x7;
    scratch[off + 8u] = x8; scratch[off + 9u] = x9; scratch[off + 10u] = x10; scratch[off + 11u] = x11;
    scratch[off + 12u] = x12; scratch[off + 13u] = x13; scratch[off + 14u] = x14; scratch[off + 15u] = x15;
    barrier(CLK_LOCAL_MEM_FENCE);
    x0 = scratch[off + 0u]; x1 = scratch[off + 1u]; x2 = scratch[off + 2u]; x3 = scratch[off + 3u];
    x4 = scratch[off + 4u]; x5 = scratch[off + 5u]; x6 = scratch[off + 6u]; x7 = scratch[off + 7u];
    x8 = scratch[off + 8u]; x9 = scratch[off + 9u]; x10 = scratch[off + 10u]; x11 = scratch[off + 11u];
    x12 = scratch[off + 12u]; x13 = scratch[off + 13u]; x14 = scratch[off + 14u]; x15 = scratch[off + 15u];

    const GF s10  = gf_add(x0,  gf_mul(x1,  tw_stage1[tw_offset1 + j]));
    const GF s11  = gf_sub(x0,  gf_mul(x1,  tw_stage1[tw_offset1 + j]));
    const GF s12  = gf_add(x2,  gf_mul(x3,  tw_stage1[tw_offset1 + j]));
    const GF s13  = gf_sub(x2,  gf_mul(x3,  tw_stage1[tw_offset1 + j]));
    const GF s14  = gf_add(x4,  gf_mul(x5,  tw_stage1[tw_offset1 + j]));
    const GF s15  = gf_sub(x4,  gf_mul(x5,  tw_stage1[tw_offset1 + j]));
    const GF s16  = gf_add(x6,  gf_mul(x7,  tw_stage1[tw_offset1 + j]));
    const GF s17  = gf_sub(x6,  gf_mul(x7,  tw_stage1[tw_offset1 + j]));
    const GF s18  = gf_add(x8,  gf_mul(x9,  tw_stage1[tw_offset1 + j]));
    const GF s19  = gf_sub(x8,  gf_mul(x9,  tw_stage1[tw_offset1 + j]));
    const GF s1A  = gf_add(x10, gf_mul(x11, tw_stage1[tw_offset1 + j]));
    const GF s1B  = gf_sub(x10, gf_mul(x11, tw_stage1[tw_offset1 + j]));
    const GF s1C  = gf_add(x12, gf_mul(x13, tw_stage1[tw_offset1 + j]));
    const GF s1D  = gf_sub(x12, gf_mul(x13, tw_stage1[tw_offset1 + j]));
    const GF s1E  = gf_add(x14, gf_mul(x15, tw_stage1[tw_offset1 + j]));
    const GF s1F  = gf_sub(x14, gf_mul(x15, tw_stage1[tw_offset1 + j]));

    const GF s20  = gf_add(s10, gf_mul(s12, tw_stage2[tw_offset2 + j]));
    const GF s22  = gf_sub(s10, gf_mul(s12, tw_stage2[tw_offset2 + j]));
    const GF s21  = gf_add(s11, gf_mul(s13, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s23  = gf_sub(s11, gf_mul(s13, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s24  = gf_add(s14, gf_mul(s16, tw_stage2[tw_offset2 + j]));
    const GF s26  = gf_sub(s14, gf_mul(s16, tw_stage2[tw_offset2 + j]));
    const GF s25  = gf_add(s15, gf_mul(s17, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s27  = gf_sub(s15, gf_mul(s17, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s28  = gf_add(s18, gf_mul(s1A, tw_stage2[tw_offset2 + j]));
    const GF s2A  = gf_sub(s18, gf_mul(s1A, tw_stage2[tw_offset2 + j]));
    const GF s29  = gf_add(s19, gf_mul(s1B, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s2B  = gf_sub(s19, gf_mul(s1B, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s2C  = gf_add(s1C, gf_mul(s1E, tw_stage2[tw_offset2 + j]));
    const GF s2E  = gf_sub(s1C, gf_mul(s1E, tw_stage2[tw_offset2 + j]));
    const GF s2D  = gf_add(s1D, gf_mul(s1F, tw_stage2[tw_offset2 + j + half_len_local]));
    const GF s2F  = gf_sub(s1D, gf_mul(s1F, tw_stage2[tw_offset2 + j + half_len_local]));

    const GF s30  = gf_add(s20, gf_mul(s24, tw_stage3[tw_offset3 + j]));
    const GF s34  = gf_sub(s20, gf_mul(s24, tw_stage3[tw_offset3 + j]));
    const GF s31  = gf_add(s21, gf_mul(s25, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s35  = gf_sub(s21, gf_mul(s25, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s32  = gf_add(s22, gf_mul(s26, tw_stage3[tw_offset3 + j]));
    const GF s36  = gf_sub(s22, gf_mul(s26, tw_stage3[tw_offset3 + j]));
    const GF s33  = gf_add(s23, gf_mul(s27, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s37  = gf_sub(s23, gf_mul(s27, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s38  = gf_add(s28, gf_mul(s2C, tw_stage3[tw_offset3 + j]));
    const GF s3C  = gf_sub(s28, gf_mul(s2C, tw_stage3[tw_offset3 + j]));
    const GF s39  = gf_add(s29, gf_mul(s2D, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s3D  = gf_sub(s29, gf_mul(s2D, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s3A  = gf_add(s2A, gf_mul(s2E, tw_stage3[tw_offset3 + j]));
    const GF s3E  = gf_sub(s2A, gf_mul(s2E, tw_stage3[tw_offset3 + j]));
    const GF s3B  = gf_add(s2B, gf_mul(s2F, tw_stage3[tw_offset3 + j + half_len_local]));
    const GF s3F  = gf_sub(s2B, gf_mul(s2F, tw_stage3[tw_offset3 + j + half_len_local]));

    a[base + j + 0u  * half_len_local] = gf_add(s30, gf_mul(s38, tw_stage4[tw_offset4 + j]));
    a[base + j + 1u  * half_len_local] = gf_sub(s30, gf_mul(s38, tw_stage4[tw_offset4 + j]));
    a[base + j + 2u  * half_len_local] = gf_add(s31, gf_mul(s39, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 3u  * half_len_local] = gf_sub(s31, gf_mul(s39, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 4u  * half_len_local] = gf_add(s32, gf_mul(s3A, tw_stage4[tw_offset4 + j]));
    a[base + j + 5u  * half_len_local] = gf_sub(s32, gf_mul(s3A, tw_stage4[tw_offset4 + j]));
    a[base + j + 6u  * half_len_local] = gf_add(s33, gf_mul(s3B, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 7u  * half_len_local] = gf_sub(s33, gf_mul(s3B, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 8u  * half_len_local] = gf_add(s34, gf_mul(s3C, tw_stage4[tw_offset4 + j]));
    a[base + j + 9u  * half_len_local] = gf_sub(s34, gf_mul(s3C, tw_stage4[tw_offset4 + j]));
    a[base + j + 10u * half_len_local] = gf_add(s35, gf_mul(s3D, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 11u * half_len_local] = gf_sub(s35, gf_mul(s3D, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 12u * half_len_local] = gf_add(s36, gf_mul(s3E, tw_stage4[tw_offset4 + j]));
    a[base + j + 13u * half_len_local] = gf_sub(s36, gf_mul(s3E, tw_stage4[tw_offset4 + j]));
    a[base + j + 14u * half_len_local] = gf_add(s37, gf_mul(s3F, tw_stage4[tw_offset4 + j + half_len_local]));
    a[base + j + 15u * half_len_local] = gf_sub(s37, gf_mul(s3F, tw_stage4[tw_offset4 + j + half_len_local]));
}

__kernel void gf61_pointwise_sqr(__global GF* a, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) a[gid] = gf_sqr(a[gid]);
}

__kernel void gf61_mul_small_digits(__global u64* digits, const u32 k, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) digits[gid] *= (u64)k;
}


inline void local_stage_dif_pow2(__local GF* x,
                                 __global const GF* twiddles,
                                 const u32 chunk,
                                 const u32 len,
                                 const u32 lid,
                                 const u32 lsize) {
    const u32 half_len_local = len >> 1;
    const u32 tw_offset = half_len_local - 1u;
    const u32 butterflies = chunk >> 1;
    for (u32 t = lid; t < butterflies; t += lsize) {
        const u32 block = t / half_len_local;
        const u32 j = t - block * half_len_local;
        const u32 i0 = block * len + j;
        const u32 i1 = i0 + half_len_local;
        const GF u = x[i0];
        const GF v = x[i1];
        x[i0] = gf_add(u, v);
        x[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
    }
}

inline void local_stage_dit_pow2(__local GF* x,
                                 __global const GF* twiddles,
                                 const u32 chunk,
                                 const u32 len,
                                 const u32 lid,
                                 const u32 lsize) {
    const u32 half_len_local = len >> 1;
    const u32 tw_offset = half_len_local - 1u;
    const u32 butterflies = chunk >> 1;
    for (u32 t = lid; t < butterflies; t += lsize) {
        const u32 block = t / half_len_local;
        const u32 j = t - block * half_len_local;
        const u32 i0 = block * len + j;
        const u32 i1 = i0 + half_len_local;
        const GF u = x[i0];
        const GF v = gf_mul(x[i1], twiddles[tw_offset + j]);
        x[i0] = gf_add(u, v);
        x[i1] = gf_sub(u, v);
    }
}

#define DECL_GF61_CENTER_FUSED(NAME, CHUNK, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, __global const GF* tw_fwd, __global const GF* tw_inv) { \
    const u32 lid = (u32)get_local_id(0); \
    const u32 group = (u32)get_group_id(0); \
    const u32 base = group * (CHUNK); \
    __local GF x[(CHUNK)]; \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (u32 len = (CHUNK); len >= 2u; len >>= 1) { \
        local_stage_dif_pow2(x, tw_fwd, (CHUNK), len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (len == 2u) break; \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = gf_sqr(x[i]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (u32 len = 2u; len <= (CHUNK); len <<= 1) { \
        local_stage_dit_pow2(x, tw_inv, (CHUNK), len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (len == (CHUNK)) break; \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; \
}

DECL_GF61_CENTER_FUSED(gf61_center_fused_16, 16u, 8u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_64, 64u, 16u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_256, 256u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_512, 512u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_1024, 1024u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_2048, 2048u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_4096, 4096u, 64u)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_center_fused_256_explicit(__global GF* a, __global const GF* tw_fwd, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 256u;
    __local GF x[256];
    for (u32 i = lid; i < 256u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 256u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 128u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 64u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 32u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 16u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 8u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 4u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 256u, 2u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 256u; i += 64u) x[i] = gf_sqr(x[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 2u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 4u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 8u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 16u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 32u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 64u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 128u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 256u, 256u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 256u; i += 64u) a[base + i] = x[i];
}

#define DECL_GF61_FORWARD_BRIDGE(NAME, CHUNK, WG, STOP_CHUNK) __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(__global GF* a, __global const GF* tw_fwd) {     const u32 lid = (u32)get_local_id(0);     const u32 group = (u32)get_group_id(0);     const u32 base = group * (CHUNK);     __local GF x[(CHUNK)];     for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i];     barrier(CLK_LOCAL_MEM_FENCE);     for (u32 len = (CHUNK); len > (STOP_CHUNK); len >>= 1) {         local_stage_dif_pow2(x, tw_fwd, (CHUNK), len, lid, (WG));         barrier(CLK_LOCAL_MEM_FENCE);     }     for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; }

#define DECL_GF61_INVERSE_BRIDGE(NAME, CHUNK, WG, START_CHUNK) __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(__global GF* a, __global const GF* tw_inv) {     const u32 lid = (u32)get_local_id(0);     const u32 group = (u32)get_group_id(0);     const u32 base = group * (CHUNK);     __local GF x[(CHUNK)];     for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i];     barrier(CLK_LOCAL_MEM_FENCE);     for (u32 len = ((START_CHUNK) << 1); len <= (CHUNK); len <<= 1) {         local_stage_dit_pow2(x, tw_inv, (CHUNK), len, lid, (WG));         barrier(CLK_LOCAL_MEM_FENCE);         if (len == (CHUNK)) break;     }     for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; }

DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_64_to_16, 64u, 16u, 16u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_16_to_64, 64u, 16u, 16u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_256_to_64, 256u, 64u, 64u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_64_to_256, 256u, 64u, 64u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_1024_to_512, 1024u, 64u, 512u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_512_to_1024, 1024u, 64u, 512u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_1024_to_256, 1024u, 64u, 256u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_256_to_1024, 1024u, 64u, 256u)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_forward_ext_1024_to_256_explicit(__global GF* a, __global const GF* tw_fwd) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];
    for (u32 i = lid; i < 1024u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 1024u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 1024u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_inverse_ext_256_to_1024_explicit(__global GF* a, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];
    for (u32 i = lid; i < 1024u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 1024u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 1024u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_forward_ext_2048_to_256_explicit(__global GF* a, __global const GF* tw_fwd) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 2048u;
    __local GF x[2048u];
    for (u32 i = lid; i < 2048u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 2048u, 2048u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 2048u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 2048u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 2048u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_inverse_ext_256_to_2048_explicit(__global GF* a, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 2048u;
    __local GF x[2048u];
    for (u32 i = lid; i < 2048u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 2048u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 2048u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 2048u, 2048u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 2048u; i += 64u) a[base + i] = x[i];
}

inline u64 digit_mask(const u32 w) {
    return (w >= 64u) ? ~0ul : ((((u64)1) << w) - 1ul);
}

inline u8 summary_suffix_all_max(const u64 value_lo,
                                 const u32 bits,
                                 const u8 mode,
                                 const u32 start) {
    if (start >= bits) return 1u;
    if (bits <= 64u) {
        const u32 rem = bits - start;
        const u64 want = digit_mask(rem);
        return (((value_lo >> start) & want) == want) ? 1u : 0u;
    }
    if (start < 64u) {
        const u32 rem_lo = 64u - start;
        const u64 want = digit_mask(rem_lo);
        if (((value_lo >> start) & want) != want) return 0u;
        return (mode == 1u) ? 1u : 0u;
    }
    return (mode == 1u) ? 1u : 0u;
}

inline u64 carry_summary_overflow(const u64 incoming,
                                  const u64 value_lo,
                                  const u32 bits,
                                  const u64 threshold,
                                  const u8 mode) {
    u64 overflow = 0ul;
    if (mode == 0u) {
        if (bits < 64u) {
            const u64 low_mask = digit_mask(bits);
            overflow = ((incoming & low_mask) + value_lo) >> bits;
            overflow += (incoming >> bits);
        } else {
            const u64 sum = incoming + value_lo;
            overflow = (sum < incoming) ? 1ul : 0ul;
        }
    } else if (mode == 1u) {
        overflow = (incoming >= threshold) ? 1ul : 0ul;
    }
    return overflow;
}


__kernel void gf61_carry_block_local(__global u64* digits,
                                     __global const u8* widths,
                                     __global u64* block_carry,
                                     __global u64* block_value_lo,
                                     __global u32* block_bits,
                                     __global u64* block_threshold,
                                     __global u8* block_mode,
                                     __global u64* seg_value_lo,
                                     __global u32* seg_bits,
                                     __global u64* seg_threshold,
                                     __global u8* seg_mode,
                                     const u32 n,
                                     const u32 block_size,
                                     const u32 items_per_worker,
                                     __local u64* ldigits,
                                     __local u64* lseg_carry,
                                     __local u64* lseg_value_lo,
                                     __local u32* lseg_bits,
                                     __local u64* lseg_threshold,
                                     __local u64* lseg_incoming,
                                     __local u8* lseg_mode) {
    const u32 group = (u32)get_group_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 local_size = (u32)get_local_size(0);
    const u32 base = group * block_size;
    const u32 start = base + lid * items_per_worker;

    u32 count = block_size;
    if (base + count > n) count = n - base;

    const u32 seg_base = lid * items_per_worker;
    u32 seg_count = 0u;
    if (seg_base < count) {
        seg_count = count - seg_base;
        if (seg_count > items_per_worker) seg_count = items_per_worker;
    }

    const u32 off = seg_base;

    if (items_per_worker == 4u && seg_count == 4u) {
        const ulong4 vd = vload4(0, digits + start);
        ldigits[off + 0u] = vd.s0;
        ldigits[off + 1u] = vd.s1;
        ldigits[off + 2u] = vd.s2;
        ldigits[off + 3u] = vd.s3;
    } else {
        for (u32 t = 0; t < seg_count; ++t) {
            ldigits[off + t] = digits[start + t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    u64 carry = 0ul;
    u64 lo = 0ul;
    u32 bitpos = 0u;
    u8 hi_all_max = 1u;

    for (u32 t = 0; t < seg_count; ++t) {
        const u32 idx = start + t;
        const u32 w = (u32)widths[idx];
        const u64 mask = digit_mask(w);
        const u64 total = ldigits[off + t] + carry;
        const u64 d = total & mask;
        ldigits[off + t] = d;
        carry = total >> w;

        if (bitpos < 64u) {
            const u32 take = min(w, 64u - bitpos);
            const u64 low_mask = digit_mask(take);
            lo |= (d & low_mask) << bitpos;
            if (w > take) {
                const u32 hiw = w - take;
                const u64 hi_mask = digit_mask(hiw);
                if ((d >> take) != hi_mask) hi_all_max = 0u;
            }
        } else {
            if (d != mask) hi_all_max = 0u;
        }
        bitpos += w;
    }

    lseg_carry[lid] = carry;
    lseg_value_lo[lid] = lo;
    lseg_bits[lid] = bitpos;
    if (bitpos <= 64u) {
        lseg_threshold[lid] = 0ul;
        lseg_mode[lid] = 0u;
    } else if (hi_all_max && lo != 0ul) {
        lseg_threshold[lid] = (~lo) + 1ul;
        lseg_mode[lid] = 1u;
    } else {
        lseg_threshold[lid] = 0ul;
        lseg_mode[lid] = 2u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        u64 incoming = 0ul;
        for (u32 s = 0; s < local_size; ++s) {
            lseg_incoming[s] = incoming;
            incoming = lseg_carry[s] + carry_summary_overflow(incoming,
                                                              lseg_value_lo[s],
                                                              lseg_bits[s],
                                                              lseg_threshold[s],
                                                              lseg_mode[s]);
        }
        block_carry[group] = incoming;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    u64 incoming = lseg_incoming[lid];
    u64 final_lo = 0ul;
    u32 final_bitpos = 0u;
    u8 final_hi_all_max = 1u;
    for (u32 t = 0; t < seg_count; ++t) {
        const u32 idx = start + t;
        const u32 w = (u32)widths[idx];
        const u64 mask = digit_mask(w);
        const u64 total = ldigits[off + t] + incoming;
        const u64 d = total & mask;
        ldigits[off + t] = d;
        incoming = total >> w;

        if (final_bitpos < 64u) {
            const u32 take = min(w, 64u - final_bitpos);
            const u64 low_mask = digit_mask(take);
            final_lo |= (d & low_mask) << final_bitpos;
            if (w > take) {
                const u32 hiw = w - take;
                const u64 hi_mask = digit_mask(hiw);
                if ((d >> take) != hi_mask) final_hi_all_max = 0u;
            }
        } else {
            if (d != mask) final_hi_all_max = 0u;
        }
        final_bitpos += w;
    }

    lseg_value_lo[lid] = final_lo;
    lseg_bits[lid] = final_bitpos;
    if (final_bitpos <= 64u) {
        lseg_threshold[lid] = 0ul;
        lseg_mode[lid] = 0u;
    } else if (final_hi_all_max && final_lo != 0ul) {
        lseg_threshold[lid] = (~final_lo) + 1ul;
        lseg_mode[lid] = 1u;
    } else {
        lseg_threshold[lid] = 0ul;
        lseg_mode[lid] = 2u;
    }

    const u32 seg_idx = group * local_size + lid;
    seg_value_lo[seg_idx] = final_lo;
    seg_bits[seg_idx] = final_bitpos;
    seg_threshold[seg_idx] = lseg_threshold[lid];
    seg_mode[seg_idx] = lseg_mode[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        u64 block_lo = 0ul;
        u32 block_bitpos = 0u;
        u8 block_hi_all_max = 1u;

        for (u32 s = 0u; s < local_size; ++s) {
            const u32 bits = lseg_bits[s];
            if (bits == 0u) continue;

            const u64 seg_lo = lseg_value_lo[s];
            const u8 seg_mode = lseg_mode[s];

            if (block_bitpos < 64u) {
                const u32 take = min(bits, 64u - block_bitpos);
                const u64 low_mask = digit_mask(take);
                block_lo |= (seg_lo & low_mask) << block_bitpos;
                if (bits > take) {
                    if (!summary_suffix_all_max(seg_lo, bits, seg_mode, take)) block_hi_all_max = 0u;
                }
            } else {
                if (!summary_suffix_all_max(seg_lo, bits, seg_mode, 0u)) block_hi_all_max = 0u;
            }
            block_bitpos += bits;
        }

        block_value_lo[group] = block_lo;
        block_bits[group] = block_bitpos;
        if (block_bitpos <= 64u) {
            block_threshold[group] = 0ul;
            block_mode[group] = 0u;
        } else if (block_hi_all_max && block_lo != 0ul) {
            block_threshold[group] = (~block_lo) + 1ul;
            block_mode[group] = 1u;
        } else {
            block_threshold[group] = 0ul;
            block_mode[group] = 2u;
        }
    }

    if (items_per_worker == 4u && seg_count == 4u) {
        const ulong4 vd = (ulong4)(ldigits[off + 0u], ldigits[off + 1u], ldigits[off + 2u], ldigits[off + 3u]);
        vstore4(vd, 0, digits + start);
    } else {
        for (u32 t = 0; t < seg_count; ++t) {
            digits[start + t] = ldigits[off + t];
        }
    }
}

__kernel void gf61_carry_block_prefix_serial(__global const u64* block_carry,
                                             __global const u64* block_value_lo,
                                             __global const u32* block_bits,
                                             __global const u64* block_threshold,
                                             __global const u8* block_mode,
                                             __global u64* block_incoming,
                                             __global u64* final_carry,
                                             const u32 num_blocks) {
    if (get_global_id(0) != 0) return;

    u64 incoming = 0ul;
    for (u32 b = 0; b < num_blocks; ++b) {
        block_incoming[b] = incoming;

        incoming = block_carry[b] + carry_summary_overflow(incoming,
                                                           block_value_lo[b],
                                                           block_bits[b],
                                                           block_threshold[b],
                                                           block_mode[b]);
    }
    final_carry[0] = incoming;
}

__kernel void gf61_carry_block_prefix_chunked64(__global const u64* block_carry,
                                                __global const u64* block_value_lo,
                                                __global const u32* block_bits,
                                                __global const u64* block_threshold,
                                                __global const u8* block_mode,
                                                __global u64* block_incoming,
                                                __global u64* final_carry,
                                                const u32 num_blocks) {
    const u32 lid = (u32)get_local_id(0);
    if (get_group_id(0) != 0u) return;
    if (get_local_size(0) != 64u) return;

    const u32 start = (num_blocks * lid) >> 6;
    const u32 end = (num_blocks * (lid + 1u)) >> 6;

    __local u64 lchunk_base[64];
    __local u64 lchunk_threshold[64];
    __local u64 lchunk_incoming[64];
    __local u8 lchunk_has_step[64];
    __local u8 lchunk_has_data[64];

    u64 base = 0ul;
    u64 threshold = 0ul;
    u8 has_step = 0u;
    u8 has_data = 0u;

    for (u32 b = start; b < end; ++b) {
        const u64 c = block_carry[b];
        const u32 bits = block_bits[b];
        const u8 mode = block_mode[b];
        u8 step = 0u;
        u64 step_threshold = 0ul;
        if (mode == 1u) {
            step = 1u;
            step_threshold = block_threshold[b];
        } else if (mode == 0u && bits >= 64u) {
            step = 1u;
            step_threshold = (~block_value_lo[b]) + 1ul;
        }

        if (!has_data) {
            has_data = 1u;
            if (step) {
                base = c;
                threshold = step_threshold;
                has_step = 1u;
            } else {
                base = c;
                has_step = 0u;
            }
            continue;
        }

        if (!step) {
            base = c;
            has_step = 0u;
        } else if (!has_step) {
            base = c + ((base >= step_threshold) ? 1ul : 0ul);
            has_step = 0u;
        } else {
            const u64 low = base;
            const u64 high = base + 1ul;
            if (low >= step_threshold) {
                base = c + 1ul;
                has_step = 0u;
            } else if (high < step_threshold) {
                base = c;
                has_step = 0u;
            } else {
                base = c;
                has_step = 1u;
            }
        }
    }

    lchunk_base[lid] = base;
    lchunk_threshold[lid] = threshold;
    lchunk_has_step[lid] = has_step;
    lchunk_has_data[lid] = has_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        u64 incoming = 0ul;
        for (u32 s = 0u; s < 64u; ++s) {
            lchunk_incoming[s] = incoming;
            if (!lchunk_has_data[s]) continue;
            incoming = lchunk_base[s] + ((lchunk_has_step[s] && incoming >= lchunk_threshold[s]) ? 1ul : 0ul);
        }
        final_carry[0] = incoming;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    u64 incoming = lchunk_incoming[lid];
    for (u32 b = start; b < end; ++b) {
        block_incoming[b] = incoming;
        incoming = block_carry[b] + carry_summary_overflow(incoming,
                                                           block_value_lo[b],
                                                           block_bits[b],
                                                           block_threshold[b],
                                                           block_mode[b]);
    }
}

__kernel void gf61_carry_block_apply_incoming(__global u64* digits,
                                              __global const u8* widths,
                                              __global const u64* block_incoming,
                                              __global const u64* seg_value_lo,
                                              __global const u32* seg_bits,
                                              __global const u64* seg_threshold,
                                              __global const u8* seg_mode,
                                              const u32 n,
                                              const u32 block_size,
                                              const u32 items_per_worker,
                                              __local u64* lseg_incoming) {
    const u32 group = (u32)get_group_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 local_size = (u32)get_local_size(0);
    const u32 num_blocks = (n + block_size - 1u) / block_size;
    if (group >= num_blocks) return;

    const u64 block_inc = block_incoming[group];
    if (block_inc == 0ul) return;

    const u32 base = group * block_size;
    u32 count = block_size;
    if (base + count > n) count = n - base;

    const u32 seg_base = lid * items_per_worker;
    u32 seg_count = 0u;
    if (seg_base < count) {
        seg_count = count - seg_base;
        if (seg_count > items_per_worker) seg_count = items_per_worker;
    }

    if (lid == 0u) {
        u64 incoming = block_inc;
        for (u32 s = 0u; s < local_size; ++s) {
            const u32 sidx = group * local_size + s;
            lseg_incoming[s] = incoming;
            const u32 bits = seg_bits[sidx];
            if (bits != 0u) {
                incoming = carry_summary_overflow(incoming,
                                                  seg_value_lo[sidx],
                                                  bits,
                                                  seg_threshold[sidx],
                                                  seg_mode[sidx]);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (seg_count == 0u) return;

    u64 incoming = lseg_incoming[lid];
    const u32 start = base + seg_base;
    if (incoming != 0ul) {
        if (items_per_worker == 4u && seg_count == 4u) {
            const uchar4 wv = vload4(0, widths + start);
            ulong4 dv = vload4(0, digits + start);
            u64 total = dv.s0 + incoming;
            dv.s0 = total & digit_mask((u32)wv.s0);
            incoming = total >> (u32)wv.s0;
            total = dv.s1 + incoming;
            dv.s1 = total & digit_mask((u32)wv.s1);
            incoming = total >> (u32)wv.s1;
            total = dv.s2 + incoming;
            dv.s2 = total & digit_mask((u32)wv.s2);
            incoming = total >> (u32)wv.s2;
            total = dv.s3 + incoming;
            dv.s3 = total & digit_mask((u32)wv.s3);
            vstore4(dv, 0, digits + start);
        } else {
            for (u32 t = 0; t < seg_count; ++t) {
                const u32 idx = start + t;
                const u32 w = (u32)widths[idx];
                const u64 mask = digit_mask(w);
                const u64 total = digits[idx] + incoming;
                digits[idx] = total & mask;
                incoming = total >> w;
                if (incoming == 0ul) break;
            }
        }
    }
}

__kernel void gf61_carry_block_apply_incoming_serial(__global u64* digits,
                                                     __global const u8* widths,
                                                     __global const u64* block_incoming,
                                                     const u32 n,
                                                     const u32 block_size) {
    const u32 b = (u32)get_global_id(0);
    const u32 num_blocks = (n + block_size - 1u) / block_size;
    if (b >= num_blocks) return;

    const u32 base = b * block_size;
    u32 count = block_size;
    if (base + count > n) count = n - base;

    u64 incoming = block_incoming[b];
    if (incoming != 0ul) {
        for (u32 i = 0; i < count; ++i) {
            const u32 idx = base + i;
            const u32 w = (u32)widths[idx];
            const u64 mask = digit_mask(w);
            const u64 total = digits[idx] + incoming;
            digits[idx] = total & mask;
            incoming = total >> w;
            if (incoming == 0ul) break;
        }
    }
}

__kernel void gf61_carry_final_wrap_serial(__global u64* digits,
                                           __global const u8* widths,
                                           __global const u64* final_carry,
                                           const u32 n) {
    if (get_global_id(0) != 0) return;

    u64 incoming = final_carry[0];
    while (incoming != 0ul) {
        for (u32 i = 0; i < n; ++i) {
            const u32 w = (u32)widths[i];
            const u64 mask = digit_mask(w);
            const u64 total = digits[i] + incoming;
            digits[i] = total & mask;
            incoming = total >> w;
            if (incoming == 0ul) break;
        }
    }
}
