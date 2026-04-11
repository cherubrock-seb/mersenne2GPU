#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include <csignal>

namespace {
std::atomic<bool> g_stop_requested{false};

void handle_sigint(int) { g_stop_requested.store(true, std::memory_order_relaxed); }

struct InterruptedRun : public std::exception {
    const char* what() const noexcept override { return "run interrupted"; }
};
} // namespace

namespace gf61 {

constexpr std::uint64_t P = (std::uint64_t(1) << 61) - 1;
constexpr std::uint64_t ROOT_H0 = 264036120304204ULL;
constexpr std::uint64_t ROOT_H1 = 4677669021635377ULL;
constexpr unsigned ROOT_ORDER_LOG2 = 62;

struct Elem {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    Elem() = default;
    Elem(std::uint64_t aa, std::uint64_t bb) : a(aa), b(bb) {}
};

inline std::uint64_t add_mod(std::uint64_t x, std::uint64_t y) {
    std::uint64_t s = x + y;
    s = (s & P) + (s >> 61);
    if (s >= P) s -= P;
    return s;
}

inline std::uint64_t sub_mod(std::uint64_t x, std::uint64_t y) {
    return (x >= y) ? (x - y) : (P - (y - x));
}

inline std::uint64_t reduce_u128(__uint128_t x) {
    std::uint64_t lo = static_cast<std::uint64_t>(x) & P;
    std::uint64_t hi = static_cast<std::uint64_t>(x >> 61);
    std::uint64_t r = lo + hi;
    r = (r & P) + (r >> 61);
    if (r >= P) r -= P;
    return r;
}

inline std::uint64_t mul_mod(std::uint64_t x, std::uint64_t y) {
    return reduce_u128(static_cast<__uint128_t>(x) * static_cast<__uint128_t>(y));
}

std::uint64_t pow_mod(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    while (e) {
        if (e & 1ULL) r = mul_mod(r, a);
        a = mul_mod(a, a);
        e >>= 1U;
    }
    return r;
}

inline Elem add(const Elem& x, const Elem& y) {
    return Elem(add_mod(x.a, y.a), add_mod(x.b, y.b));
}

inline Elem sub(const Elem& x, const Elem& y) {
    return Elem(sub_mod(x.a, y.a), sub_mod(x.b, y.b));
}

inline Elem mul(const Elem& x, const Elem& y) {
    const std::uint64_t ac = mul_mod(x.a, y.a);
    const std::uint64_t bd = mul_mod(x.b, y.b);
    const std::uint64_t ad = mul_mod(x.a, y.b);
    const std::uint64_t bc = mul_mod(x.b, y.a);
    return Elem(sub_mod(ac, bd), add_mod(ad, bc));
}

inline Elem sqr(const Elem& x) {
    const std::uint64_t aa = mul_mod(x.a, x.a);
    const std::uint64_t bb = mul_mod(x.b, x.b);
    const std::uint64_t ab = mul_mod(x.a, x.b);
    return Elem(sub_mod(aa, bb), add_mod(ab, ab));
}

inline Elem conj(const Elem& x) {
    return Elem(x.a, x.b == 0 ? 0 : (P - x.b));
}

Elem inv(const Elem& x) {
    const std::uint64_t denom = add_mod(mul_mod(x.a, x.a), mul_mod(x.b, x.b));
    const std::uint64_t inv_denom = pow_mod(denom, P - 2);
    const Elem c = conj(x);
    return Elem(mul_mod(c.a, inv_denom), mul_mod(c.b, inv_denom));
}

Elem primitive_root_pow2(std::size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) throw std::runtime_error("NTT size must be a power of two");
    unsigned k = 0;
    while ((std::size_t(1) << k) < n) ++k;
    if (k > ROOT_ORDER_LOG2) throw std::runtime_error("NTT size exceeds 2-adic order in GF(M61^2)");
    Elem r(ROOT_H0, ROOT_H1);
    for (unsigned i = 0; i < ROOT_ORDER_LOG2 - k; ++i) r = sqr(r);
    return r;
}

} // namespace gf61

namespace ibdwt {

struct Layout {
    std::uint32_t p = 0;
    unsigned ln = 0;
    std::size_t n = 0;
    std::vector<std::uint8_t> digit_width;
    std::vector<std::uint8_t> shifts;
};

inline unsigned transform_size_log2(std::uint32_t p) {
    unsigned ln = 2;
    std::uint32_t w = 0;
    do {
        ++ln;
        w = p >> ln;
    } while (ln + 2u * (w + 1u) >= 61u);
    return ln;
}

inline std::uint8_t log2_root_two(std::size_t n) {
    return static_cast<std::uint8_t>(((std::uint64_t(1) << 60) / n) % 61u);
}

static std::uint32_t bit_reverse(std::uint32_t x, unsigned bits) {
    std::uint32_t r = 0;
    for (unsigned i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

inline Layout make_layout(std::uint32_t p) {
    Layout out;
    out.p = p;
    out.ln = transform_size_log2(p);
    out.n = std::size_t(1) << out.ln;
    out.digit_width.assign(out.n, 0);
    out.shifts.assign(out.n, 0);

    const std::uint8_t lr2 = log2_root_two(out.n);
    std::uint32_t prev_ceil = 0;
    for (std::size_t j = 0; j <= out.n; ++j) {
        const std::uint64_t qj = std::uint64_t(p) * std::uint64_t(j);
        const std::uint32_t ceil_qj_n = (j == 0) ? 0u : static_cast<std::uint32_t>(((qj - 1u) >> out.ln) + 1u);
        if (j > 0) {
            out.digit_width[j - 1] = static_cast<std::uint8_t>(ceil_qj_n - prev_ceil);
            if (j < out.n) {
                const std::uint32_t r = static_cast<std::uint32_t>(qj & (out.n - 1u));
                out.shifts[j] = static_cast<std::uint8_t>((std::uint32_t(lr2) * static_cast<std::uint32_t>(out.n - r)) % 61u);
            }
        }
        prev_ceil = ceil_qj_n;
    }
    out.shifts[0] = 0;
    return out;
}

static std::vector<std::uint64_t> from_small(std::uint64_t value, const Layout& layout) {
    std::vector<std::uint64_t> digits(layout.n, 0);
    for (std::size_t i = 0; i < layout.n; ++i) {
        const std::uint8_t w = layout.digit_width[i];
        const std::uint64_t mask = (w == 64) ? ~0ULL : ((std::uint64_t(1) << w) - 1ULL);
        digits[i] = value & mask;
        value >>= w;
    }
    if (value != 0) throw std::runtime_error("from_small overflow for layout");
    return digits;
}

static bool all_max_digits(const std::vector<std::uint64_t>& digits, const Layout& layout) {
    for (std::size_t i = 0; i < layout.n; ++i) {
        const std::uint8_t w = layout.digit_width[i];
        const std::uint64_t mask = (std::uint64_t(1) << w) - 1ULL;
        if (digits[i] != mask) return false;
    }
    return true;
}

static void canonicalize_zero(std::vector<std::uint64_t>& digits, const Layout& layout) {
    if (all_max_digits(digits, layout)) std::fill(digits.begin(), digits.end(), 0ULL);
}

static bool equals_small(const std::vector<std::uint64_t>& digits, const Layout& layout, std::uint64_t value) {
    return digits == from_small(value, layout);
}

} // namespace ibdwt

namespace clwrap {

inline void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

static std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("unable to open kernel file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct DeviceInfo {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string name;
    std::size_t max_work_group_size = 0;
    cl_ulong local_mem_size = 0;
};

std::vector<DeviceInfo> list_devices() {
    cl_uint nplat = 0;
    check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> plats(nplat);
    check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    std::vector<DeviceInfo> out;
    for (cl_platform_id plat : plats) {
        cl_uint ndev = 0;
        cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
        if (err == CL_DEVICE_NOT_FOUND) continue;
        check(err, "clGetDeviceIDs(count)");
        std::vector<cl_device_id> devs(ndev);
        check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr), "clGetDeviceIDs(list)");
        for (cl_device_id dev : devs) {
            size_t sz = 0;
            check(clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &sz), "clGetDeviceInfo(name size)");
            std::string name(sz, '\0');
            check(clGetDeviceInfo(dev, CL_DEVICE_NAME, sz, name.data(), nullptr), "clGetDeviceInfo(name)");
            while (!name.empty() && (name.back() == '\0' || name.back() == '\n' || name.back() == '\r')) name.pop_back();
            size_t max_wg = 0;
            cl_ulong local_mem = 0;
            check(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr), "clGetDeviceInfo(max wg)");
            check(clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, nullptr), "clGetDeviceInfo(local mem)");
            out.push_back(DeviceInfo{plat, dev, name, max_wg, local_mem});
        }
    }
    if (out.empty()) throw std::runtime_error("no OpenCL devices found");
    return out;
}

struct StageInfo {
    std::uint32_t len = 0;
    std::uint32_t half_len = 0;
    std::uint32_t offset = 0;
};

struct ProfileEntry {
    double ms = 0.0;
    std::uint64_t launches = 0;
};

struct GpuPrp {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel k_weight_first_stage_dif = nullptr;
    cl_kernel k_weight_first_stage_dif_wg16 = nullptr;
    cl_kernel k_weight_first_stage_dif_wg64 = nullptr;
    cl_kernel k_ntt_stage_dif = nullptr;
    cl_kernel k_ntt_stage_dit = nullptr;
    cl_kernel k_ntt_stage_dif_radix4 = nullptr;
    cl_kernel k_ntt_stage_dit_radix4 = nullptr;
    cl_kernel k_ntt_stage_dif_radix8 = nullptr;
    cl_kernel k_ntt_stage_dit_radix8 = nullptr;
    cl_kernel k_ntt_stage_dif_radix16 = nullptr;
    cl_kernel k_ntt_stage_dit_radix16 = nullptr;
    cl_kernel k_last_stage_dit_unweight = nullptr;
    cl_kernel k_last_stage_dit_unweight_wg16 = nullptr;
    cl_kernel k_last_stage_dit_unweight_wg64 = nullptr;
    cl_kernel k_pointwise_sqr = nullptr;
    cl_kernel k_center_fused_16 = nullptr;
    cl_kernel k_center_fused_64 = nullptr;
    cl_kernel k_center_fused_256 = nullptr;
    cl_kernel k_center_fused_256_explicit = nullptr;
    cl_kernel k_center_fused_512 = nullptr;
    cl_kernel k_center_fused_1024 = nullptr;
    cl_kernel k_center_fused_2048 = nullptr;
    cl_kernel k_center_fused_4096 = nullptr;
    cl_kernel k_forward_bridge_64_to_16 = nullptr;
    cl_kernel k_inverse_bridge_16_to_64 = nullptr;
    cl_kernel k_forward_bridge_256_to_64 = nullptr;
    cl_kernel k_inverse_bridge_64_to_256 = nullptr;
    cl_kernel k_forward_bridge_1024_to_512 = nullptr;
    cl_kernel k_inverse_bridge_512_to_1024 = nullptr;
    cl_kernel k_forward_bridge_1024_to_256 = nullptr;
    cl_kernel k_inverse_bridge_256_to_1024 = nullptr;
    cl_kernel k_forward_ext_1024_to_256_explicit = nullptr;
    cl_kernel k_inverse_ext_256_to_1024_explicit = nullptr;
    cl_kernel k_forward_ext_2048_to_256_explicit = nullptr;
    cl_kernel k_inverse_ext_256_to_2048_explicit = nullptr;
    cl_kernel k_mul_small = nullptr;
    cl_kernel k_carry_block_local = nullptr;
    cl_kernel k_carry_block_prefix = nullptr;
    cl_kernel k_carry_block_prefix_chunked64 = nullptr;
    cl_kernel k_carry_block_apply_incoming = nullptr;
    cl_kernel k_carry_block_apply_incoming_serial = nullptr;
    cl_kernel k_carry_final_wrap = nullptr;

    cl_mem bufDigits = nullptr;
    cl_mem bufField = nullptr;
    cl_mem bufShift = nullptr;
    cl_mem bufWidth = nullptr;
    cl_mem bufTwFwd = nullptr;
    cl_mem bufTwInv = nullptr;
    cl_mem bufBlockCarry = nullptr;
    cl_mem bufBlockValueLo = nullptr;
    cl_mem bufBlockBits = nullptr;
    cl_mem bufBlockThreshold = nullptr;
    cl_mem bufBlockMode = nullptr;
    cl_mem bufBlockIncoming = nullptr;
    cl_mem bufFinalCarry = nullptr;
    cl_mem bufSegValueLo = nullptr;
    cl_mem bufSegBits = nullptr;
    cl_mem bufSegThreshold = nullptr;
    cl_mem bufSegMode = nullptr;
    cl_uint carry_buffer_blocks = 0;
    cl_uint carry_buffer_segments = 0;

    std::size_t n = 0;
    cl_uint log_n = 0;
    std::size_t max_work_group_size = 0;
    cl_ulong local_mem_size = 0;
    bool profile_kernels = false;
    std::vector<StageInfo> stages;
    std::vector<std::pair<std::string, cl_event>> pending_profile_events;
    std::map<std::string, ProfileEntry> profile_totals;
    std::vector<std::string> profile_order;

    ~GpuPrp() {
        if (bufSegMode) clReleaseMemObject(bufSegMode);
        if (bufSegThreshold) clReleaseMemObject(bufSegThreshold);
        if (bufSegBits) clReleaseMemObject(bufSegBits);
        if (bufSegValueLo) clReleaseMemObject(bufSegValueLo);
        if (bufFinalCarry) clReleaseMemObject(bufFinalCarry);
        if (bufBlockIncoming) clReleaseMemObject(bufBlockIncoming);
        if (bufBlockMode) clReleaseMemObject(bufBlockMode);
        if (bufBlockThreshold) clReleaseMemObject(bufBlockThreshold);
        if (bufBlockBits) clReleaseMemObject(bufBlockBits);
        if (bufBlockValueLo) clReleaseMemObject(bufBlockValueLo);
        if (bufBlockCarry) clReleaseMemObject(bufBlockCarry);
        if (bufTwInv) clReleaseMemObject(bufTwInv);
        if (bufTwFwd) clReleaseMemObject(bufTwFwd);
        if (bufWidth) clReleaseMemObject(bufWidth);
        if (bufShift) clReleaseMemObject(bufShift);
        if (bufField) clReleaseMemObject(bufField);
        if (bufDigits) clReleaseMemObject(bufDigits);
        if (k_carry_final_wrap) clReleaseKernel(k_carry_final_wrap);
        if (k_carry_block_apply_incoming_serial) clReleaseKernel(k_carry_block_apply_incoming_serial);
        if (k_carry_block_apply_incoming) clReleaseKernel(k_carry_block_apply_incoming);
        if (k_carry_block_prefix_chunked64) clReleaseKernel(k_carry_block_prefix_chunked64);
        if (k_carry_block_prefix) clReleaseKernel(k_carry_block_prefix);
        if (k_carry_block_local) clReleaseKernel(k_carry_block_local);
        if (k_mul_small) clReleaseKernel(k_mul_small);
        if (k_inverse_bridge_256_to_1024) clReleaseKernel(k_inverse_bridge_256_to_1024);
        if (k_forward_bridge_1024_to_256) clReleaseKernel(k_forward_bridge_1024_to_256);
        if (k_inverse_bridge_512_to_1024) clReleaseKernel(k_inverse_bridge_512_to_1024);
        if (k_forward_bridge_1024_to_512) clReleaseKernel(k_forward_bridge_1024_to_512);
        if (k_inverse_bridge_64_to_256) clReleaseKernel(k_inverse_bridge_64_to_256);
        if (k_forward_bridge_256_to_64) clReleaseKernel(k_forward_bridge_256_to_64);
        if (k_inverse_bridge_16_to_64) clReleaseKernel(k_inverse_bridge_16_to_64);
        if (k_forward_bridge_64_to_16) clReleaseKernel(k_forward_bridge_64_to_16);
        if (k_center_fused_4096) clReleaseKernel(k_center_fused_4096);
        if (k_center_fused_2048) clReleaseKernel(k_center_fused_2048);
        if (k_center_fused_1024) clReleaseKernel(k_center_fused_1024);
        if (k_center_fused_512) clReleaseKernel(k_center_fused_512);
        if (k_center_fused_256) clReleaseKernel(k_center_fused_256);
        if (k_center_fused_256_explicit) clReleaseKernel(k_center_fused_256_explicit);
        if (k_center_fused_64) clReleaseKernel(k_center_fused_64);
        if (k_center_fused_16) clReleaseKernel(k_center_fused_16);
        if (k_pointwise_sqr) clReleaseKernel(k_pointwise_sqr);
        if (k_last_stage_dit_unweight_wg64) clReleaseKernel(k_last_stage_dit_unweight_wg64);
        if (k_last_stage_dit_unweight_wg16) clReleaseKernel(k_last_stage_dit_unweight_wg16);
        if (k_last_stage_dit_unweight) clReleaseKernel(k_last_stage_dit_unweight);
        if (k_ntt_stage_dit_radix16) clReleaseKernel(k_ntt_stage_dit_radix16);
        if (k_ntt_stage_dif_radix16) clReleaseKernel(k_ntt_stage_dif_radix16);
        if (k_ntt_stage_dit_radix8) clReleaseKernel(k_ntt_stage_dit_radix8);
        if (k_ntt_stage_dif_radix8) clReleaseKernel(k_ntt_stage_dif_radix8);
        if (k_ntt_stage_dit_radix4) clReleaseKernel(k_ntt_stage_dit_radix4);
        if (k_ntt_stage_dif_radix4) clReleaseKernel(k_ntt_stage_dif_radix4);
        if (k_ntt_stage_dit) clReleaseKernel(k_ntt_stage_dit);
        if (k_ntt_stage_dif) clReleaseKernel(k_ntt_stage_dif);
        if (k_weight_first_stage_dif_wg64) clReleaseKernel(k_weight_first_stage_dif_wg64);
        if (k_weight_first_stage_dif_wg16) clReleaseKernel(k_weight_first_stage_dif_wg16);
        if (k_weight_first_stage_dif) clReleaseKernel(k_weight_first_stage_dif);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static std::vector<gf61::Elem> build_stage_twiddles(std::size_t n, bool inverse, std::vector<StageInfo>& stages_out) {
    gf61::Elem root = gf61::primitive_root_pow2(n);
    if (inverse) root = gf61::inv(root);
    std::vector<gf61::Elem> all;
    stages_out.clear();
    std::uint32_t offset = 0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        gf61::Elem wlen = root;
        for (std::size_t step = len; step < n; step <<= 1) wlen = gf61::sqr(wlen);
        const std::size_t half_len = len >> 1;
        stages_out.push_back(StageInfo{static_cast<std::uint32_t>(len), static_cast<std::uint32_t>(half_len), offset});
        gf61::Elem w(1, 0);
        for (std::size_t j = 0; j < half_len; ++j) {
            all.push_back(w);
            w = gf61::mul(w, wlen);
        }
        offset += static_cast<std::uint32_t>(half_len);
    }
    return all;
}

GpuPrp make_gpu(const DeviceInfo& info, const std::string& kernel_path, const ibdwt::Layout& layout, bool profile_kernels = false) {
    GpuPrp gpu;
    gpu.profile_kernels = profile_kernels;
    cl_int err = CL_SUCCESS;
    gpu.context = clCreateContext(nullptr, 1, &info.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
#if defined(CL_VERSION_2_0)
    if (profile_kernels) {
        const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        gpu.queue = clCreateCommandQueueWithProperties(gpu.context, info.device, props, &err);
        if (err != CL_SUCCESS || !gpu.queue) {
            err = CL_SUCCESS;
            gpu.queue = clCreateCommandQueue(gpu.context, info.device, CL_QUEUE_PROFILING_ENABLE, &err);
        }
    } else {
        gpu.queue = clCreateCommandQueue(gpu.context, info.device, 0, &err);
    }
#else
    gpu.queue = clCreateCommandQueue(gpu.context, info.device, profile_kernels ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
#endif
    check(err, profile_kernels ? "clCreateCommandQueue(profile)" : "clCreateCommandQueue");

    const std::string source = load_text_file(kernel_path);
    const char* src_ptr = source.c_str();
    const size_t src_len = source.size();
    gpu.program = clCreateProgramWithSource(gpu.context, 1, &src_ptr, &src_len, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(gpu.program, 1, &info.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(gpu.program, info.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size) clGetProgramBuildInfo(gpu.program, info.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw std::runtime_error("Build failed:\n" + log);
    }

    auto make_kernel = [&](const char* name, cl_kernel* out) {
        *out = clCreateKernel(gpu.program, name, &err);
        check(err, name);
    };
    make_kernel("gf61_weight_first_stage_dif", &gpu.k_weight_first_stage_dif);
    make_kernel("gf61_weight_first_stage_dif_wg16", &gpu.k_weight_first_stage_dif_wg16);
    make_kernel("gf61_weight_first_stage_dif_wg64", &gpu.k_weight_first_stage_dif_wg64);
    make_kernel("gf61_ntt_stage_dif", &gpu.k_ntt_stage_dif);
    make_kernel("gf61_ntt_stage_dit", &gpu.k_ntt_stage_dit);
    make_kernel("gf61_ntt_stage_dif_radix4", &gpu.k_ntt_stage_dif_radix4);
    make_kernel("gf61_ntt_stage_dit_radix4", &gpu.k_ntt_stage_dit_radix4);
    make_kernel("gf61_ntt_stage_dif_radix8", &gpu.k_ntt_stage_dif_radix8);
    make_kernel("gf61_ntt_stage_dit_radix8", &gpu.k_ntt_stage_dit_radix8);
    make_kernel("gf61_ntt_stage_dif_radix16", &gpu.k_ntt_stage_dif_radix16);
    make_kernel("gf61_ntt_stage_dit_radix16", &gpu.k_ntt_stage_dit_radix16);
    make_kernel("gf61_last_stage_dit_unweight", &gpu.k_last_stage_dit_unweight);
    make_kernel("gf61_last_stage_dit_unweight_wg16", &gpu.k_last_stage_dit_unweight_wg16);
    make_kernel("gf61_last_stage_dit_unweight_wg64", &gpu.k_last_stage_dit_unweight_wg64);
    make_kernel("gf61_pointwise_sqr", &gpu.k_pointwise_sqr);
    make_kernel("gf61_center_fused_16", &gpu.k_center_fused_16);
    make_kernel("gf61_center_fused_64", &gpu.k_center_fused_64);
    make_kernel("gf61_center_fused_256", &gpu.k_center_fused_256);
    make_kernel("gf61_center_fused_256_explicit", &gpu.k_center_fused_256_explicit);
    make_kernel("gf61_center_fused_512", &gpu.k_center_fused_512);
    make_kernel("gf61_center_fused_1024", &gpu.k_center_fused_1024);
    make_kernel("gf61_center_fused_2048", &gpu.k_center_fused_2048);
    make_kernel("gf61_center_fused_4096", &gpu.k_center_fused_4096);
    make_kernel("gf61_forward_bridge_64_to_16", &gpu.k_forward_bridge_64_to_16);
    make_kernel("gf61_inverse_bridge_16_to_64", &gpu.k_inverse_bridge_16_to_64);
    make_kernel("gf61_forward_bridge_256_to_64", &gpu.k_forward_bridge_256_to_64);
    make_kernel("gf61_inverse_bridge_64_to_256", &gpu.k_inverse_bridge_64_to_256);
    make_kernel("gf61_forward_bridge_1024_to_512", &gpu.k_forward_bridge_1024_to_512);
    make_kernel("gf61_inverse_bridge_512_to_1024", &gpu.k_inverse_bridge_512_to_1024);
    make_kernel("gf61_forward_bridge_1024_to_256", &gpu.k_forward_bridge_1024_to_256);
    make_kernel("gf61_inverse_bridge_256_to_1024", &gpu.k_inverse_bridge_256_to_1024);
    make_kernel("gf61_forward_ext_1024_to_256_explicit", &gpu.k_forward_ext_1024_to_256_explicit);
    make_kernel("gf61_inverse_ext_256_to_1024_explicit", &gpu.k_inverse_ext_256_to_1024_explicit);
    make_kernel("gf61_forward_ext_2048_to_256_explicit", &gpu.k_forward_ext_2048_to_256_explicit);
    make_kernel("gf61_inverse_ext_256_to_2048_explicit", &gpu.k_inverse_ext_256_to_2048_explicit);
    make_kernel("gf61_mul_small_digits", &gpu.k_mul_small);
    make_kernel("gf61_carry_block_local", &gpu.k_carry_block_local);
    make_kernel("gf61_carry_block_prefix_serial", &gpu.k_carry_block_prefix);
    make_kernel("gf61_carry_block_prefix_chunked64", &gpu.k_carry_block_prefix_chunked64);
    make_kernel("gf61_carry_block_apply_incoming", &gpu.k_carry_block_apply_incoming);
    make_kernel("gf61_carry_block_apply_incoming_serial", &gpu.k_carry_block_apply_incoming_serial);
    make_kernel("gf61_carry_final_wrap_serial", &gpu.k_carry_final_wrap);

    gpu.n = layout.n;
    gpu.max_work_group_size = info.max_work_group_size;
    gpu.local_mem_size = info.local_mem_size;
    const std::size_t n = layout.n;
    gpu.bufDigits = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * sizeof(std::uint64_t), nullptr, &err);
    check(err, "clCreateBuffer(bufDigits)");
    gpu.bufField = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * sizeof(gf61::Elem), nullptr, &err);
    check(err, "clCreateBuffer(bufField)");
    gpu.bufShift = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(std::uint8_t), const_cast<std::uint8_t*>(layout.shifts.data()), &err);
    check(err, "clCreateBuffer(bufShift)");
    gpu.bufWidth = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(std::uint8_t), const_cast<std::uint8_t*>(layout.digit_width.data()), &err);
    check(err, "clCreateBuffer(bufWidth)");

    std::vector<StageInfo> stages_fwd;
    auto tw_fwd = build_stage_twiddles(n, false, stages_fwd);
    std::vector<StageInfo> stages_inv;
    auto tw_inv = build_stage_twiddles(n, true, stages_inv);
    if (stages_fwd.size() != stages_inv.size()) throw std::runtime_error("stage count mismatch");
    gpu.stages = stages_fwd;
    gpu.log_n = static_cast<cl_uint>(stages_fwd.size());

    gpu.bufTwFwd = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_fwd.size() * sizeof(gf61::Elem), tw_fwd.data(), &err);
    check(err, "clCreateBuffer(bufTwFwd)");
    gpu.bufTwInv = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_inv.size() * sizeof(gf61::Elem), tw_inv.data(), &err);
    check(err, "clCreateBuffer(bufTwInv)");

    return gpu;
}

static bool g_planner_debug = false;

struct CarryConfig {
    cl_uint block_size = 256;
    cl_uint items_per_worker = 4;
    cl_uint local_size = 64;
    cl_uint num_blocks = 0;
};

static CarryConfig choose_carry_config(const DeviceInfo& dev, std::size_t n, cl_uint block_override, cl_uint items_override) {
    auto fits = [&](cl_uint block_size, cl_uint items_per_worker) -> bool {
        if (block_size == 0 || items_per_worker == 0 || (block_size % items_per_worker) != 0) return false;
        const cl_uint local_size = block_size / items_per_worker;
        const std::size_t local_bytes = std::size_t(block_size) * (sizeof(std::uint64_t) + sizeof(std::uint8_t));
        return local_size <= dev.max_work_group_size && local_bytes <= static_cast<std::size_t>(dev.local_mem_size);
    };

    CarryConfig cfg;
    if (block_override != 0) cfg.block_size = block_override;
    if (items_override != 0) cfg.items_per_worker = items_override;

    const bool user_overrode = (block_override != 0 || items_override != 0);
    if (!user_overrode) {
        const bool gfx906_like = (dev.max_work_group_size <= 256 && dev.local_mem_size <= 65536);
        if (gfx906_like) {
            if (n <= 1024u && fits(64u, 4u)) {
                cfg.block_size = 64u;
                cfg.items_per_worker = 4u;
            } else if (n >= (1u << 20) && fits(1024u, 64u)) {
                cfg.block_size = 1024u;
                cfg.items_per_worker = 64u;
            } else if (n >= (1u << 16) && fits(512u, 32u)) {
                cfg.block_size = 512u;
                cfg.items_per_worker = 32u;
            }
        }
    }

    if (cfg.block_size == 0 || cfg.items_per_worker == 0 || (cfg.block_size % cfg.items_per_worker) != 0) {
        throw std::runtime_error("carry config requires block_size % items_per_worker == 0 and both non-zero");
    }
    cfg.local_size = cfg.block_size / cfg.items_per_worker;

    if (!fits(cfg.block_size, cfg.items_per_worker)) {
        const std::pair<cl_uint, cl_uint> fallbacks[] = {
            {1024u,64u}, {512u,32u}, {256u,4u}, {128u,4u}, {64u,4u}, {64u,2u}
        };
        bool found = false;
        for (auto [b, v] : fallbacks) {
            if (fits(b, v)) {
                cfg.block_size = b;
                cfg.items_per_worker = v;
                cfg.local_size = b / v;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("no valid carry block configuration for this device");
    }

    cfg.num_blocks = static_cast<cl_uint>((n + cfg.block_size - 1) / cfg.block_size);
    return cfg;
}


struct CenterKernelConfig {
    cl_kernel kernel = nullptr;
    cl_uint chunk = 0;
    cl_uint local_size = 0;
    bool enabled = false;
};

struct BridgeKernelConfig {
    cl_kernel forward_kernel = nullptr;
    cl_kernel inverse_kernel = nullptr;
    cl_uint outer_chunk = 0;
    cl_uint inner_chunk = 0;
    cl_uint local_size = 0;
    bool enabled = false;
};

static bool is_gfx906_like(const GpuPrp& gpu) {
    return gpu.max_work_group_size <= 256 && gpu.local_mem_size <= 65536;
}

static void profile_record_completed_event(GpuPrp& gpu, const std::string& label, cl_event ev) {
    if (!gpu.profile_kernels || !ev) return;
    cl_ulong t0 = 0, t1 = 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr) == CL_SUCCESS &&
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr) == CL_SUCCESS && t1 >= t0) {
        auto it = gpu.profile_totals.find(label);
        if (it == gpu.profile_totals.end()) gpu.profile_order.push_back(label);
        auto& slot = gpu.profile_totals[label];
        slot.ms += double(t1 - t0) * 1e-6;
        slot.launches += 1;
    }
}

static void profile_flush_pending(GpuPrp& gpu) {
    if (!gpu.profile_kernels) return;
    for (auto& item : gpu.pending_profile_events) {
        if (item.second) {
            clWaitForEvents(1, &item.second);
            profile_record_completed_event(gpu, item.first, item.second);
            clReleaseEvent(item.second);
        }
    }
    gpu.pending_profile_events.clear();
}

static void profile_print_summary(const GpuPrp& gpu, const std::string& title = "Kernel profile summary") {
    if (!gpu.profile_kernels) return;
    if (gpu.profile_order.empty()) {
        std::cout << title << ": no event timing data collected.\n";
        return;
    }
    double total_ms = 0.0;
    for (const auto& name : gpu.profile_order) total_ms += gpu.profile_totals.at(name).ms;
    if (total_ms <= 0.0) {
        std::cout << title << ": profiling events were recorded but timings are unavailable on this queue/device.\n";
        return;
    }
    std::cout << title << ":\n";
    std::vector<std::pair<std::string, ProfileEntry>> rows;
    rows.reserve(gpu.profile_order.size());
    for (const auto& name : gpu.profile_order) rows.push_back({name, gpu.profile_totals.at(name)});
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    for (const auto& row : rows) {
        const double pct = 100.0 * row.second.ms / total_ms;
        std::cout << "  " << row.first << ": " << std::fixed << std::setprecision(3) << row.second.ms
                  << " ms total (" << std::setprecision(1) << pct << "%, launches=" << row.second.launches << ")\n";
    }
}

static void enqueue_kernel(GpuPrp& gpu, cl_kernel kernel, size_t global, const size_t* local, const char* what, const char* profile_label) {
    cl_event ev = nullptr;
    check(clEnqueueNDRangeKernel(gpu.queue, kernel, 1, nullptr, &global, local, 0, nullptr, gpu.profile_kernels ? &ev : nullptr), what);
    if (gpu.profile_kernels && ev) gpu.pending_profile_events.push_back({profile_label, ev});
}

static cl_uint choose_default_center_cap(const GpuPrp& gpu) {
    if (is_gfx906_like(gpu)) {
        if (gpu.log_n >= 20u) return 256u;
        if (gpu.log_n >= 16u) return 256u;
        return 512u;
    }
    return 4096u;
}

static CenterKernelConfig choose_center_kernel(GpuPrp& gpu, cl_uint center_max = 0) {
    const cl_uint cap = center_max ? center_max : choose_default_center_cap(gpu);
    struct Candidate { cl_uint chunk; cl_uint wg; cl_kernel GpuPrp::*kernel_member; };
    const bool prefer_explicit_256 = is_gfx906_like(gpu) && gpu.log_n >= 20u;
    const Candidate candidates[] = {
        {4096u, 64u, &GpuPrp::k_center_fused_4096},
        {2048u, 64u, &GpuPrp::k_center_fused_2048},
        {1024u, 64u, &GpuPrp::k_center_fused_1024},
        {512u, 64u, &GpuPrp::k_center_fused_512},
        {256u, 64u, prefer_explicit_256 ? &GpuPrp::k_center_fused_256_explicit : &GpuPrp::k_center_fused_256},
        {64u, 16u, &GpuPrp::k_center_fused_64},
        {16u, 8u, &GpuPrp::k_center_fused_16},
    };
    for (const auto& cand : candidates) {
        if (cand.chunk > cap) continue;
        const std::size_t local_bytes = std::size_t(cand.chunk) * sizeof(gf61::Elem);
        if (gpu.n < std::size_t(cand.chunk) * 2u) continue;
        if ((gpu.n % cand.chunk) != 0u) continue;
        if (cand.wg > gpu.max_work_group_size) continue;
        if (local_bytes > static_cast<std::size_t>(gpu.local_mem_size)) continue;
        cl_kernel k = gpu.*(cand.kernel_member);
        if (!k) continue;
        return CenterKernelConfig{k, cand.chunk, cand.wg, true};
    }
    return {};
}

static BridgeKernelConfig choose_bridge_kernel(GpuPrp& gpu, const CenterKernelConfig& center) {
    if (is_gfx906_like(gpu) && gpu.log_n >= 20u) return {};
    struct Candidate {
        cl_uint outer_chunk;
        cl_uint inner_chunk;
        cl_uint wg;
        cl_kernel GpuPrp::*forward_member;
        cl_kernel GpuPrp::*inverse_member;
    };
    const Candidate candidates[] = {
        {1024u, 512u, 64u, &GpuPrp::k_forward_bridge_1024_to_512, &GpuPrp::k_inverse_bridge_512_to_1024},
        {1024u, 256u, 64u, &GpuPrp::k_forward_bridge_1024_to_256, &GpuPrp::k_inverse_bridge_256_to_1024},
        {256u, 64u, 64u, &GpuPrp::k_forward_bridge_256_to_64, &GpuPrp::k_inverse_bridge_64_to_256},
        {64u, 16u, 16u, &GpuPrp::k_forward_bridge_64_to_16, &GpuPrp::k_inverse_bridge_16_to_64},
    };
    for (const auto& cand : candidates) {
        if (center.chunk != cand.inner_chunk) continue;
        if (gpu.n < std::size_t(cand.outer_chunk) * 2u) continue;
        if ((gpu.n % cand.outer_chunk) != 0u) continue;
        const std::size_t local_bytes = std::size_t(cand.outer_chunk) * sizeof(gf61::Elem);
        if (cand.wg > gpu.max_work_group_size) continue;
        if (local_bytes > static_cast<std::size_t>(gpu.local_mem_size)) continue;
        cl_kernel fk = gpu.*(cand.forward_member);
        cl_kernel ik = gpu.*(cand.inverse_member);
        if (!fk || !ik) continue;
        return BridgeKernelConfig{fk, ik, cand.outer_chunk, cand.inner_chunk, cand.wg, true};
    }
    return {};
}

static std::pair<cl_kernel, size_t> choose_weight_first_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 2;
    if (global >= 64u && gpu.k_weight_first_stage_dif_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_weight_first_stage_dif_wg64, 64u};
    if (global >= 16u && gpu.k_weight_first_stage_dif_wg16 && gpu.max_work_group_size >= 16u) return {gpu.k_weight_first_stage_dif_wg16, 16u};
    return {gpu.k_weight_first_stage_dif, 0u};
}

static std::pair<cl_kernel, size_t> choose_last_stage_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 2;
    if (global >= 64u && gpu.k_last_stage_dit_unweight_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_last_stage_dit_unweight_wg64, 64u};
    if (global >= 16u && gpu.k_last_stage_dit_unweight_wg16 && gpu.max_work_group_size >= 16u) return {gpu.k_last_stage_dit_unweight_wg16, 16u};
    return {gpu.k_last_stage_dit_unweight, 0u};
}


static bool can_use_true_ext1024_path(GpuPrp& gpu, cl_uint center_max) {
    (void)center_max;
    if (!gpu.k_forward_ext_1024_to_256_explicit) return false;
    if (!gpu.k_inverse_ext_256_to_1024_explicit) return false;
    if (!gpu.k_center_fused_256_explicit) return false;
    if (gpu.n < 1024u) return false;
    if ((gpu.n % 1024u) != 0u) return false;
    return true;
}

static void enqueue_true_ext1024_forward(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_forward_ext_1024_to_256_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext1024_fwd a");
    check(clSetKernelArg(gpu.k_forward_ext_1024_to_256_explicit, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ext1024_fwd tw");
    const size_t global = (gpu.n / 1024u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_forward_ext_1024_to_256_explicit, global, &local, "enqueue true_ext1024_forward", "ntt_external_forward");
}

static void enqueue_true_ext1024_inverse(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_1024_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext1024_inv a");
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_1024_explicit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set ext1024_inv tw");
    const size_t global = (gpu.n / 1024u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_inverse_ext_256_to_1024_explicit, global, &local, "enqueue true_ext1024_inverse", "ntt_external_inverse");
}

static bool can_use_true_ext2048_path(GpuPrp& gpu, cl_uint center_max) {
    (void)center_max;
    if (!gpu.k_forward_ext_2048_to_256_explicit) return false;
    if (!gpu.k_inverse_ext_256_to_2048_explicit) return false;
    if (!gpu.k_center_fused_256_explicit) return false;
    if (gpu.n < 2048u) return false;
    if ((gpu.n % 2048u) != 0u) return false;
    return true;
}

static void enqueue_true_ext2048_forward(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_forward_ext_2048_to_256_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext2048_fwd a");
    check(clSetKernelArg(gpu.k_forward_ext_2048_to_256_explicit, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ext2048_fwd tw");
    const size_t global = (gpu.n / 2048u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_forward_ext_2048_to_256_explicit, global, &local, "enqueue true_ext2048_forward", "ntt_external_forward");
}

static void enqueue_true_ext2048_inverse(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_2048_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext2048_inv a");
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_2048_explicit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set ext2048_inv tw");
    const size_t global = (gpu.n / 2048u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_inverse_ext_256_to_2048_explicit, global, &local, "enqueue true_ext2048_inverse", "ntt_external_inverse");
}

static void enqueue_bridge_forward(GpuPrp& gpu, const BridgeKernelConfig& cfg) {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.forward_kernel, 0, sizeof(cl_mem), &gpu.bufField), "set bridge_fwd a");
    check(clSetKernelArg(cfg.forward_kernel, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set bridge_fwd tw");
    const size_t global = (gpu.n / cfg.outer_chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    check(clEnqueueNDRangeKernel(gpu.queue, cfg.forward_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue bridge_forward");
}

static void enqueue_bridge_inverse(GpuPrp& gpu, const BridgeKernelConfig& cfg) {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.inverse_kernel, 0, sizeof(cl_mem), &gpu.bufField), "set bridge_inv a");
    check(clSetKernelArg(cfg.inverse_kernel, 1, sizeof(cl_mem), &gpu.bufTwInv), "set bridge_inv tw");
    const size_t global = (gpu.n / cfg.outer_chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    check(clEnqueueNDRangeKernel(gpu.queue, cfg.inverse_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue bridge_inverse");
}

static void enqueue_center_fused(GpuPrp& gpu, const CenterKernelConfig& cfg) {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.kernel, 0, sizeof(cl_mem), &gpu.bufField), "set center_fused a");
    check(clSetKernelArg(cfg.kernel, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set center_fused tw_fwd");
    check(clSetKernelArg(cfg.kernel, 2, sizeof(cl_mem), &gpu.bufTwInv), "set center_fused tw_inv");
    const size_t global = (gpu.n / cfg.chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    enqueue_kernel(gpu, cfg.kernel, global, &local, "enqueue center_fused", "center_fused");
}

static void enqueue_forward_pipeline_partial(GpuPrp& gpu, cl_uint stop_chunk);

static void enqueue_forward_pipeline(GpuPrp& gpu) {
    enqueue_forward_pipeline_partial(gpu, 0u);
}

static void enqueue_forward_pipeline_partial(GpuPrp& gpu, cl_uint stop_chunk) {
    const StageInfo& first = gpu.stages.back();

    const auto weight_kernel = choose_weight_first_kernel(gpu);
    check(clSetKernelArg(weight_kernel.first, 0, sizeof(cl_mem), &gpu.bufDigits), "set weight_first digits");
    check(clSetKernelArg(weight_kernel.first, 1, sizeof(cl_mem), &gpu.bufShift), "set weight_first shifts");
    check(clSetKernelArg(weight_kernel.first, 2, sizeof(cl_mem), &gpu.bufField), "set weight_first field");
    check(clSetKernelArg(weight_kernel.first, 3, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_first tw");
    check(clSetKernelArg(weight_kernel.first, 4, sizeof(cl_uint), &first.offset), "set weight_first off");
    check(clSetKernelArg(weight_kernel.first, 5, sizeof(cl_uint), &first.len), "set weight_first len");
    check(clSetKernelArg(weight_kernel.first, 6, sizeof(cl_uint), &first.half_len), "set weight_first half");
    {
        const size_t global = gpu.n / 2;
        const size_t* local_ptr = weight_kernel.second ? &weight_kernel.second : nullptr;
        enqueue_kernel(gpu, weight_kernel.first, global, local_ptr, "enqueue weight_first_stage_dif", "weight_first_stage_dif");
    }

    std::vector<StageInfo> todo;
    for (auto it = gpu.stages.rbegin() + 1; it != gpu.stages.rend(); ++it) {
        const StageInfo& st = *it;
        if (stop_chunk != 0u && st.len <= stop_chunk) break;
        todo.push_back(st);
    }

    for (std::size_t idx = 0; idx < todo.size();) {
        const StageInfo& st = todo[idx];
        const bool can_triplet = (idx + 2 < todo.size()) &&
                                 (todo[idx + 1].len * 2u == st.len) &&
                                 (todo[idx + 2].len * 4u == st.len) &&
                                 st.len >= 8u && gpu.k_ntt_stage_dif_radix8 && gpu.n >= 512u;
        if (can_triplet) {
            const StageInfo& st2 = todo[idx + 1];
            const StageInfo& st3 = todo[idx + 2];
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif_radix8 a");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix8 tw1");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 2, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix8 tw2");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 3, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix8 tw3");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 4, sizeof(cl_uint), &st.offset), "set ntt_stage_dif_radix8 off1");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 5, sizeof(cl_uint), &st2.offset), "set ntt_stage_dif_radix8 off2");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 6, sizeof(cl_uint), &st3.offset), "set ntt_stage_dif_radix8 off3");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix8, 7, sizeof(cl_uint), &st.len), "set ntt_stage_dif_radix8 len");
            const size_t global = gpu.n / 8;
            const size_t local = 64u;
            enqueue_kernel(gpu, gpu.k_ntt_stage_dif_radix8, global, &local, "enqueue ntt_stage_dif_radix8 fwd", "ntt_radix8_forward");
            idx += 3;
            continue;
        }
        const bool can_pair = (idx + 1 < todo.size()) && (todo[idx + 1].len * 2u == st.len) && st.len >= 4u && gpu.k_ntt_stage_dif_radix4 && gpu.n >= 256u;
        if (can_pair) {
            const StageInfo& st2 = todo[idx + 1];
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif_radix4 a");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4 tw1");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 2, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4 tw2");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 3, sizeof(cl_uint), &st.offset), "set ntt_stage_dif_radix4 off1");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 4, sizeof(cl_uint), &st2.offset), "set ntt_stage_dif_radix4 off2");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_radix4, 5, sizeof(cl_uint), &st.len), "set ntt_stage_dif_radix4 len");
            const size_t global = gpu.n / 4;
            const size_t local = 64u;
            enqueue_kernel(gpu, gpu.k_ntt_stage_dif_radix4, global, &local, "enqueue ntt_stage_dif_radix4 fwd", "ntt_radix4_forward");
            idx += 2;
            continue;
        }
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif a");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 2, sizeof(cl_uint), &st.offset), "set ntt_stage_dif off");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 3, sizeof(cl_uint), &st.len), "set ntt_stage_dif len");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 4, sizeof(cl_uint), &st.half_len), "set ntt_stage_dif half");
        const size_t global = gpu.n / 2;
        enqueue_kernel(gpu, gpu.k_ntt_stage_dif, global, nullptr, "enqueue ntt_stage_dif fwd", "ntt_stage_dif");
        idx += 1;
    }
}

static void enqueue_inverse_pipeline_partial(GpuPrp& gpu, cl_uint skip_chunk);

static void enqueue_inverse_pipeline(GpuPrp& gpu) {
    enqueue_inverse_pipeline_partial(gpu, 0u);
}

static void enqueue_inverse_pipeline_partial(GpuPrp& gpu, cl_uint skip_chunk) {
    const StageInfo& last = gpu.stages.back();
    std::vector<StageInfo> todo;
    for (std::size_t idx = 0; idx + 1 < gpu.stages.size(); ++idx) {
        const StageInfo& st = gpu.stages[idx];
        if (skip_chunk != 0u && st.len <= skip_chunk) continue;
        todo.push_back(st);
    }
    for (std::size_t idx = 0; idx < todo.size();) {
        const StageInfo& st = todo[idx];
        const bool can_triplet = (idx + 2 < todo.size()) &&
                                 (todo[idx + 1].len == st.len * 2u) &&
                                 (todo[idx + 2].len == st.len * 4u) &&
                                 st.len >= 2u && gpu.k_ntt_stage_dit_radix8 && gpu.n >= 512u;
        if (can_triplet) {
            const StageInfo& st2 = todo[idx + 1];
            const StageInfo& st3 = todo[idx + 2];
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit_radix8 a");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix8 tw1");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 2, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix8 tw2");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 3, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix8 tw3");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 4, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit_radix8 off1");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 5, sizeof(cl_uint), &st2.offset), "set inv_ntt_stage_dit_radix8 off2");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 6, sizeof(cl_uint), &st3.offset), "set inv_ntt_stage_dit_radix8 off3");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix8, 7, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit_radix8 len");
            const size_t global = gpu.n / 8;
            const size_t local = 64u;
            enqueue_kernel(gpu, gpu.k_ntt_stage_dit_radix8, global, &local, "enqueue ntt_stage_dit_radix8 inv", "ntt_radix8_inverse");
            idx += 3;
            continue;
        }
        const bool can_pair = (idx + 1 < todo.size()) && (todo[idx + 1].len == st.len * 2u) && st.len >= 2u && gpu.k_ntt_stage_dit_radix4 && gpu.n >= 256u;
        if (can_pair) {
            const StageInfo& st2 = todo[idx + 1];
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit_radix4 a");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4 tw1");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 2, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4 tw2");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 3, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit_radix4 off1");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 4, sizeof(cl_uint), &st2.offset), "set inv_ntt_stage_dit_radix4 off2");
            check(clSetKernelArg(gpu.k_ntt_stage_dit_radix4, 5, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit_radix4 len");
            const size_t global = gpu.n / 4;
            const size_t local = 64u;
            enqueue_kernel(gpu, gpu.k_ntt_stage_dit_radix4, global, &local, "enqueue ntt_stage_dit_radix4 inv", "ntt_radix4_inverse");
            idx += 2;
            continue;
        }
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit a");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 2, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit off");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 3, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit len");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 4, sizeof(cl_uint), &st.half_len), "set inv_ntt_stage_dit half");
        const size_t global = gpu.n / 2;
        enqueue_kernel(gpu, gpu.k_ntt_stage_dit, global, nullptr, "enqueue ntt_stage_dit inv", "ntt_stage_dit");
        idx += 1;
    }

    const auto last_kernel = choose_last_stage_kernel(gpu);
    check(clSetKernelArg(last_kernel.first, 0, sizeof(cl_mem), &gpu.bufField), "set last_stage a");
    check(clSetKernelArg(last_kernel.first, 1, sizeof(cl_mem), &gpu.bufTwInv), "set last_stage tw");
    check(clSetKernelArg(last_kernel.first, 2, sizeof(cl_mem), &gpu.bufShift), "set last_stage shifts");
    check(clSetKernelArg(last_kernel.first, 3, sizeof(cl_mem), &gpu.bufDigits), "set last_stage digits");
    const cl_uint log_n = static_cast<cl_uint>(gpu.stages.size());
    check(clSetKernelArg(last_kernel.first, 4, sizeof(cl_uint), &log_n), "set last_stage logn");
    check(clSetKernelArg(last_kernel.first, 5, sizeof(cl_uint), &last.offset), "set last_stage off");
    check(clSetKernelArg(last_kernel.first, 6, sizeof(cl_uint), &last.len), "set last_stage len");
    check(clSetKernelArg(last_kernel.first, 7, sizeof(cl_uint), &last.half_len), "set last_stage half");
    {
        const size_t global = gpu.n / 2;
        const size_t* local_ptr = last_kernel.second ? &last_kernel.second : nullptr;
        enqueue_kernel(gpu, last_kernel.first, global, local_ptr, "enqueue last_stage_dit_unweight", "last_stage_dit_unweight");
    }
}

static void enqueue_square_mod(GpuPrp& gpu, cl_uint center_max = 0) {
    const cl_uint plan_log_n = gpu.log_n ? gpu.log_n : static_cast<cl_uint>(gpu.stages.size());
    const bool cond_log_n_1024 = plan_log_n >= 20u;
    const bool cond_n_1024 = gpu.n >= 1024u;
    const bool cond_mod_1024 = ((gpu.n % 1024u) == 0u);
    const bool cond_k_fwd_1024 = (gpu.k_forward_ext_1024_to_256_explicit != nullptr);
    const bool cond_k_inv_1024 = (gpu.k_inverse_ext_256_to_1024_explicit != nullptr);
    const bool cond_k_center_256 = (gpu.k_center_fused_256_explicit != nullptr);
    const bool cond_true_1024 = cond_log_n_1024 && cond_n_1024 && cond_mod_1024 && cond_k_fwd_1024 && cond_k_inv_1024 && cond_k_center_256;
    const bool cond_true_2048 = (can_use_true_ext2048_path(gpu, center_max) && plan_log_n >= 20u);
    if (g_planner_debug) {
        std::cerr
            << "[planner-debug] log_n=" << plan_log_n
            << " n=" << gpu.n
            << " center_max=" << center_max
            << " cond_log_n_1024=" << (cond_log_n_1024 ? 1 : 0)
            << " cond_n_1024=" << (cond_n_1024 ? 1 : 0)
            << " cond_mod_1024=" << (cond_mod_1024 ? 1 : 0)
            << " cond_k_fwd_1024=" << (cond_k_fwd_1024 ? 1 : 0)
            << " cond_k_inv_1024=" << (cond_k_inv_1024 ? 1 : 0)
            << " cond_k_center_256=" << (cond_k_center_256 ? 1 : 0)
            << " cond_true_1024=" << (cond_true_1024 ? 1 : 0)
            << " cond_true_2048=" << (cond_true_2048 ? 1 : 0)
            << "\n";
    }
    if (cond_true_1024) {
        if (g_planner_debug) std::cerr << "[planner] TRUE_EXT1024 path\n";
        const CenterKernelConfig center{gpu.k_center_fused_256_explicit, 256u, 64u, true};
        enqueue_forward_pipeline_partial(gpu, 1024u);
        enqueue_true_ext1024_forward(gpu);
        enqueue_center_fused(gpu, center);
        enqueue_true_ext1024_inverse(gpu);
        enqueue_inverse_pipeline_partial(gpu, 1024u);
        return;
    }
    if (cond_true_2048) {
        if (g_planner_debug) std::cerr << "[planner] TRUE_EXT2048 path\n";
        const CenterKernelConfig center{gpu.k_center_fused_256_explicit, 256u, 64u, true};
        enqueue_forward_pipeline_partial(gpu, 2048u);
        enqueue_true_ext2048_forward(gpu);
        enqueue_center_fused(gpu, center);
        enqueue_true_ext2048_inverse(gpu);
        enqueue_inverse_pipeline_partial(gpu, 2048u);
        return;
    }
    if (g_planner_debug) std::cerr << "[planner] fallback path\n";
    const CenterKernelConfig center = choose_center_kernel(gpu, center_max);
    if (center.enabled) {
        const BridgeKernelConfig bridge = choose_bridge_kernel(gpu, center);
        if (g_planner_debug) {
            std::cerr
                << "[planner-debug-fallback] center_enabled=" << (center.enabled ? 1 : 0)
                << " center_chunk=" << center.chunk
                << " center_local=" << center.local_size
                << " bridge_enabled=" << (bridge.enabled ? 1 : 0)
                << " bridge_outer_chunk=" << bridge.outer_chunk
                << " bridge_inner_chunk=" << bridge.inner_chunk
                << " bridge_local=" << bridge.local_size
                << "\n";
        }
        const cl_uint outer_stop = bridge.enabled ? bridge.outer_chunk : center.chunk;
        enqueue_forward_pipeline_partial(gpu, outer_stop);
        if (bridge.enabled) enqueue_bridge_forward(gpu, bridge);
        enqueue_center_fused(gpu, center);
        if (bridge.enabled) enqueue_bridge_inverse(gpu, bridge);
        enqueue_inverse_pipeline_partial(gpu, outer_stop);
        return;
    }
    enqueue_forward_pipeline(gpu);
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_pointwise_sqr, 0, sizeof(cl_mem), &gpu.bufField), "set pointwise_sqr a");
    check(clSetKernelArg(gpu.k_pointwise_sqr, 1, sizeof(cl_uint), &n_u32), "set pointwise_sqr n");
    {
        const size_t global = gpu.n;
        enqueue_kernel(gpu, gpu.k_pointwise_sqr, global, nullptr, "enqueue pointwise_sqr", "pointwise_sqr");
    }
    enqueue_inverse_pipeline(gpu);
}

static void ensure_carry_buffers(GpuPrp& gpu, cl_uint num_blocks, cl_uint local_size) {
    const cl_uint num_segments = num_blocks * std::max<cl_uint>(local_size, 1u);
    if (gpu.carry_buffer_blocks >= num_blocks && gpu.carry_buffer_segments >= num_segments && gpu.bufBlockCarry && gpu.bufBlockValueLo && gpu.bufBlockBits &&
        gpu.bufBlockThreshold && gpu.bufBlockMode && gpu.bufBlockIncoming && gpu.bufFinalCarry && gpu.bufSegValueLo && gpu.bufSegBits && gpu.bufSegThreshold && gpu.bufSegMode) return;

    if (gpu.bufSegMode) { clReleaseMemObject(gpu.bufSegMode); gpu.bufSegMode = nullptr; }
    if (gpu.bufSegThreshold) { clReleaseMemObject(gpu.bufSegThreshold); gpu.bufSegThreshold = nullptr; }
    if (gpu.bufSegBits) { clReleaseMemObject(gpu.bufSegBits); gpu.bufSegBits = nullptr; }
    if (gpu.bufSegValueLo) { clReleaseMemObject(gpu.bufSegValueLo); gpu.bufSegValueLo = nullptr; }
    if (gpu.bufFinalCarry) { clReleaseMemObject(gpu.bufFinalCarry); gpu.bufFinalCarry = nullptr; }
    if (gpu.bufBlockIncoming) { clReleaseMemObject(gpu.bufBlockIncoming); gpu.bufBlockIncoming = nullptr; }
    if (gpu.bufBlockMode) { clReleaseMemObject(gpu.bufBlockMode); gpu.bufBlockMode = nullptr; }
    if (gpu.bufBlockThreshold) { clReleaseMemObject(gpu.bufBlockThreshold); gpu.bufBlockThreshold = nullptr; }
    if (gpu.bufBlockBits) { clReleaseMemObject(gpu.bufBlockBits); gpu.bufBlockBits = nullptr; }
    if (gpu.bufBlockValueLo) { clReleaseMemObject(gpu.bufBlockValueLo); gpu.bufBlockValueLo = nullptr; }
    if (gpu.bufBlockCarry) { clReleaseMemObject(gpu.bufBlockCarry); gpu.bufBlockCarry = nullptr; }

    cl_int err = CL_SUCCESS;
    const std::size_t blocks = std::size_t(std::max<cl_uint>(num_blocks, 1u));
    const std::size_t segments = std::size_t(std::max<cl_uint>(num_segments, 1u));
    gpu.bufBlockCarry = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockCarry)");
    gpu.bufBlockValueLo = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockValueLo)");
    gpu.bufBlockBits = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_uint), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockBits)");
    gpu.bufBlockThreshold = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockThreshold)");
    gpu.bufBlockMode = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_uchar), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockMode)");
    gpu.bufBlockIncoming = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, blocks * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufBlockIncoming)");
    gpu.bufFinalCarry = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufFinalCarry)");
    gpu.bufSegValueLo = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, segments * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufSegValueLo)");
    gpu.bufSegBits = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, segments * sizeof(cl_uint), nullptr, &err);
    check(err, "clCreateBuffer(bufSegBits)");
    gpu.bufSegThreshold = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, segments * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(bufSegThreshold)");
    gpu.bufSegMode = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, segments * sizeof(cl_uchar), nullptr, &err);
    check(err, "clCreateBuffer(bufSegMode)");
    gpu.carry_buffer_blocks = num_blocks;
    gpu.carry_buffer_segments = num_segments;
}

static bool should_use_parallel_apply(const GpuPrp& gpu, const CarryConfig& cfg) {
    if (gpu.n < (1u << 20)) return false;
    if (cfg.block_size < 512u) return false;
    if (cfg.num_blocks < 256u) return false;
    if (cfg.local_size < 16u) return false;
    return true;
}

static void enqueue_carry(GpuPrp& gpu, const CarryConfig& cfg) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    ensure_carry_buffers(gpu, cfg.num_blocks, cfg.local_size);

    check(clSetKernelArg(gpu.k_carry_block_local, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_local digits");
    check(clSetKernelArg(gpu.k_carry_block_local, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_local widths");
    check(clSetKernelArg(gpu.k_carry_block_local, 2, sizeof(cl_mem), &gpu.bufBlockCarry), "set carry_local blockcarry");
    check(clSetKernelArg(gpu.k_carry_block_local, 3, sizeof(cl_mem), &gpu.bufBlockValueLo), "set carry_local block_value_lo");
    check(clSetKernelArg(gpu.k_carry_block_local, 4, sizeof(cl_mem), &gpu.bufBlockBits), "set carry_local block_bits");
    check(clSetKernelArg(gpu.k_carry_block_local, 5, sizeof(cl_mem), &gpu.bufBlockThreshold), "set carry_local block_threshold");
    check(clSetKernelArg(gpu.k_carry_block_local, 6, sizeof(cl_mem), &gpu.bufBlockMode), "set carry_local block_mode");
    check(clSetKernelArg(gpu.k_carry_block_local, 7, sizeof(cl_mem), &gpu.bufSegValueLo), "set carry_local seg_value_lo");
    check(clSetKernelArg(gpu.k_carry_block_local, 8, sizeof(cl_mem), &gpu.bufSegBits), "set carry_local seg_bits");
    check(clSetKernelArg(gpu.k_carry_block_local, 9, sizeof(cl_mem), &gpu.bufSegThreshold), "set carry_local seg_threshold");
    check(clSetKernelArg(gpu.k_carry_block_local, 10, sizeof(cl_mem), &gpu.bufSegMode), "set carry_local seg_mode");
    check(clSetKernelArg(gpu.k_carry_block_local, 11, sizeof(cl_uint), &n_u32), "set carry_local n");
    check(clSetKernelArg(gpu.k_carry_block_local, 12, sizeof(cl_uint), &cfg.block_size), "set carry_local block_size");
    check(clSetKernelArg(gpu.k_carry_block_local, 13, sizeof(cl_uint), &cfg.items_per_worker), "set carry_local items_per_worker");
    check(clSetKernelArg(gpu.k_carry_block_local, 14, sizeof(cl_ulong) * cfg.block_size, nullptr), "set carry_local ldigits");
    check(clSetKernelArg(gpu.k_carry_block_local, 15, sizeof(cl_ulong) * cfg.local_size, nullptr), "set carry_local lseg_carry");
    check(clSetKernelArg(gpu.k_carry_block_local, 16, sizeof(cl_ulong) * cfg.local_size, nullptr), "set carry_local lseg_value_lo");
    check(clSetKernelArg(gpu.k_carry_block_local, 17, sizeof(cl_uint) * cfg.local_size, nullptr), "set carry_local lseg_bits");
    check(clSetKernelArg(gpu.k_carry_block_local, 18, sizeof(cl_ulong) * cfg.local_size, nullptr), "set carry_local lseg_threshold");
    check(clSetKernelArg(gpu.k_carry_block_local, 19, sizeof(cl_ulong) * cfg.local_size, nullptr), "set carry_local lseg_incoming");
    check(clSetKernelArg(gpu.k_carry_block_local, 20, sizeof(cl_uchar) * cfg.local_size, nullptr), "set carry_local lseg_mode");
    {
        const size_t global = std::size_t(cfg.num_blocks) * cfg.local_size;
        const size_t local = cfg.local_size;
        enqueue_kernel(gpu, gpu.k_carry_block_local, global, &local, "enqueue carry block local", "carry_block_local");
    }

    const bool use_chunked_prefix = (cfg.num_blocks >= 1024u && cfg.block_size >= 512u && gpu.max_work_group_size >= 64u);
    cl_kernel prefix_kernel = use_chunked_prefix ? gpu.k_carry_block_prefix_chunked64 : gpu.k_carry_block_prefix;
    check(clSetKernelArg(prefix_kernel, 0, sizeof(cl_mem), &gpu.bufBlockCarry), "set carry_prefix block_carry");
    check(clSetKernelArg(prefix_kernel, 1, sizeof(cl_mem), &gpu.bufBlockValueLo), "set carry_prefix block_value_lo");
    check(clSetKernelArg(prefix_kernel, 2, sizeof(cl_mem), &gpu.bufBlockBits), "set carry_prefix block_bits");
    check(clSetKernelArg(prefix_kernel, 3, sizeof(cl_mem), &gpu.bufBlockThreshold), "set carry_prefix block_threshold");
    check(clSetKernelArg(prefix_kernel, 4, sizeof(cl_mem), &gpu.bufBlockMode), "set carry_prefix block_mode");
    check(clSetKernelArg(prefix_kernel, 5, sizeof(cl_mem), &gpu.bufBlockIncoming), "set carry_prefix block_incoming");
    check(clSetKernelArg(prefix_kernel, 6, sizeof(cl_mem), &gpu.bufFinalCarry), "set carry_prefix final_carry");
    check(clSetKernelArg(prefix_kernel, 7, sizeof(cl_uint), &cfg.num_blocks), "set carry_prefix num_blocks");
    {
        const size_t global = use_chunked_prefix ? 64u : 1u;
        const size_t local = global;
        enqueue_kernel(gpu, prefix_kernel, global, &local, "enqueue carry block prefix", "carry_block_prefix");
    }

    if (should_use_parallel_apply(gpu, cfg)) {
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_apply digits");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_apply widths");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 2, sizeof(cl_mem), &gpu.bufBlockIncoming), "set carry_apply incoming");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 3, sizeof(cl_mem), &gpu.bufSegValueLo), "set carry_apply seg_value_lo");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 4, sizeof(cl_mem), &gpu.bufSegBits), "set carry_apply seg_bits");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 5, sizeof(cl_mem), &gpu.bufSegThreshold), "set carry_apply seg_threshold");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 6, sizeof(cl_mem), &gpu.bufSegMode), "set carry_apply seg_mode");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 7, sizeof(cl_uint), &n_u32), "set carry_apply n");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 8, sizeof(cl_uint), &cfg.block_size), "set carry_apply block_size");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 9, sizeof(cl_uint), &cfg.items_per_worker), "set carry_apply items_per_worker");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming, 10, sizeof(cl_ulong) * cfg.local_size, nullptr), "set carry_apply lseg_incoming");
        {
            const size_t global = std::size_t(cfg.num_blocks) * cfg.local_size;
            const size_t local = cfg.local_size;
            enqueue_kernel(gpu, gpu.k_carry_block_apply_incoming, global, &local, "enqueue carry block apply incoming", "carry_block_apply_parallel");
        }
    } else {
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming_serial, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_apply_serial digits");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming_serial, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_apply_serial widths");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming_serial, 2, sizeof(cl_mem), &gpu.bufBlockIncoming), "set carry_apply_serial incoming");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming_serial, 3, sizeof(cl_uint), &n_u32), "set carry_apply_serial n");
        check(clSetKernelArg(gpu.k_carry_block_apply_incoming_serial, 4, sizeof(cl_uint), &cfg.block_size), "set carry_apply_serial block_size");
        {
            const size_t global = std::size_t(cfg.num_blocks);
            enqueue_kernel(gpu, gpu.k_carry_block_apply_incoming_serial, global, nullptr, "enqueue carry block apply incoming serial", "carry_block_apply_serial");
        }
    }

    check(clSetKernelArg(gpu.k_carry_final_wrap, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_final digits");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_final widths");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 2, sizeof(cl_mem), &gpu.bufFinalCarry), "set carry_final carry");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 3, sizeof(cl_uint), &n_u32), "set carry_final n");
    {
        const size_t global = 1;
        enqueue_kernel(gpu, gpu.k_carry_final_wrap, global, &global, "enqueue carry final wrap", "carry_final_wrap");
    }
}

static void enqueue_mul_small(GpuPrp& gpu, cl_uint k) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_mul_small, 0, sizeof(cl_mem), &gpu.bufDigits), "set mul_small digits");
    check(clSetKernelArg(gpu.k_mul_small, 1, sizeof(cl_uint), &k), "set mul_small k");
    check(clSetKernelArg(gpu.k_mul_small, 2, sizeof(cl_uint), &n_u32), "set mul_small n");
    const size_t global = gpu.n;
    enqueue_kernel(gpu, gpu.k_mul_small, global, nullptr, "enqueue mul_small", "mul_small");
}

static void upload_digits(GpuPrp& gpu, const std::vector<std::uint64_t>& digits) {
    check(clEnqueueWriteBuffer(gpu.queue, gpu.bufDigits, CL_FALSE, 0, digits.size() * sizeof(std::uint64_t), digits.data(), 0, nullptr, nullptr),
          "write digits");
}

static std::vector<std::uint64_t> read_digits(GpuPrp& gpu) {
    std::vector<std::uint64_t> digits(gpu.n);
    check(clEnqueueReadBuffer(gpu.queue, gpu.bufDigits, CL_TRUE, 0, digits.size() * sizeof(std::uint64_t), digits.data(), 0, nullptr, nullptr),
          "read digits");
    return digits;
}

} // namespace clwrap

namespace mersenne_prp {

static bool prp_mersenne_pow2_base3_gpu(std::uint32_t p, bool verbose, clwrap::GpuPrp& gpu, const clwrap::CarryConfig& carry_cfg, cl_uint center_max = 0, std::uint32_t profile_every = 0) {
    if (p < 2) throw std::runtime_error("exponent must be >= 2");
    if (p == 2) return true;

    const ibdwt::Layout layout = ibdwt::make_layout(p);
    if (layout.n != gpu.n) throw std::runtime_error("layout/GPU size mismatch");

    const std::vector<std::uint64_t> init = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, init);

    const auto t0 = std::chrono::steady_clock::now();
    constexpr std::uint32_t report_interval = 1000;
    const std::uint32_t effective_profile_every = gpu.profile_kernels ? (profile_every ? profile_every : report_interval) : 0;
    for (std::uint32_t iter = 0; iter < p; ++iter) {
        clwrap::enqueue_square_mod(gpu, center_max);
        clwrap::enqueue_carry(gpu, carry_cfg);

        const bool do_report = verbose && ((iter + 1) % report_interval == 0 || iter + 1 == p);
        const bool do_profile_report = effective_profile_every && (((iter + 1) % effective_profile_every) == 0 || iter + 1 == p);
        const bool stop_requested = g_stop_requested.load(std::memory_order_relaxed);

        if (do_report || do_profile_report || stop_requested) {
            clwrap::check(clFinish(gpu.queue), stop_requested ? "clFinish(interrupt)" : "clFinish(progress)");
            clwrap::profile_flush_pending(gpu);
            const auto now = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(now - t0).count();
            const double rate = static_cast<double>(iter + 1) / std::max(sec, 1e-9);
            if (do_report || stop_requested) {
                std::cout << "iter " << (iter + 1) << "/" << p
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * double(iter + 1) / double(p)) << "%), elapsed "
                          << std::setprecision(2) << sec << " s, it/s " << std::setprecision(1) << rate << "\n";
            }
            if (do_profile_report || stop_requested) {
                std::ostringstream title;
                title << "Kernel profile summary at iter " << (iter + 1);
                clwrap::profile_print_summary(gpu, title.str());
            }
            if (stop_requested) throw InterruptedRun();
        } else if (((iter + 1) & 255u) == 0u) {
            clFlush(gpu.queue);
        }
    }

    clwrap::check(clFinish(gpu.queue), "clFinish(final)");
    clwrap::profile_flush_pending(gpu);
    std::vector<std::uint64_t> r = clwrap::read_digits(gpu);
    ibdwt::canonicalize_zero(r, layout);
    clwrap::profile_print_summary(gpu, "Kernel profile summary (final)");
    return ibdwt::equals_small(r, layout, 9);
}

static void selftest(const clwrap::DeviceInfo& dev, const std::string& kernel_path, cl_uint carry_block_override, cl_uint carry_items_override) {
    struct Case { std::uint32_t p; bool expect; };
    const std::vector<Case> cases = {
        {3, true}, {5, true}, {7, true}, {11, false}, {13, true}, {17, true},
        {19, true}, {23, false}, {31, true}, {61, true}, {89, true}, {107, true}, {127, true}
    };
    for (const auto& c : cases) {
        const auto layout = ibdwt::make_layout(c.p);
        auto gpu = clwrap::make_gpu(dev, kernel_path, layout, false);
        const auto carry_cfg = clwrap::choose_carry_config(dev, layout.n, carry_block_override, carry_items_override);
        const bool got = prp_mersenne_pow2_base3_gpu(c.p, false, gpu, carry_cfg, 0, 0);
        if (got != c.expect) {
            std::ostringstream oss;
            oss << "In-place fused GPU PRP self-test failed at p=" << c.p << ": got=" << got << ", expected=" << c.expect;
            throw std::runtime_error(oss.str());
        }
    }
}

} // namespace mersenne_prp

struct Options {
    std::uint32_t exponent = 0;
    bool verbose = true;
    bool selftest_only = false;
    int device_index = 0;
    std::string kernel_path = "gf61_ntt_prp_opencl_inplace_nobitrev_fused_powonly_blockcarry.cl";
    cl_uint carry_block = 0;
    cl_uint carry_items = 0;
    cl_uint center_max = 0;
    bool profile_kernels = false;
    bool planner_debug = false;
    std::uint32_t profile_every = 0;
};

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quiet") {
            opt.verbose = false;
        } else if (arg == "--selftest") {
            opt.selftest_only = true;
        } else if (arg == "--device") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --device");
            opt.device_index = std::stoi(argv[++i]);
        } else if (arg == "--kernel") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --kernel");
            opt.kernel_path = argv[++i];
        } else if (arg == "--carry-block") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --carry-block");
            opt.carry_block = static_cast<cl_uint>(std::stoul(argv[++i]));
        } else if (arg == "--carry-items") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --carry-items");
            opt.carry_items = static_cast<cl_uint>(std::stoul(argv[++i]));
        } else if (arg == "--center-max") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --center-max");
            opt.center_max = static_cast<cl_uint>(std::stoul(argv[++i]));
        } else if (arg == "--profile-kernels") {
            opt.profile_kernels = true;
        } else if (arg == "--planner-debug") {
            opt.planner_debug = true;
        } else if (arg == "--profile-every") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --profile-every");
            opt.profile_every = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  ./gf61_ntt_prp_opencl_inplace_nobitrev_fused_powonly <exponent_p> [--device k] [--quiet]\n"
                << "  ./gf61_ntt_prp_opencl_inplace_nobitrev_fused_powonly --selftest [--device k]\n\n"
                << "OpenCL correctness-first Mersenne PRP prototype in GF(M61^2) with IBDWT-style shifts.\n"
                << "This variant removes the standalone bit-reverse kernel.\n"
                << "Forward NTT uses radix-2 plus fused radix-4 DIF stages and inverse NTT uses radix-2 plus fused radix-4 DIT stages,\n"
                << "so the pipeline stays in-place and GPU-resident without a separate permutation pass.\n"
                << "Permanent tables stay on GPU. clFinish() is only used for progress checkpoints and final readback.\n"
                << "Optional tuning: --center-max {16,64,256,512,1024,2048,4096} caps the fused center kernel size.\n"
                << "Debugging: --profile-kernels prints a per-kernel timing summary using OpenCL profiling events.\n"
                << "           --profile-every N prints cumulative profiling every N iterations and on Ctrl+C.\n"
                << "           --planner-debug prints planner path decisions.\n";
            std::exit(0);
        } else if (!arg.empty() && arg[0] != '-') {
            opt.exponent = static_cast<std::uint32_t>(std::stoul(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const auto devices = clwrap::list_devices();
        if (opt.device_index < 0 || static_cast<std::size_t>(opt.device_index) >= devices.size()) {
            throw std::runtime_error("invalid device index");
        }
        const auto& dev = devices[static_cast<std::size_t>(opt.device_index)];
        std::cout << "Using OpenCL device: [" << opt.device_index << "] " << dev.name << "\n";

        if (opt.selftest_only) {
            const auto carry_cfg = clwrap::choose_carry_config(dev, ibdwt::make_layout(127u).n, opt.carry_block, opt.carry_items);
            std::cout << "carry config: block=" << carry_cfg.block_size << ", items/worker=" << carry_cfg.items_per_worker
                      << ", local=" << carry_cfg.local_size << ", blocks(for selftest max size)=" << carry_cfg.num_blocks << "\n";
            mersenne_prp::selftest(dev, opt.kernel_path, opt.carry_block, opt.carry_items);
            std::cout << "In-place OpenCL GF(M61^2) PRP self-tests: PASS\n";
            return 0;
        }

        if (opt.exponent == 0) throw std::runtime_error("missing exponent p");
        const auto layout = ibdwt::make_layout(opt.exponent);
        std::signal(SIGINT, handle_sigint);
        g_stop_requested.store(false, std::memory_order_relaxed);
        auto gpu = clwrap::make_gpu(dev, opt.kernel_path, layout, opt.profile_kernels);
        clwrap::g_planner_debug = opt.planner_debug;

        std::cout << "p=" << opt.exponent << ", ln=" << layout.ln
                  << ", transform=" << layout.n << " (in-place field buffer, no bit-reverse kernel)\n";
        const auto run_carry_cfg = clwrap::choose_carry_config(dev, layout.n, opt.carry_block, opt.carry_items);
        std::cout << "carry config: block=" << run_carry_cfg.block_size << ", items/worker=" << run_carry_cfg.items_per_worker
                  << ", local=" << run_carry_cfg.local_size << ", blocks=" << run_carry_cfg.num_blocks << "\n";
        const bool prp = mersenne_prp::prp_mersenne_pow2_base3_gpu(opt.exponent, opt.verbose, gpu, run_carry_cfg, opt.center_max, opt.profile_every);
        std::cout << "2^" << opt.exponent << " - 1 is " << (prp ? "PRP" : "composite") << "\n";
        return prp ? 0 : 1;
    } catch (const InterruptedRun&) {
        std::cerr << "Run interrupted.\n";
        return 130;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
