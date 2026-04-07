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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
            out.push_back(DeviceInfo{plat, dev, name});
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

struct GpuPrp {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel k_weight_first_stage_dif = nullptr;
    cl_kernel k_ntt_stage_dif = nullptr;
    cl_kernel k_ntt_stage_dit = nullptr;
    cl_kernel k_last_stage_dit_unweight = nullptr;
    cl_kernel k_pointwise_sqr = nullptr;
    cl_kernel k_mul_small = nullptr;
    cl_kernel k_carry = nullptr;

    cl_mem bufDigits = nullptr;
    cl_mem bufField = nullptr;
    cl_mem bufShift = nullptr;
    cl_mem bufWidth = nullptr;
    cl_mem bufTwFwd = nullptr;
    cl_mem bufTwInv = nullptr;

    std::size_t n = 0;
    std::vector<StageInfo> stages;
    gf61::Elem inv_n_factor{};

    ~GpuPrp() {
        if (bufTwInv) clReleaseMemObject(bufTwInv);
        if (bufTwFwd) clReleaseMemObject(bufTwFwd);
        if (bufWidth) clReleaseMemObject(bufWidth);
        if (bufShift) clReleaseMemObject(bufShift);
        if (bufField) clReleaseMemObject(bufField);
        if (bufDigits) clReleaseMemObject(bufDigits);
        if (k_carry) clReleaseKernel(k_carry);
        if (k_mul_small) clReleaseKernel(k_mul_small);
        if (k_pointwise_sqr) clReleaseKernel(k_pointwise_sqr);
        if (k_last_stage_dit_unweight) clReleaseKernel(k_last_stage_dit_unweight);
        if (k_ntt_stage_dit) clReleaseKernel(k_ntt_stage_dit);
        if (k_ntt_stage_dif) clReleaseKernel(k_ntt_stage_dif);
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

GpuPrp make_gpu(const DeviceInfo& info, const std::string& kernel_path, const ibdwt::Layout& layout) {
    GpuPrp gpu;
    cl_int err = CL_SUCCESS;
    gpu.context = clCreateContext(nullptr, 1, &info.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    gpu.queue = clCreateCommandQueue(gpu.context, info.device, 0, &err);
    check(err, "clCreateCommandQueue");

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
    make_kernel("gf61_ntt_stage_dif", &gpu.k_ntt_stage_dif);
    make_kernel("gf61_ntt_stage_dit", &gpu.k_ntt_stage_dit);
    make_kernel("gf61_last_stage_dit_unweight", &gpu.k_last_stage_dit_unweight);
    make_kernel("gf61_pointwise_sqr", &gpu.k_pointwise_sqr);
    make_kernel("gf61_mul_small_digits", &gpu.k_mul_small);
    make_kernel("gf61_carry_normalize_serial", &gpu.k_carry);

    gpu.n = layout.n;
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

    gpu.bufTwFwd = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_fwd.size() * sizeof(gf61::Elem), tw_fwd.data(), &err);
    check(err, "clCreateBuffer(bufTwFwd)");
    gpu.bufTwInv = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_inv.size() * sizeof(gf61::Elem), tw_inv.data(), &err);
    check(err, "clCreateBuffer(bufTwInv)");

    const std::uint64_t inv_n_scalar = gf61::pow_mod(static_cast<std::uint64_t>(n % gf61::P), gf61::P - 2);
    gpu.inv_n_factor = gf61::Elem(inv_n_scalar, 0);

    return gpu;
}

static void enqueue_forward_pipeline(GpuPrp& gpu) {
    const StageInfo& first = gpu.stages.back();

    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 0, sizeof(cl_mem), &gpu.bufDigits), "set weight_first digits");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 1, sizeof(cl_mem), &gpu.bufShift), "set weight_first shifts");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 2, sizeof(cl_mem), &gpu.bufField), "set weight_first field");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 3, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_first tw");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 4, sizeof(cl_uint), &first.offset), "set weight_first off");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 5, sizeof(cl_uint), &first.len), "set weight_first len");
    check(clSetKernelArg(gpu.k_weight_first_stage_dif, 6, sizeof(cl_uint), &first.half_len), "set weight_first half");
    {
        const size_t global = gpu.n / 2;
        check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_weight_first_stage_dif, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
              "enqueue weight_first_stage_dif");
    }

    for (auto it = gpu.stages.rbegin() + 1; it != gpu.stages.rend(); ++it) {
        const StageInfo& st = *it;
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif a");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 2, sizeof(cl_uint), &st.offset), "set ntt_stage_dif off");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 3, sizeof(cl_uint), &st.len), "set ntt_stage_dif len");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 4, sizeof(cl_uint), &st.half_len), "set ntt_stage_dif half");
        const size_t global = gpu.n / 2;
        check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_ntt_stage_dif, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
              "enqueue ntt_stage_dif fwd");
    }
}

static void enqueue_inverse_pipeline(GpuPrp& gpu) {
    const StageInfo& last = gpu.stages.back();
    for (std::size_t idx = 0; idx + 1 < gpu.stages.size(); ++idx) {
        const StageInfo& st = gpu.stages[idx];
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit a");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 2, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit off");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 3, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit len");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 4, sizeof(cl_uint), &st.half_len), "set inv_ntt_stage_dit half");
        const size_t global = gpu.n / 2;
        check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_ntt_stage_dit, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
              "enqueue ntt_stage_dit inv");
    }

    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 0, sizeof(cl_mem), &gpu.bufField), "set last_stage a");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 1, sizeof(cl_mem), &gpu.bufTwInv), "set last_stage tw");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 2, sizeof(cl_mem), &gpu.bufShift), "set last_stage shifts");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 3, sizeof(cl_mem), &gpu.bufDigits), "set last_stage digits");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 4, sizeof(gf61::Elem), &gpu.inv_n_factor), "set last_stage invn");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 5, sizeof(cl_uint), &last.offset), "set last_stage off");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 6, sizeof(cl_uint), &last.len), "set last_stage len");
    check(clSetKernelArg(gpu.k_last_stage_dit_unweight, 7, sizeof(cl_uint), &last.half_len), "set last_stage half");
    {
        const size_t global = gpu.n / 2;
        check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_last_stage_dit_unweight, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
              "enqueue last_stage_dit_unweight");
    }
}

static void enqueue_square_mod(GpuPrp& gpu) {
    enqueue_forward_pipeline(gpu);
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_pointwise_sqr, 0, sizeof(cl_mem), &gpu.bufField), "set pointwise_sqr a");
    check(clSetKernelArg(gpu.k_pointwise_sqr, 1, sizeof(cl_uint), &n_u32), "set pointwise_sqr n");
    {
        const size_t global = gpu.n;
        check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_pointwise_sqr, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
              "enqueue pointwise_sqr");
    }
    enqueue_inverse_pipeline(gpu);
}

static void enqueue_carry(GpuPrp& gpu) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_carry, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry digits");
    check(clSetKernelArg(gpu.k_carry, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry widths");
    check(clSetKernelArg(gpu.k_carry, 2, sizeof(cl_uint), &n_u32), "set carry n");
    const size_t global = 1;
    check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_carry, 1, nullptr, &global, &global, 0, nullptr, nullptr),
          "enqueue carry");
}

static void enqueue_mul_small(GpuPrp& gpu, cl_uint k) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_mul_small, 0, sizeof(cl_mem), &gpu.bufDigits), "set mul_small digits");
    check(clSetKernelArg(gpu.k_mul_small, 1, sizeof(cl_uint), &k), "set mul_small k");
    check(clSetKernelArg(gpu.k_mul_small, 2, sizeof(cl_uint), &n_u32), "set mul_small n");
    const size_t global = gpu.n;
    check(clEnqueueNDRangeKernel(gpu.queue, gpu.k_mul_small, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
          "enqueue mul_small");
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

static bool prp_mersenne_base3_gpu(std::uint32_t p, bool verbose, clwrap::GpuPrp& gpu) {
    if (p < 2) throw std::runtime_error("exponent must be >= 2");
    if (p == 2) return true;

    const ibdwt::Layout layout = ibdwt::make_layout(p);
    if (layout.n != gpu.n) throw std::runtime_error("layout/GPU size mismatch");

    const std::vector<std::uint64_t> init = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, init);

    const auto t0 = std::chrono::steady_clock::now();
    constexpr std::uint32_t report_interval = 1000;
    for (std::uint32_t iter = 0; iter < p - 1; ++iter) {
        clwrap::enqueue_square_mod(gpu);
        clwrap::enqueue_carry(gpu);
        clwrap::enqueue_mul_small(gpu, 3u);
        clwrap::enqueue_carry(gpu);

        if (verbose && ((iter + 1) % report_interval == 0 || iter + 1 == p - 1)) {
            clwrap::check(clFinish(gpu.queue), "clFinish(progress)");
            const auto now = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(now - t0).count();
            const double rate = static_cast<double>(iter + 1) / std::max(sec, 1e-9);
            std::cout << "iter " << (iter + 1) << "/" << (p - 1)
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * double(iter + 1) / double(p - 1)) << "%), elapsed "
                      << std::setprecision(2) << sec << " s, it/s " << std::setprecision(1) << rate << "\n";
        } else if (((iter + 1) & 255u) == 0u) {
            clFlush(gpu.queue);
        }
    }

    clwrap::check(clFinish(gpu.queue), "clFinish(final)");
    std::vector<std::uint64_t> r = clwrap::read_digits(gpu);
    ibdwt::canonicalize_zero(r, layout);
    return ibdwt::equals_small(r, layout, 3);
}

static void selftest(const clwrap::DeviceInfo& dev, const std::string& kernel_path) {
    struct Case { std::uint32_t p; bool expect; };
    const std::vector<Case> cases = {
        {3, true}, {5, true}, {7, true}, {11, false}, {13, true}, {17, true},
        {19, true}, {23, false}, {31, true}, {61, true}, {89, true}, {107, true}, {127, true}
    };
    for (const auto& c : cases) {
        const auto layout = ibdwt::make_layout(c.p);
        auto gpu = clwrap::make_gpu(dev, kernel_path, layout);
        const bool got = prp_mersenne_base3_gpu(c.p, false, gpu);
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
    std::string kernel_path = "gf61_ntt_prp_opencl_inplace_nobitrev_fused.cl";
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  ./gf61_ntt_prp_opencl_inplace_nobitrev_fused <exponent_p> [--device k] [--quiet]\n"
                << "  ./gf61_ntt_prp_opencl_inplace_nobitrev_fused --selftest [--device k]\n\n"
                << "OpenCL correctness-first Mersenne PRP prototype in GF(M61^2) with IBDWT-style shifts.\n"
                << "This variant removes the standalone bit-reverse kernel.\n"
                << "Forward NTT uses radix-2 DIF stages and inverse NTT uses radix-2 DIT stages,\n"
                << "so the pipeline stays in-place and GPU-resident without a separate permutation pass.\n"
                << "Permanent tables stay on GPU. clFinish() is only used for progress checkpoints and final readback.\n";
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
            mersenne_prp::selftest(dev, opt.kernel_path);
            std::cout << "In-place OpenCL GF(M61^2) PRP self-tests: PASS\n";
            return 0;
        }

        if (opt.exponent == 0) throw std::runtime_error("missing exponent p");
        const auto layout = ibdwt::make_layout(opt.exponent);
        auto gpu = clwrap::make_gpu(dev, opt.kernel_path, layout);

        std::cout << "p=" << opt.exponent << ", ln=" << layout.ln
                  << ", transform=" << layout.n << " (in-place field buffer, no bit-reverse kernel)\n";
        const bool prp = mersenne_prp::prp_mersenne_base3_gpu(opt.exponent, opt.verbose, gpu);
        std::cout << "2^" << opt.exponent << " - 1 is " << (prp ? "PRP" : "composite") << "\n";
        return prp ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
