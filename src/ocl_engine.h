#pragma once

// OpenCL modular multiplication engine.
//
// Two reduction modes:
//  * k*2^n+c (small k, small |c|): special "fold" reduction, one FFT product per operation.
//    A slot holding z represents x = k^2 * z (mod N), or x = k * z for k*2^n - 1 with the IBDWT.
//  * any odd N: Montgomery reduction with R = 2^mont_bits(), three FFT products per operation.
//    A slot holding z represents x = z * R^-1 (mod N). Call set_mont_inverse() before use.
// Callers convert: upload z = x * scale^-1 mod N, download x = z * scale mod N
// (scale = k^2 or R^-1). The engine itself only moves raw z values.

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace gpu
{
    struct DeviceInfo
    {
        int index;
        std::string name;
        std::string platform;
        bool fp64;          // false: hybrid engine only
        bool is_gpu;
    };

    class Engine
    {
    public:
        // device: index from list_devices(), or -1 to pick the first GPU with FP64 (else the first GPU).
        // Without FP64 only the hybrid FP32 + NTT fold engine runs; numbers that need the FP64 FFT
        // (Montgomery, small lengths, digits the fused carry does not cover) throw.
        // bits_reduce: lower the digit size by this many bits (used after round-off errors).
        Engine(uint32_t k, uint32_t n, int32_t c, int device = -1, int bits_reduce = 0);
        // k*b^n+c for b >= 3 (not a power of two): digits in base B = b^e, fold reduction.
        // Use set_digits()/get_digits(); a slot holding z represents x = k^2 * z (mod N).
        struct BaseB { };   // tag: keeps this overload distinct from Engine(k, n, c, device)
        Engine(BaseB, uint32_t k, uint32_t b, uint32_t n, int32_t c, int device = -1, int bits_reduce = 0);
        // Montgomery mode for an arbitrary odd modulus (little-endian uint32 limbs). mmgw: allow
        // R = 2^K - 1 (cyclic/negacyclic products, hybrid engine only), see mont_cyclic().
        Engine(const uint32_t* N, int size, int device = -1, int bits_reduce = 0, bool mmgw = true);
        ~Engine();
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;

        static std::vector<DeviceInfo> list_devices();
        // Global device index of the index-th GPU whose platform matches vendor
        // (BOINC gpu_type: "ATI"/"AMD", "NVIDIA", "intel_gpu", "apple_gpu"; empty = any). -1 if not found.
        static int find_gpu(const std::string& vendor, int index);

        int alloc_slot();
        // Set slot to a non-negative value (little-endian uint32 limbs), value < 2*k*2^n.
        void set(int slot, const uint32_t* data, int size);
        // Read slot value as sign + magnitude (little-endian uint32 limbs). Returns true if negative.
        bool get(int slot, std::vector<uint32_t>& magnitude);
        // Digit-level access (any mode): balanced little-endian digits in base digit_base().
        int64_t digit_base() const;
        bool binary_digits() const { return _b == 2 || _mont; }
        void set_digits(int slot, const std::vector<int64_t>& digits);
        void get_digits(int slot, std::vector<int64_t>& digits);
        uint32_t digits_capacity() const { return _D; }
        void copy(int dst, int src);
        // Montgomery mode: R = 2^mont_bits(), or 2^mont_bits() - 1 if mont_cyclic();
        // set -N^-1 mod R (little-endian uint32 limbs). N and R must be coprime.
        bool montgomery() const { return _mont; }
        uint32_t mont_bits() const { return _q*_w; }
        bool mont_cyclic() const { return _mmgw; }
        // k*2^n - 1 with the IBDWT (no zero padding): a slot holding z represents x = k*z (mod N)
        // instead of k^2*z. scale_power(): the power of k in that factor (fold engines).
        bool ibdwt() const { return _ibdwt; }
        int scale_power() const { return _ibdwt ? 1 : 2; }
        void set_mont_inverse(const uint32_t* data, int size);

        // slot = slot^2 * mulc (repeated count times). mulc is a small constant (< 2^13).
        void square(int slot, int count = 1, uint32_t mulc = 1);
        void mul(int dst, int src);          // dst = dst*src
        // Fixed multiplier: transform(spec, src) keeps the forward FFT of src in slot spec (which
        // then no longer holds a value); mul_transformed(dst, spec) = dst*src at the cost of a squaring.
        void transform(int spec, int src);
        void mul_transformed(int dst, int spec);
        // Waits for the queue. Returns max round-off since last reset (0..0.5).
        double max_roundoff(bool reset = true);
        // Waits for the queue. True if a carry propagation failure was detected since last reset.
        bool carry_error(bool reset = true);
        // Asynchronous download: snapshot(a, b) enqueues non-blocking reads of slots a and b (b < 0:
        // none) and of the error flags, which are then reset; later GPU work can be queued right away.
        // snapshot_wait() blocks until the data is on the host (thread safe). The snapshot_* getters
        // only touch host memory, so a worker thread can convert while the GPU keeps running.
        void snapshot(int a, int b);
        void snapshot_wait();
        double snapshot_roundoff() const;
        bool snapshot_carry_error() const;
        void snapshot_digits(int k, std::vector<int64_t>& digits) const;
        bool snapshot_get(int k, std::vector<uint32_t>& magnitude) const;
        void finish();

        std::string description() const;
        struct Impl;
        uint32_t fft_complex_len() const { return _D; }
        uint32_t digit_bits() const { return _w; }

    private:
        void init_device(int device, int bits_reduce);
        void to_digits(const uint32_t* data, int size, std::vector<double>& digits);
        void finish_product(void* dst, uint32_t mulc);
        void mmgw_product(void* x, void* spec, void* dst, uint32_t mulc);
        void ib_product(void* x, void* spec, void* dst, uint32_t mulc);
        bool ib_setup(uint32_t k, uint32_t n, int device, int bits_reduce);
        void ib_to_digits(const uint32_t* data, int size, std::vector<double>& digits) const;
        bool magnitude(const std::vector<double>& digits, std::vector<uint32_t>& magnitude) const;

        std::unique_ptr<Impl> _impl;
        bool _mont = false;
        bool _mmgw = false;   // Montgomery with R = 2^K - 1, Q = 2^K + 1
        bool _ibdwt = false;  // k*2^n - 1: weighted transform, L = 2D digits of q0 or q0 + 1 bits
        uint32_t _ibq0 = 0, _ibr = 0;   // n = q0*L + r
        uint32_t _b = 2, _e = 0;                  // digit base B = b^e
        uint32_t _fq = 0;                         // fold: b^n = B^_fq * _fbs
        int64_t _fbs = 1, _fbmul = 1;
        uint32_t _q = 0;      // Montgomery: digits in R
        uint32_t _k, _n;
        int32_t _c;
        uint32_t _D = 0;      // complex FFT length (real length 2D)
        uint32_t _w = 0;      // bits per digit
        uint32_t _nd = 0;     // digits needed for a reduced value
    };
}
