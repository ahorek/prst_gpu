#pragma once

// Gerbicz-checked exponentiation X = X0^(2^n) mod k*2^n+c on an OpenCL device.
// Drop-in for StrongCheckMultipointExp with smooth base 2: same points (incl. proof points
// and on_point callbacks), same states and checkpoints, so a run can resume on CPU or GPU.

#ifdef OPENCL

#include <memory>
#include <string>
#include <future>
#include "exp.h"
#include "ocl_engine.h"

struct GPUBenchmarkResult
{
    std::string description;
    size_t bits = 0;
    int ops = 0;
    double ms_per_op = 0;
    double roundoff = 0;
    bool carry_error = false;
};
bool gpu_benchmark(InputNum& input, int device, int reduce, double seconds, int min_ops, GPUBenchmarkResult& res, std::string& reason);

class GPUStrongCheckExp : public StrongCheckMultipointExp
{
public:
    // smooth: exp is the base b (2), X = X0^(b^n). Otherwise exp is an arbitrary exponent, X = x0^exp.
    GPUStrongCheckExp(const arithmetic::Giant& exp, bool smooth, const std::vector<Point>& points, int L, int L2, std::function<bool(int, State*, Logging&)> on_point, int device);

    // Same points and L/L2 as GerbiczCheckExp(b, n, count, nullptr, L).
    static GPUStrongCheckExp* gerbicz(arithmetic::Giant& b, int n, int count, int L, int device)
    {
        GerbiczCheckExp tmp(b, n, count, nullptr, L);
        return new GPUStrongCheckExp(b, true, tmp.points(), tmp._L, tmp._L2, nullptr, device);
    }

    // Same points and L/L2 as FastLiCheckExp(exp, count, L): x0^exp for a small x0 (Pocklington, generic Fermat).
    static GPUStrongCheckExp* li(const arithmetic::Giant& exp, int count, int L, int device)
    {
        arithmetic::Giant e;
        e = exp;
        FastLiCheckExp tmp(std::move(e), count, L);
        return new GPUStrongCheckExp(exp, false, tmp.points(), tmp._L, tmp._L2, nullptr, device);
    }

    // Same points and L/L2 as LiCheckExp(exp, count, L): X0^exp for a giant X0 set with init_giant
    // (certificate verification of Pocklington/Li proofs).
    static GPUStrongCheckExp* li_giant(const arithmetic::Giant& exp, int count, int L, int device)
    {
        arithmetic::Giant e;
        e = exp;
        LiCheckExp tmp(std::move(e), count, L);
        return new GPUStrongCheckExp(exp, false, tmp.points(), tmp._L, tmp._L2, nullptr, device);
    }

    ~GPUStrongCheckExp();

    // True if the fast k*2^n+c engine supports N = input (small k and c). Other numbers use Montgomery.
    static bool supported(InputNum& input, std::string& reason);

protected:
    void execute() override;
    void release() override;

private:
    void engine_init();
    void skip_small_start(int len);
    void upload(int slot, arithmetic::Giant& value);
    void download(int slot, arithmetic::Giant& value);
    void check_errors(const char* where);
    void power_step(int slot, bool bit);
    // Asynchronous checkpoint: the GPU keeps running while a worker converts a snapshot of X and D.
    void async_start(int iteration, int recovery);
    void async_finish(bool wait);       // wait == false: only if the worker is done
    void async_discard();
    struct Async
    {
        bool active = false;
        int iteration = 0, recovery = 0;
        std::future<void> done;
    };
    Async _async;

    int _device;
    int _bits_reduce = 0;
    double _last_loop_err = 0;      // max round-off of the last check_errors (PRST_GPU_ERRSTAT)
    std::unique_ptr<gpu::Engine> _engine;
    struct Mpz;
    std::unique_ptr<Mpz> _mpz;      // GMP copies of N and the slot scale factors
    int _sX = -1, _sD = -1, _sR = -1, _sT = -1, _sU = -1, _sB = -1, _sBF = -1;
};

#endif
