#ifdef OPENCL

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include "gwnum.h"
#include "cpuid.h"
#include "gpu_exp.h"
#include "exception.h"
#include <gmp.h>
#include "gmp_digits.h"

using namespace arithmetic;

// Host-side big-number math for the GPU path uses GMP directly (prst always links it), not the
// default Giant arithmetic: without libgmp-gw1.dll that falls back to gwnum giants, which cannot
// invert modulo 2^m and is far too slow at tens of millions of bits.
struct GPUStrongCheckExp::Mpz
{
    mpz_t N, scale, scale_inv, t;
    std::unique_ptr<gpu::DigitConverter> conv;   // non-binary digit base (k*b^n+c, b >= 3)
    // asynchronous checkpoint: results of the worker (X, D) and its own converter
    mpz_t ax, ad;
    std::unique_ptr<gpu::DigitConverter> aconv;
    Mpz() { mpz_init(N); mpz_init(scale); mpz_init(scale_inv); mpz_init(t); mpz_init(ax); mpz_init(ad); }
    ~Mpz() { mpz_clear(N); mpz_clear(scale); mpz_clear(scale_inv); mpz_clear(t); mpz_clear(ax); mpz_clear(ad); }
};

static void giant_to_mpz(mpz_t r, const Giant& g)
{
    if (g.size() == 0)
        mpz_set_ui(r, 0);
    else
        mpz_import(r, g.size(), -1, 4, 0, 0, g.data());
    if (g.arithmetic().cmp(g, 0) < 0)
        mpz_neg(r, r);
}

// N == k*b^n+c with k < 2^31 and |c| < 2^20 (after moving factors of b out of k, and
// rewriting power-of-two bases as base 2): the fold engines apply.
static bool fold_form(InputNum& input, const mpz_t N, uint32_t& k, uint32_t& b, uint32_t& n, int32_t& c)
{
    if (input.type() != InputNum::KBNC)
        return false;
    int64_t cc = input.c();
    uint64_t bb = input.b();
    if (cc == 0 || std::llabs(cc) >= (1 << 20) || bb < 2 || bb >= (1u << 16) || input.gk().bitlen() > 64)
        return false;
    uint64_t nn = input.n();
    if ((bb & (bb - 1)) == 0)
    {
        int t = 0;
        while (((uint64_t)1 << t) < bb) t++;
        nn *= t;
        bb = 2;
    }
    mpz_t kk, v;
    mpz_init(kk);
    mpz_init(v);
    giant_to_mpz(kk, input.gk());
    while (mpz_sgn(kk) > 0 && mpz_divisible_ui_p(kk, (unsigned long)bb))
    {
        mpz_divexact_ui(kk, kk, (unsigned long)bb);
        nn++;
    }
    bool ok = mpz_sgn(kk) > 0 && mpz_sizeinbase(kk, 2) <= 31 && nn < (1u << 31);
    if (ok)
    {
        k = (uint32_t)mpz_get_ui(kk);
        b = (uint32_t)bb;
        n = (uint32_t)nn;
        c = (int32_t)cc;
        mpz_ui_pow_ui(v, b, n);
        mpz_mul_ui(v, v, k);
        if (c > 0)
            mpz_add_ui(v, v, (unsigned long)c);
        else
            mpz_sub_ui(v, v, (unsigned long)(-c));
        ok = mpz_cmp(v, N) == 0;
    }
    mpz_clear(kk);
    mpz_clear(v);
    return ok;
}

static void mpz_to_limbs(const mpz_t a, std::vector<uint32_t>& v)
{
    v.assign((mpz_sizeinbase(a, 2) + 31)/32 + 1, 0);
    size_t written = 0;
    mpz_export(v.data(), &written, -1, 4, 0, 0, a);
    v.resize(written);
}

static bool reduce_k(Giant& k, uint32_t& n)
{
    if (k == 0)
        return false;
    while (!k.bit(0))
    {
        k.arithmetic().shiftright(k, 1, k);
        n++;
    }
    return true;
}

// Montgomery engine for odd N, R in *R: cyclic/negacyclic (R = 2^K - 1) where it applies and N is
// prime to R, else R = 2^K.
static gpu::Engine* mont_engine(const mpz_t N, int device, int reduce, mpz_t R)
{
    std::vector<uint32_t> limbs;
    mpz_to_limbs(N, limbs);
    for (int mm = 1; ; mm = 0)
    {
        std::unique_ptr<gpu::Engine> e;
        try
        {
            e.reset(new gpu::Engine(limbs.data(), (int)limbs.size(), device, reduce, mm != 0));
        }
        catch (const std::exception&)
        {
            if (mm == 0)
                throw;
            continue;
        }
        mpz_set_ui(R, 0);
        mpz_setbit(R, e->mont_bits());
        if (e->mont_cyclic())
            mpz_sub_ui(R, R, 1);
        mpz_t g;
        mpz_init(g);
        mpz_gcd(g, N, R);
        bool ok = mpz_cmp_ui(g, 1) == 0;
        mpz_clear(g);
        if (ok || mm == 0)
            return e.release();
    }
}

bool GPUStrongCheckExp::supported(InputNum& input, std::string& reason)
{
    if (input.type() != InputNum::KBNC || input.b() != 2)
    {
        reason = "only k*2^n+c numbers are supported";
        return false;
    }
    Giant k;
    k = input.gk();
    uint32_t n = input.n();
    if (!reduce_k(k, n) || k.bitlen() > 31)
    {
        reason = "k must be below 2^31";
        return false;
    }
    int64_t c = input.c();
    if (c == 0 || std::llabs(c) >= (1 << 20))
    {
        reason = "|c| must be between 1 and 2^20";
        return false;
    }
    uint64_t a = k.data()[0], b = (uint64_t)std::llabs(c);
    while (b != 0) { uint64_t t = a%b; a = b; b = t; }
    if (a != 1)
    {
        reason = "k and c must be coprime";
        return false;
    }
    return true;
}

// Benchmark: squarings of a random residue in the engine prst would pick for N.
// Runs at least min_ops and at least `seconds`. Returns false (with reason) if no engine fits.
bool gpu_benchmark(InputNum& input, int device, int reduce, double seconds, int min_ops, GPUBenchmarkResult& res, std::string& reason)
{
    try
    {
        mpz_t N, r, t;
        mpz_init(N); mpz_init(r); mpz_init(t);
        struct Z { mpz_t v; Z() { mpz_init(v); } ~Z() { mpz_clear(v); } } Rinv, ea, eb;   // Montgomery self-test reference
        Giant value = input.value();
        giant_to_mpz(N, value);
        if (mpz_even_p(N))
        {
            mpz_clear(N); mpz_clear(r); mpz_clear(t);
            reason = "N is even";
            return false;
        }
        std::unique_ptr<gpu::Engine> engine;
        std::vector<uint32_t> limbs;
        uint32_t fk = 0, fb = 0, fn = 0;
        int32_t fc = 0;
        if (fold_form(input, N, fk, fb, fn, fc))
        {
            if (fb == 2)
                engine.reset(new gpu::Engine(fk, fn, fc, device, reduce));
            else
                engine.reset(new gpu::Engine(gpu::Engine::BaseB(), fk, fb, fn, fc, device, reduce));
            // self-test reference: a product of slots is x*y*k^scale_power
            mpz_set_ui(Rinv.v, fk);
            if (engine->scale_power() == 2)
                mpz_mul_ui(Rinv.v, Rinv.v, fk);
        }
        else
        {
            engine.reset(mont_engine(N, device, reduce, t));
            mpz_mod(r, t, N);
            mpz_invert(Rinv.v, r, N);
            mpz_invert(r, N, t);
            mpz_sub(r, t, r);
            mpz_to_limbs(r, limbs);
            engine->set_mont_inverse(limbs.data(), (int)limbs.size());
        }
        gmp_randstate_t rs;
        gmp_randinit_default(rs);
        gmp_randseed_ui(rs, 12345);
        mpz_urandomm(r, rs, N);
        gmp_randclear(rs);
        // PRST_GPU_CHECK_X=<v>: start from the slot of the small value v (z = v/k^2 mod N), as a test does
        if (getenv("PRST_GPU_CHECK_X") != nullptr && fk > 0)
        {
            mpz_set_ui(t, fk);
            if (engine->scale_power() == 2)
                mpz_mul_ui(t, t, fk);
            mpz_invert(t, t, N);
            mpz_mul_ui(r, t, (unsigned long)atoi(getenv("PRST_GPU_CHECK_X")));
            mpz_mod(r, r, N);
        }
        int slot = engine->alloc_slot();
        if (engine->binary_digits())
        {
            mpz_to_limbs(r, limbs);
            engine->set(slot, limbs.data(), (int)limbs.size());
        }
        else
        {
            gpu::DigitConverter conv(engine->digit_base());
            std::vector<int64_t> digits;
            conv.to_digits(r, digits);
            engine->set_digits(slot, digits);
        }
        res.description = engine->description();
        res.bits = mpz_sizeinbase(N, 2);
        // PRST_GPU_CHECK=<n>: self-test instead of timing. A fixed sequence of operations (n squarings,
        // squarings with a multiplier, mul, mul_transformed); prints the low 64 bits of the result
        // mod N, to compare engine versions and devices against each other.
        if (const char* chk = getenv("PRST_GPU_CHECK"))
        {
            int cnt = atoi(chk);
            int s2 = engine->alloc_slot(), s3 = engine->alloc_slot();
            if (getenv("PRST_GPU_CHECK_SQ") != nullptr && !engine->montgomery())
            {
                // squarings only, against GMP: a slot holding z stands for k^2*z, so z' == k^2*z^2 (mod N)
                // (IBDWT: k*z, z' == k*z^2)
                mpz_t e;
                mpz_init_set(e, r);
                for (int i = 0; i < cnt; i++)
                {
                    mpz_mul(e, e, e);
                    mpz_mul_ui(e, e, fk);
                    if (engine->scale_power() == 2)
                        mpz_mul_ui(e, e, fk);
                    mpz_mod(e, e, N);
                }
                engine->square(slot, cnt);
                if (engine->binary_digits())
                {
                    bool neg = engine->get(slot, limbs);
                    mpz_import(r, limbs.size(), -1, 4, 0, 0, limbs.data());
                    if (neg)
                        mpz_neg(r, r);
                }
                else
                {
                    gpu::DigitConverter conv(engine->digit_base());
                    std::vector<int64_t> digits;
                    engine->get_digits(slot, digits);
                    conv.from_digits(digits, r);
                }
                mpz_mod(r, r, N);
                bool ce = engine->carry_error();
                printf("  CHECK_SQ %d: %s%s\n", cnt, mpz_cmp(r, e) == 0 ? "ok" : "MISMATCH", ce ? "  CARRY ERROR" : "");
                mpz_clear(e);
            }
            // the same sequence on the host (each product times R^-1, or k^2 / k), compared below
            uint32_t cm = getenv("PRST_GPU_CHECK_M") ? (uint32_t)atoi(getenv("PRST_GPU_CHECK_M")) : 3;
            mpz_set(ea.v, r);
            {
                auto mm = [&](mpz_t d, mpz_t a, mpz_t b, uint32_t m) { mpz_mul(d, a, b); mpz_mul_ui(d, d, m); mpz_mul(d, d, Rinv.v); mpz_mod(d, d, N); };
                for (int i = 0; i < cnt; i++) mm(ea.v, ea.v, ea.v, 1);
                mpz_set(eb.v, ea.v);
                for (int i = 0; i < 3; i++) mm(eb.v, eb.v, eb.v, cm);
                mm(ea.v, ea.v, eb.v, 1);
                mm(ea.v, ea.v, eb.v, 1);
                for (int i = 0; i < 2; i++) mm(ea.v, ea.v, ea.v, 1);
            }
            engine->square(slot, cnt);
            engine->copy(s2, slot);
            // PRST_GPU_CHECK_M=<m>: the multiplier of the squarings with a multiplier (default 3)
            engine->square(s2, 3, cm);
            engine->mul(slot, s2);
            engine->transform(s3, s2);
            engine->mul_transformed(slot, s3);
            engine->square(slot, 2);
            if (engine->binary_digits())
            {
                bool neg = engine->get(slot, limbs);
                mpz_import(r, limbs.size(), -1, 4, 0, 0, limbs.data());
                if (neg)
                    mpz_neg(r, r);
            }
            else
            {
                gpu::DigitConverter conv(engine->digit_base());
                std::vector<int64_t> digits;
                engine->get_digits(slot, digits);
                conv.from_digits(digits, r);
            }
            mpz_mod(r, r, N);
            mpz_fdiv_r_2exp(t, r, 64);
            double ro = engine->max_roundoff();
            bool ce = engine->carry_error();
            gmp_printf("  CHECK %d: %016ZX  round-off %.3f%s%s\n", cnt, t, ro, ce ? "  CARRY ERROR" : "",
                       mpz_cmp(r, ea.v) == 0 ? "  (GMP: ok)" : "  (GMP: MISMATCH)");
            mpz_clear(N); mpz_clear(r); mpz_clear(t);
            res.ops = 1;
            res.ms_per_op = 0;
            res.roundoff = ro;
            res.carry_error = ce;
            return true;
        }
        mpz_clear(N); mpz_clear(r); mpz_clear(t);

        // warm-up: kernel compilation is done in the constructor, this settles clocks and queues
        engine->square(slot, 20);
        engine->finish();
        engine->max_roundoff();
        int ops = 0, batch = 10;
        auto t0 = std::chrono::steady_clock::now();
        double elapsed = 0;
        while ((ops < min_ops || elapsed < seconds) && !Task::abort_flag())
        {
            engine->square(slot, batch);
            engine->finish();
            ops += batch;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (batch < 1000 && elapsed < seconds/4)
                batch *= 2;
        }
        res.ops = ops;
        res.ms_per_op = ops > 0 ? 1000*elapsed/ops : 0;
        res.roundoff = engine->max_roundoff();
        res.carry_error = engine->carry_error();
        return ops > 0;
    }
    catch (const std::exception& e)
    {
        reason = e.what();
        return false;
    }
}

void GPUStrongCheckExp::engine_init()
{
    if (_engine)
        return;
    int reduce = _bits_reduce + _gwstate->next_fft_count;
    _mpz.reset(new Mpz());
    Mpz& M = *_mpz;
    giant_to_mpz(M.N, *_gwstate->N);
    std::vector<uint32_t> limbs;
    uint32_t fk = 0, fb = 0, fn = 0;     // set by fold_form() when it returns true
    int32_t fc = 0;
    if (fold_form(*_input, M.N, fk, fb, fn, fc))
    {
        // k*b^n+c: fold reduction (one FFT product per operation), slots hold z with x = k^2*z
        // (k*2^n - 1 with the IBDWT: x = k*z)
        if (fb == 2)
            _engine.reset(new gpu::Engine(fk, fn, fc, _device, reduce));
        else
        {
            _engine.reset(new gpu::Engine(gpu::Engine::BaseB(), fk, fb, fn, fc, _device, reduce));
            M.conv.reset(new gpu::DigitConverter(_engine->digit_base()));
        }
        mpz_set_ui(M.scale, fk);
        if (_engine->scale_power() == 2)
            mpz_mul_ui(M.scale, M.scale, fk);
        mpz_mod(M.scale, M.scale, M.N);
    }
    else
    {
        // any odd N: Montgomery, slots hold z with x = z*R^-1
        mpz_t R;
        mpz_init(R);
        _engine.reset(mont_engine(M.N, _device, reduce, R));
        if (!mpz_invert(M.t, M.N, R))
            throw std::runtime_error("N is not invertible modulo R.");
        mpz_sub(M.t, R, M.t);                       // -N^-1 mod R
        mpz_to_limbs(M.t, limbs);
        _engine->set_mont_inverse(limbs.data(), (int)limbs.size());
        mpz_mod(M.scale, R, M.N);
        mpz_clear(R);
        if (!mpz_invert(M.scale, M.scale, M.N))     // R^-1 mod N
            throw std::runtime_error("R is not invertible modulo N.");
    }
    if (!mpz_invert(M.scale_inv, M.scale, M.N))
        throw std::runtime_error("scale is not invertible modulo N.");
    _sX = _engine->alloc_slot();
    _sD = _engine->alloc_slot();
    _sR = _engine->alloc_slot();
    _sT = _engine->alloc_slot();
    _sU = _engine->alloc_slot();
    if (!smooth() && _x0 == 0)
    {
        // giant base X0 (certificate verification): keep its forward FFT for the multiplications
        _sB = _engine->alloc_slot();
        _sBF = _engine->alloc_slot();
        Giant x0;
        x0 = _X0;
        upload(_sB, x0);
        _engine->transform(_sBF, _sB);
    }
    _logging->info("GPU: %s.\n", _engine->description().data());
}

GPUStrongCheckExp::GPUStrongCheckExp(const Giant& exp, bool smooth, const std::vector<Point>& points, int L, int L2, std::function<bool(int, State*, Logging&)> on_point, int device)
    : StrongCheckMultipointExp(exp, smooth, points, L, L2, on_point), _device(device)
{
}

GPUStrongCheckExp::~GPUStrongCheckExp() = default;

void GPUStrongCheckExp::release()
{
    async_discard();
    _engine.reset();
    _mpz.reset();
    StrongCheckMultipointExp::release();
}

// Slots hold z with x = scale*z (mod N), scale = k^2 (fold) or R^-1 (Montgomery).
void GPUStrongCheckExp::upload(int slot, Giant& value)
{
    Mpz& M = *_mpz;
    giant_to_mpz(M.t, value);
    mpz_mul(M.t, M.t, M.scale_inv);
    mpz_mod(M.t, M.t, M.N);                         // fdiv: 0 <= z < N
    if (M.conv)
    {
        std::vector<int64_t> digits;
        M.conv->to_digits(M.t, digits);
        _engine->set_digits(slot, digits);
        return;
    }
    std::vector<uint32_t> limbs;
    mpz_to_limbs(M.t, limbs);
    _engine->set(slot, limbs.data(), (int)limbs.size());
}

void GPUStrongCheckExp::download(int slot, Giant& value)
{
    Mpz& M = *_mpz;
    std::vector<uint32_t> mag;
    if (M.conv)
    {
        std::vector<int64_t> digits;
        _engine->get_digits(slot, digits);
        M.conv->from_digits(digits, M.t);
    }
    else
    {
        bool negative = _engine->get(slot, mag);
        if (mag.empty())
            mpz_set_ui(M.t, 0);
        else
            mpz_import(M.t, mag.size(), -1, 4, 0, 0, mag.data());
        if (negative)
            mpz_neg(M.t, M.t);
    }
    mpz_mul(M.t, M.t, M.scale);
    mpz_mod(M.t, M.t, M.N);
    mpz_to_limbs(M.t, mag);
    Giant z;
    z.arithmetic().init(mag.data(), (int)mag.size(), z);
    value = std::move(z);
}

void GPUStrongCheckExp::check_errors(const char* where)
{
    double err = _engine->max_roundoff();
    bool carry = _engine->carry_error();
    _last_loop_err = err;
    if (err < 0.4 && !carry)
    {
        _logging->debug("GPU max roundoff %.4f %s.\n", err, where);
        return;
    }
    _logging->warning("GPU %s %s (max roundoff %.3f), restarting with smaller digits.\n", carry ? "carry error" : "roundoff error", where, err);
    _bits_reduce++;
    if (_file != nullptr)
        _file->clear();
    _state.reset(new TaskState(5));
    _state->set(_state_recovery->iteration());
    _restart_op = _recovery_op;
    throw TaskRestartException();
}

// Snapshot of X and D (iteration, recovery iteration of the state to write) and a worker that
// converts them to integers mod N. Only host memory and read-only N/scale are shared with it.
void GPUStrongCheckExp::async_start(int iteration, int recovery)
{
    Mpz& M = *_mpz;
    _engine->snapshot(_sX, _sD);
    if (M.conv && !M.aconv)
        M.aconv.reset(new gpu::DigitConverter(_engine->digit_base()));
    _async.active = true;
    _async.iteration = iteration;
    _async.recovery = recovery;
    gpu::Engine* engine = _engine.get();
    _async.done = std::async(std::launch::async, [engine, &M]()
    {
        engine->snapshot_wait();
        mpz_t* out[2] = { &M.ax, &M.ad };
        for (int k = 0; k < 2; k++)
        {
            if (M.aconv)
            {
                std::vector<int64_t> digits;
                engine->snapshot_digits(k, digits);
                M.aconv->from_digits(digits, *out[k]);
            }
            else
            {
                std::vector<uint32_t> mag;
                bool negative = engine->snapshot_get(k, mag);
                if (mag.empty())
                    mpz_set_ui(*out[k], 0);
                else
                    mpz_import(*out[k], mag.size(), -1, 4, 0, 0, mag.data());
                if (negative)
                    mpz_neg(*out[k], *out[k]);
            }
            mpz_mul(*out[k], *out[k], M.scale);
            mpz_mod(*out[k], *out[k], M.N);
        }
    });
}

void GPUStrongCheckExp::async_finish(bool wait)
{
    if (!_async.active)
        return;
    if (!wait && _async.done.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;
    _async.active = false;
    _async.done.get();
    // the snapshot's own error flags decide whether the state may be written
    double err = _engine->snapshot_roundoff();
    bool carry = _engine->snapshot_carry_error();
    if (err >= 0.4 || carry)
    {
        _logging->warning("GPU %s before checkpoint (max roundoff %.3f), restarting with smaller digits.\n", carry ? "carry error" : "roundoff error", err);
        _bits_reduce++;
        if (_file != nullptr)
            _file->clear();
        _state.reset(new TaskState(5));
        _state->set(_state_recovery->iteration());
        _restart_op = _recovery_op;
        throw TaskRestartException();
    }
    _logging->debug("GPU max roundoff %.4f before checkpoint.\n", err);
    Mpz& M = *_mpz;
    std::vector<uint32_t> mag;
    for (int k = 0; k < 2; k++)
    {
        mpz_to_limbs(k == 0 ? M.ax : M.ad, mag);
        Giant z;
        z.arithmetic().init(mag.data(), (int)mag.size(), z);
        (k == 0 ? X() : D()) = std::move(z);
    }
    set_state<StrongCheckState>(_async.iteration, _async.recovery, X(), D());
}

void GPUStrongCheckExp::async_discard()
{
    if (!_async.active)
        return;
    _async.active = false;
    try { _async.done.get(); } catch (...) { }
}

// A small start value x (the base, or a^k computed on the CPU for a smooth exponent) is stored as
// z = x/k^2 mod N (fold engines), and for small k that is a periodic digit pattern (1/k^2 repeats
// with period ord(b, k^2)). Products of periodic inputs have coefficients near the worst case,
// D*B^2/4, instead of about sqrt(D)*B^2: far beyond what the FP32 half of the hybrid engine can
// estimate (round-off 0.5 on the first squaring for 27*2^n+1), and 40x the usual round-off on the
// FP64 engine. The structure is gone once x has wrapped mod N, so the first few iterations (about
// log2(log2 N)) are done here, and the GPU starts after them as from a checkpoint. A checkpoint
// value is a full residue and never this small. Not when a proof point falls into those iterations.
void GPUStrongCheckExp::skip_small_start(int len)
{
    if (!smooth() && _x0 == 0)
        return;
    Mpz& M = *_mpz;
    size_t nbits = mpz_sizeinbase(M.N, 2);
    Giant g;
    state()->to_GWNum(R());
    g = R();
    mpz_t x;
    mpz_init(x);
    giant_to_mpz(x, g);
    mpz_mod(x, x, M.N);
    int s = state()->iteration(), t = 0, end = _points.back().pos;
    bool wrapped = mpz_sizeinbase(x, 2) + 64 >= nbits;
    while (!wrapped && s + t + 1 < end && t < 64)
    {
        mpz_mul(x, x, x);
        if (!smooth() && _exp.bit(len - (s + t) - 1))
            mpz_mul_ui(x, x, _x0);
        t++;
        wrapped = mpz_sizeinbase(x, 2) >= nbits + 64;
        mpz_mod(x, x, M.N);
    }
    bool ok = t > 0 && wrapped;
    for (auto& p : _points)
        if (p.pos > s && p.pos <= s + t)
            ok = false;
    if (ok)
    {
        std::vector<uint32_t> limbs;
        mpz_to_limbs(x, limbs);
        Giant z;
        z.arithmetic().init(limbs.data(), (int)limbs.size(), z);
        _logging->debug("small start value: iterations %d to %d on the CPU.\n", s, s + t);
        _state_recovery.reset(new StateValue(s + t, z));
        _state_recovery->set_written();
        _recovery_op = 0;
        _state.reset(new TaskState(5));
        _state->set(s + t);
    }
    mpz_clear(x);
}

// slot = slot^2 * base^bit
void GPUStrongCheckExp::power_step(int slot, bool bit)
{
    _engine->square(slot, 1, bit && _x0 > 0 ? _x0 : 1);
    if (bit && _x0 == 0)
    {
        _engine->mul_transformed(slot, _sBF);
        _op_count += 1;
    }
}

void GPUStrongCheckExp::execute()
{
    if (result() != nullptr)
        return;
    // GPU handles smooth base-2 exponents and arbitrary exponents with a small base x0 or a giant base X0.
    bool gpu_ok = smooth() ? b() == 2 && state() != nullptr : _x0 > 0 || !_X0.empty();
    if (!gpu_ok || (state() != nullptr && state()->iteration() >= _points.back().pos))
    {
        StrongCheckMultipointExp::execute();
        return;
    }
    if (state() == nullptr)
    {
        if (_x0 > 0)
            init_state(new StateValue(0, _x0));
        else
            init_state(new StateValue(0, _X0));
        state()->set_written();
    }
    int len = _exp.bitlen() - 1;
    async_discard();
    try
    {
        engine_init();
    }
    catch (const std::exception& e)
    {
        _logging->error("GPU initialization failed: %s Continuing on CPU.\n", e.what());
        _engine.reset();
        StrongCheckMultipointExp::execute();
        return;
    }

    if (state_check() == nullptr)
        skip_small_start(len);

    int i, j, next_point, next_check;
    Giant tmp;
    Giant tmp2;
    State* tmp_state;

    state()->to_GWNum(R());
    tmp = R();
    upload(_sR, tmp);
    if (state_check() == nullptr)
    {
        i = state()->iteration();
        _engine->copy(_sX, _sR);
        _engine->copy(_sD, _sR);
    }
    else
    {
        i = state_check()->iteration();
        X() = state_check()->X();
        tmp = X();
        upload(_sX, tmp);
        D() = state_check()->D();
        tmp = D();
        upload(_sD, tmp);
    }

    int last_sync = i;
    for (next_point = 0; next_point < (int)_points.size() && state()->iteration() >= _points[next_point].pos; next_point++);
    while (next_point < (int)_points.size())
    {
        for (next_check = next_point; !_points[next_check].check; next_check++);
        int L = _L;
        int L2 = _L2;
        // Short distance to the next check point (after a resume, or the CPU start of skip_small_start):
        // the largest L with L*L <= distance, so the distance takes 2-3 checks, not one per halving of L.
        int dist = _points[next_check].pos - state()->iteration();
        if (dist < L2 && L > 1)
        {
            L = std::max(1, (int)std::sqrt((double)dist));
            while ((L + 1)*(L + 1) <= dist)
                L++;
            while (L > 1 && L*L > dist)
                L--;
            L2 = L*L;
        }
        if (i - state()->iteration() > L2)
        {
            i = state()->iteration();
            _engine->copy(_sX, _sR);
            _engine->copy(_sD, _sR);
            _state.reset(new TaskState(5));
            _state->set(i);
        }
        else
            for (; next_point < next_check && i >= _points[next_point].pos; next_point++);

        for (j = i - state()->iteration(); j < L2; j++, i++)
        {
            bool bit = !smooth() && _exp.bit(len - i - 1);
            _engine->square(_sX, 1, bit && _x0 > 0 ? _x0 : 1);
            _op_count += 1;
            if (bit && _x0 == 0)
            {
                _engine->mul_transformed(_sX, _sBF);
                _op_count += 1;
            }
            if (j + 1 != L2 && (j + 1)%L == 0)
            {
                _engine->mul(_sD, _sX);
                _op_count += 1;
            }
            int it = i + 1;
            if (j + 1 != L2 && it == _points[next_point].pos && !_points[next_point].check)
            {
                // Proof point between Gerbicz checks (state saved after this iteration's ops).
                async_finish(true);
                check();
                check_errors("before proof point");
                download(_sX, tmp);
                X() = tmp;
                download(_sD, tmp);
                D() = tmp;
                _logging->progress().update(_points[next_point].pos/(double)iterations(), ops());
                _logging->progress_save();
                if (_on_point != nullptr)
                {
                    tmp_state = State::cast(_points[next_point].value, _tmp_state_recovery);
                    tmp_state->set(_points[next_point].pos, X());
                    _on_point(next_point, tmp_state, *_logging);
                }
                _logging->progress().update(_points[next_point].pos/(double)iterations(), ops());
                set_state<StrongCheckState>(_points[next_point].pos, state()->iteration(), X(), D());
                next_point++;
                continue;
            }
            if (j + 1 != L2 && (it - last_sync >= state_update_period() || it == iterations() || abort_flag() || _logging->state_save_flag()))
            {
                // Sync with the GPU regularly, but download the state (costly for big numbers)
                // only when it is about to be written to disk.
                last_sync = it;
                async_finish(false);
                auto now = std::chrono::system_clock::now();
                bool urgent = it == iterations() || abort_flag() || _logging->state_save_flag();
                bool write_due = urgent || std::chrono::duration_cast<std::chrono::seconds>(now - _last_write).count() >= DISK_WRITE_TIME;
                // Periodic checkpoints are downloaded and converted asynchronously (PRST_GPU_ASYNC=0:
                // synchronously): the snapshot carries its own error flags, checked before writing.
                static bool async_ok = !(getenv("PRST_GPU_ASYNC") != nullptr && getenv("PRST_GPU_ASYNC")[0] == '0');
                if (write_due && !urgent && async_ok)
                {
                    if (!_async.active)
                        async_start(it, state()->iteration());
                    write_due = false;
                }
                else
                    check_errors("during exponentiation");
                if (write_due)
                {
                    async_finish(true);
                    download(_sX, tmp);
                    X() = tmp;
                    download(_sD, tmp);
                    D() = tmp;
                    set_state<StrongCheckState>(it, state()->iteration(), X(), D());
                }
                else
                {
                    _logging->progress().update(it/(double)iterations(), ops());
                    if (std::chrono::duration_cast<std::chrono::seconds>(now - _last_progress).count() >= PROGRESS_TIME)
                    {
                        _logging->report_progress();
                        _last_progress = now;
                    }
                    _logging->heartbeat();
                }
            }
        }
        async_finish(true);
        check_errors("before Gerbicz check");
        _logging->progress().update(i/(double)iterations(), ops());
        if (next_point != next_check)
        {
            _logging->error("point missed, invalid parameters.\n");
            throw TaskAbortException();
        }

        _logging->debug("performing Gerbicz check at %d,%d, L2 = %d*%d.\n", next_check, i, L, L2/L);
        Giant Xi;
        download(_sX, Xi);
        tmp_state = State::cast(i == _points[next_check].pos && _points[next_check].value, _tmp_state_recovery);
        X() = Xi;
        tmp_state->set(i, X());

        // PRST_GPU_ERRSTAT=1 (diagnostic): the max round-off of each kind of operation of the check,
        // printed per check; the squarings before it are in "before Gerbicz check" (debug level).
        static const bool errstat = getenv("PRST_GPU_ERRSTAT") != nullptr && getenv("PRST_GPU_ERRSTAT")[0] == '1';
        double es_loop = 0, es_dx = 0, es_pu = 0, es_tu = 0, es_pt = 0, es_tr = 0;
        auto es = [&](double& m) { if (errstat) m = std::max(m, _engine->max_roundoff(true)); };
        if (errstat)
            es_loop = _last_loop_err;
        // D_new = X_i*D;  check: R*D^(2^L) == D_new
        _engine->copy(_sT, _sD);
        _engine->mul(_sD, _sX);
        es(es_dx);
        if (smooth())
            _engine->square(_sT, L);
        else
        {
            // Li check: T = (D * x0^carry)^(2^L) * x0^low, where the exponent bits of this
            // segment, cut into L-bit chunks and summed, give carry*2^L + low.
            for (j = 0, tmp = 0; j*L < L2; j++, tmp += tmp2)
                _exp.arithmetic().substr(_exp, len - state()->iteration() - j*L - L, L, tmp2);
            if (tmp.bitlen() > L)
            {
                tmp.arithmetic().shiftright(tmp, L, tmp2);
                tmp.arithmetic().substr(tmp, 0, L, tmp);
                if (_x0 > 0)
                {
                    // U = x0^carry with a small carry (a few bits more than log2(L2/L)): exactly on the
                    // host. Squaring such a short value on the GPU had the largest round-off of the whole
                    // test (base 101, 101^2 digits: 0.065, every other operation <= 0.006).
                    Giant u;
                    u = _x0;
                    u.arithmetic().powermod(u, tmp2, *_gwstate->N, u);
                    upload(_sU, u);
                }
                else
                {
                    _engine->copy(_sU, _sB);
                    for (j = tmp2.bitlen() - 2; j >= 0; j--)
                        power_step(_sU, tmp2.bit(j));
                }
                es(es_pu);
                _engine->mul(_sT, _sU);
                es(es_tu);
            }
            for (j = L - 1; j >= 0; j--)
                power_step(_sT, tmp.bit(j));
            es(es_pt);
        }
        _engine->mul(_sT, _sR);
        es(es_tr);
        if (errstat)
            _logging->warning("errstat at %d: loop %.4f, D*X %.4f, U steps %.4f, T*U %.4f, T steps %.4f, T*R %.4f\n", i, es_loop, es_dx, es_pu, es_tu, es_pt, es_tr);
        _op_count += L + 2;
        check_errors("during Gerbicz check");
        download(_sT, tmp);
        download(_sD, tmp2);
        if (tmp != tmp2 || tmp2 == 0)
        {
            _logging->warning("Gerbicz check failed at %.1f%%.\n", 100.0*i/iterations());
            if (_file != nullptr)
                _file->clear();
            _state.reset(new TaskState(5));
            _state->set(_state_recovery->iteration());
            _restart_op = _recovery_op;
            throw TaskRestartException();
        }

        _engine->copy(_sR, _sX);
        _engine->copy(_sD, _sX);
        R() = X();
        if (i == _points[next_check].pos)
        {
            if (_on_point != nullptr)
            {
                _logging->progress_save();
                if (_on_point(next_check, tmp_state, *_logging))
                {
                    tmp_state->set_written();
                    _last_write = std::chrono::system_clock::now();
                    if (!_file_recovery_empty)
                    {
                        _file_recovery->clear();
                        _file_recovery_empty = true;
                    }
                }
            }
            next_point++;
        }
        _tmp_state_recovery.swap(_state_recovery);
        _state.reset(new TaskState(5));
        _state->set(i);
        _logging->progress().update(i/(double)iterations(), ops());
        on_state();
        _recovery_op = _restart_op;
        _restart_count = 0;
    }

    // Tail multiplication (c != 1) and completion run on the CPU path.
    if (result() != nullptr)
        done();
    else
        StrongCheckMultipointExp::execute();
}

#endif
