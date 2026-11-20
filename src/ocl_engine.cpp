// OpenCL modular multiplication engine, see ocl_engine.h.
//
// Montgomery mode (any odd N): R = 2^(q*w), N' = -N^-1 mod R. For a product P = z1*z2*mulc:
//   t = P mod R (low q digits), m = t*N' mod R (low q digits of the product), u = (P + m*N)/R.
// With |z| < N and R >= 2*mulc*N, |u| < N again. Three FFT products per operation;
// the FFTs of N and N' are precomputed.
//
// Arithmetic
// ----------
// N = k*2^n + c. A slot holds an integer z as balanced digits of w bits
// (d_i in [-2^(w-1), 2^(w-1))), stored as doubles in a buffer of 2D reals.
// Only the low D digits are ever non-zero, so the length-2D cyclic convolution
// computed with a real FFT is an exact (acyclic) product.
//
// Reduction ("fold"): write P = H*2^n + L. Because k*2^n == -c (mod N),
//     fold(P) = k*L - c*H == k*P (mod N).
// Applying fold twice to a product gives k^2*P. If slots represent x = k^2*z then
//     k^2 * fold(fold(z1*z2)) == k^2 * k^2 * z1*z2 == x1*x2  (mod N)
// i.e. rep(x1*x2) = fold(fold(z1*z2)). For |z| <= B*2^n the result is bounded
// by about 1.5*k*2^n + |c|*(1.5k + |c|*B^2), so magnitudes stay ~1.5*k*2^n.
//
// Pipeline per multiplication:
//   FFT(z) -> pointwise square/mul (real-FFT post/pre-processing) -> IFFT
//   -> round + carry (chunked) -> carry fix-up x2
//   -> fold -> carry -> fix x2 -> fold -> carry -> fix x2 -> to double.

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <map>
#include <regex>
#include "ocl_engine.h"

namespace gpu
{

static const char* kernel_source = R"CLC(
// FP64 is optional: without it only the hybrid engine's kernels are compiled (the FP64 FFT, its
// carries and the Montgomery kernels are left out). dparam: a double kernel argument the hybrid
// kernels do not use, the same size so the host sets it the same way on every device.
#if defined(NO_FP64)
#elif defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define FP64 1
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define FP64 1
#endif
#ifdef FP64
typedef double dparam;
#else
typedef ulong dparam;
#endif

inline long f2l(float v)                       // v integer valued
{
    uint b = as_uint(v);
    int ex = (int)((b >> 23) & 0xFF) - 150;
    long M = (long)((b & 0x7FFFFF) | ((b & 0x7F800000) ? 0x800000u : 0u));
    long r = ex >= 0 ? (M << ex) : (ex > -24 ? (M >> (-ex)) : 0);
    return (b >> 31) ? -r : r;
}

#ifdef FP64
typedef double2 C;

// fma: 2 multiplies + 2 fused multiply-adds instead of 4 + 2, and one rounding fewer
inline C cmul(C a, C b) { return (C)(fma(a.x, b.x, -a.y*b.y), fma(a.x, b.y, a.y*b.x)); }
inline C cconj(C a) { return (C)(a.x, -a.y); }

// Stockham radix-4 pass. N complex points, N/4 work items. tw[m] = exp(-2*pi*i*m/N).
__kernel void fft_r4(__global const C* in, __global C* out, __global const C* tw, uint N, uint Ns, int inv)
{
    uint j = get_global_id(0);
    uint q = N >> 2;
    uint jm = j % Ns;
    C v0 = in[j], v1 = in[j + q], v2 = in[j + 2*q], v3 = in[j + 3*q];
    if (Ns > 1)
    {
        uint step = N/(Ns*4);
        C w1 = tw[jm*step], w2 = tw[2*jm*step], w3 = tw[3*jm*step];
        if (inv) { w1 = cconj(w1); w2 = cconj(w2); w3 = cconj(w3); }
        v1 = cmul(v1, w1); v2 = cmul(v2, w2); v3 = cmul(v3, w3);
    }
    C a = v0 + v2, b = v0 - v2, c = v1 + v3, d = v1 - v3;
    d = inv ? (C)(-d.y, d.x) : (C)(d.y, -d.x);
    uint o = (j/Ns)*Ns*4 + jm;
    out[o] = a + c;
    out[o + Ns] = b + d;
    out[o + 2*Ns] = a - c;
    out[o + 3*Ns] = b - d;
}

// 4-point DFT in place (forward: exp(-2*pi*i/4) = -i).
inline void dft4(C* a, C* b, C* c, C* d, int inv)
{
    C s0 = *a + *c, s1 = *a - *c, s2 = *b + *d, s3 = *b - *d;
    s3 = inv ? (C)(-s3.y, s3.x) : (C)(s3.y, -s3.x);
    *a = s0 + s2; *b = s1 + s3; *c = s0 - s2; *d = s1 - s3;
}

// Stockham radix-16 pass (two radix-4 stages in registers). N/16 work items.
__kernel void fft_r16(__global const C* in, __global C* out, __global const C* tw, uint N, uint Ns, int inv)
{
    uint j = get_global_id(0);
    uint q = N >> 4;
    uint jm = j % Ns;
    C v[16];
    for (uint r = 0; r < 16; r++)
        v[r] = in[j + r*q];
    if (Ns > 1)
    {
        uint step = N/(Ns*16);
        for (uint r = 1; r < 16; r++)
        {
            C w = tw[r*jm*step];
            if (inv) w = cconj(w);
            v[r] = cmul(v[r], w);
        }
    }
    // n = 4a + b, k = c + 4d: inner DFT4 over a for each b, twiddle W16^(b*c), outer DFT4 over b.
    for (uint b = 0; b < 4; b++)
        dft4(&v[b], &v[b + 4], &v[b + 8], &v[b + 12], inv);
    // v[b + 4c] now holds y_b[c]; apply W16^(b*c), W16^m = exp(-+2*pi*i*m/16)
    const double c1 = 0.92387953251128675613, s1 = 0.38268343236508977173, c2 = 0.70710678118654752440;
    const C w1 = (C)(c1, inv ? s1 : -s1), w2 = (C)(c2, inv ? c2 : -c2), w3 = (C)(s1, inv ? c1 : -c1);
    const C w4 = (C)(0.0, inv ? 1.0 : -1.0), w6 = (C)(-c2, inv ? c2 : -c2), w9 = (C)(-c1, inv ? -s1 : s1);
    v[5] = cmul(v[5], w1);  v[9] = cmul(v[9], w2);  v[13] = cmul(v[13], w3);   // b=1: c=1,2,3
    v[6] = cmul(v[6], w2);  v[10] = cmul(v[10], w4); v[14] = cmul(v[14], w6);  // b=2: c=1,2,3
    v[7] = cmul(v[7], w3);  v[11] = cmul(v[11], w6); v[15] = cmul(v[15], w9);  // b=3: c=1,2,3
    for (uint cc = 0; cc < 4; cc++)
        dft4(&v[4*cc], &v[4*cc + 1], &v[4*cc + 2], &v[4*cc + 3], inv);
    // v[4c + d] = X[c + 4d]
    uint o = (j/Ns)*Ns*16 + jm;
    for (uint cc = 0; cc < 4; cc++)
        for (uint d = 0; d < 4; d++)
            out[o + (cc + 4*d)*Ns] = v[4*cc + d];
}

// Stockham radix-2 pass. N/2 work items.
__kernel void fft_r2(__global const C* in, __global C* out, __global const C* tw, uint N, uint Ns, int inv)
{
    uint j = get_global_id(0);
    uint h = N >> 1;
    uint jm = j % Ns;
    C v0 = in[j], v1 = in[j + h];
    if (Ns > 1)
    {
        C w = tw[jm*(N/(Ns*2))];
        if (inv) w = cconj(w);
        v1 = cmul(v1, w);
    }
    uint o = (j/Ns)*Ns*2 + jm;
    out[o] = v0 + v1;
    out[o + Ns] = v0 - v1;
}

// Last Stockham pass of a 3*2^k FFT: radix 3 with Ns = N/3, so butterfly j reads j + r*Ns and
// writes j + k*Ns. N/3 work items.
__kernel void fft_r3(__global const C* in, __global C* out, __global const C* tw, uint N, int inv)
{
    uint j = get_global_id(0);
    uint h = N/3;
    if (j >= h)
        return;
    C v0 = in[j], v1 = in[j + h], v2 = in[j + 2*h];
    if (j != 0)
    {
        C w1 = tw[j], w2 = tw[2*j];
        if (inv) { w1 = cconj(w1); w2 = cconj(w2); }
        v1 = cmul(v1, w1);
        v2 = cmul(v2, w2);
    }
    // W3 = exp(-+2*pi*i/3) = -1/2 -+ i*sqrt(3)/2
    const double s3 = 0.86602540378443864676;
    C s = v1 + v2, d = v1 - v2;
    C t = v0 - 0.5*s;
    C id = (C)(-d.y, d.x)*(inv ? s3 : -s3);     // -+ i*sqrt(3)/2 * d
    out[j] = v0 + s;
    out[j + h] = t + id;
    out[j + 2*h] = t - id;
}

// Real-FFT post-processing, pointwise product, and pre-processing for the inverse.
// A holds the D-point FFT of the packed real sequence (x[2j] + i*x[2j+1]).
// tw2[k] = exp(-i*pi*k/D). Work items k = 0..D/2.
inline void unpack(C Zk, C Zm, C t, C* Xk, C* Xm)
{
    C E = 0.5*(Zk + cconj(Zm));
    C O = 0.5*(Zk - cconj(Zm));
    O = (C)(O.y, -O.x);              // divide by i
    C tO = cmul(t, O);
    *Xk = E + tO;
    *Xm = cconj(E - tO);
}

__kernel void realmul(__global C* A, __global const C* B, __global const C* tw2, uint D, int square)
{
    uint k = get_global_id(0);
    if (k > (D >> 1))
        return;
    uint mk = k == 0 ? 0 : D - k;
    C t = tw2[k];
    C Xk, Xm;
    unpack(A[k], A[mk], t, &Xk, &Xm);
    C Yk, Ym;
    if (square)
    {
        Yk = cmul(Xk, Xk);
        Ym = cmul(Xm, Xm);
    }
    else
    {
        C Wk, Wm;
        unpack(B[k], B[mk], t, &Wk, &Wm);
        Yk = cmul(Xk, Wk);
        Ym = cmul(Xm, Wm);
    }
    C E2 = 0.5*(Yk + cconj(Ym));
    C O2 = cmul(0.5*(Yk - cconj(Ym)), cconj(t));
    C Zk = E2 + (C)(-O2.y, O2.x);
    C cO2 = cconj(O2);
    C Zm = cconj(E2) + (C)(-cO2.y, cO2.x);
    A[k] = Zk;
    if (k != 0 && k != (D >> 1))
        A[mk] = Zm;
}
#endif

// Balanced digit of v in base B (w != 0: B = 2^w), carry = (v - digit)/B exactly.
// General B: quotient estimated in double, then corrected with exact integer arithmetic
// (two estimates handle |v| up to 2^63; the final step fixes the last +-1).
// Without FP64: single precision estimates, each one cuts |v| by about 2^-23 relative to v/B.
inline long balanced(long v, uint w, long* carry, long B, dparam invB)
{
#ifndef FP64
    if (w == 0)
    {
        float invBf = 1.0f/(float)B;
        long lo = -(B >> 1), hi = lo + B - 1;
        long q = 0, d = v;
        for (int it = 0; it < 4 && (d < lo || d > hi); it++)
        {
            long q2 = f2l(rint(convert_float(d)*invBf));
            q += q2;
            d -= q2*B;
        }
        if (d > hi) { d -= B; q++; }
        else if (d < lo) { d += B; q--; }
        *carry = q;
        return d;
    }
#else
    if (w == 0)
    {
        long q = (long)rint((double)v*invB);
        long d = v - q*B;
        long lo = -(B >> 1), hi = lo + B - 1;
        if (d < lo || d > hi)
        {
            long q2 = (long)rint((double)d*invB);
            q += q2;
            d -= q2*B;
            if (d > hi) { d -= B; q++; }
            else if (d < lo) { d += B; q--; }
        }
        *carry = q;
        return d;
    }
#endif
    long d = v & ((1L << w) - 1);
    if (d >= (1L << (w - 1)))
        d -= (1L << w);
    *carry = (v - d) >> w;
    return d;
}

// Round inverse FFT output, measure round-off, carry within chunks of CH digits.
// Chunked carry kernels: work-groups of CW items, each owning CHUNK consecutive digits. Digits are
// staged through local memory so global reads and writes are contiguous (coalesced); the serial
// carry chain of each chunk then runs out of local memory. PADL avoids bank conflicts.
#define CW 64
#ifndef CHUNK
#define CHUNK 16
#endif
// Normalized digits (|d| <= B/2 < 2^31) are stored as 32-bit integers between kernels: half the
// memory traffic of int64. Carries and intermediate values stay 64-bit.
typedef int digit;
#define PADL(i) ((i) + ((i) >> 4))

// Carry between the chunks of one work-group, in local memory: the carry out of chunk lid is added
// to chunk lid + 1 (usually it settles within a digit or two). A carry that runs through a whole
// chunk is passed on in another round, so long runs of edge digits are still handled exactly.
// Chunks at or beyond `total` pass carries through untouched. Returns, in work item CW - 1, the
// carry out of the whole group; the global carry passes then only work at group granularity.
inline long group_carry(__local digit* ys, __local long* cs, __local int* more, long carry, uint lid, uint chunk_start, uint total,
                        uint w, long B, dparam invB)
{
    long out = 0;
    cs[lid] = carry;
    for (;;)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        long c = lid ? cs[lid - 1] : 0;
        if (lid == CW - 1)
            out += cs[lid];
        if (lid == 0)
            *more = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (chunk_start < total)
        {
            uint lb = lid*CHUNK;
            for (uint t = 0; t < CHUNK && c != 0; t++)
                ys[PADL(lb + t)] = (digit)balanced((long)ys[PADL(lb + t)] + c, w, &c, B, invB);
        }
        cs[lid] = c;
        if (c != 0)
            *more = 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (*more == 0)
            break;
    }
    return out;
}

#ifdef FP64
__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void carry_d(__global const double* in, __global digit* out, __global long* carries, double scale, uint w, uint CH, __global uint* maxerr, long mulc, long B, double invB, uint nch)
{
    __local double xs[PADL(CW*CHUNK) + 1];
    __local digit ys[PADL(CW*CHUNK) + 1];
    __local long cs[CW];
    __local int more;
    long lcarry;
    uint lid = get_local_id(0);
    uint g = get_global_id(0);
    uint gbase = get_group_id(0)*CW*CHUNK;
    uint total = nch*CHUNK;
    for (uint k = lid; k < CW*CHUNK; k += CW)
        xs[PADL(k)] = gbase + k < total ? in[gbase + k] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    uint lb = lid*CHUNK;
    float me = 0.0f;
    if (w == 0 && mulc <= 16)
    {
        // Non-binary base: carry in double. Values stay below 2^53, the estimate q = rint(v/B)
        // is off by at most one, and fma(-q, B, v) gives the exact small remainder.
        double Bd = (double)B, lo = (double)(-(B >> 1)), hi = lo + Bd - 1.0, m = (double)mulc;
        double carry = 0.0;
        for (uint t = 0; t < CHUNK; t++)
        {
            double x = xs[PADL(lb + t)]*scale;
            double r = rint(x);
            me = fmax(me, (float)fabs(x - r));
            double v = r*m + carry;
            double q = rint(v*invB);
            double d = fma(-q, Bd, v);
            if (d > hi) { d -= Bd; q += 1.0; }
            else if (d < lo) { d += Bd; q -= 1.0; }
            ys[PADL(lb + t)] = (digit)d;
            carry = q;
        }
        lcarry = (long)carry;
    }
    else
    {
        long carry = 0;
        for (uint t = 0; t < CHUNK; t++)
        {
            double x = xs[PADL(lb + t)]*scale;
            double r = rint(x);
            me = fmax(me, (float)fabs(x - r));
            ys[PADL(lb + t)] = (digit)balanced((long)r*mulc + carry, w, &carry, B, invB);
        }
        lcarry = carry;
    }
    long gc = group_carry(ys, cs, &more, lcarry, lid, gbase + lb, total, w, B, invB);
    if (lid == CW - 1)
        carries[get_group_id(0)] = gc;
    for (uint k = lid; k < CW*CHUNK; k += CW)
        if (gbase + k < total)
            out[gbase + k] = (digit)ys[PADL(k)];
    if (g < nch)
        atomic_max(maxerr, as_uint(me));
}
#endif

// Carry within chunks for integer input.
__kernel void carry_l(__global digit* d, __global long* carries, uint w, uint CH, long B, dparam invB)
{
    uint g = get_global_id(0);
    uint base = g*CH;
    long carry = 0;
    for (uint t = 0; t < CH; t++)
        d[base + t] = (digit)balanced((long)d[base + t] + carry, w, &carry, B, invB);
    carries[g] = carry;
}

// Digit buffers are int64, or double when dbl != 0 (the final pass writes straight into an FFT slot).
// Without FP64 slots hold 32-bit integers and dbl is always 0.
#ifdef FP64
inline long ld(__global digit* d, uint i, int dbl) { return dbl ? (long)((__global double*)d)[i] : (long)d[i]; }
inline void st(__global digit* d, uint i, long v, int dbl) { if (dbl) ((__global double*)d)[i] = (double)v; else d[i] = (digit)v; }
#else
inline long ld(__global digit* d, uint i, int dbl) { return (long)d[i]; }
inline void st(__global digit* d, uint i, long v, int dbl) { d[i] = (digit)v; }
#endif

// Add the carry out of the previous chunk and propagate. last != 0: any leftover is an error.
// Chunk g starts at digit g*CH - off (off != 0: the shifted grid of carry_fold; chunk 0 is short).
// ek != 0: ein[] holds, per chunk, the carries out of carry_fold's first fold as three numbers
// (L, H, E): carries of the low part, of the high part (both at input level) and of e itself.
// e gets k*L - tcb*H + E digit by digit, and each e digit enters the slot multiplied by ek = k.
// tn != 0: chunk 0 first applies the second half of carry_fold's fold 2: z -= c*H', where H' digit j
// is bmul * (e digit q+1+j) = bmul*etop[j + 1], plus the upper part of e digit q (etop[0]) at
// j = 0. tn = number of top digits (at most the length of chunk 0), tcb = c*bmul, tc = c.
inline void carry_fix_g(uint g, __global digit* d, __global const long* cin, __global long* cout, uint w, uint CH, uint nch, __global uint* err, int last, long B, dparam invB, int dbl, uint len,
                        uint off, __global const long* ein, long ek, __global const digit* etop, uint tn, long tcb, long tc, int wrap)
{
    // wrap = +-1: digits mod B^len -+ 1 (cyclic / negacyclic), the carry out of the top enters chunk 0
    long carry = g ? cin[g - 1] : wrap*cin[nch - 1];
    long eL = 0, eH = 0, ec = 0;
    if (g && ek != 0)
    {
        eL = ein[3*(g - 1)];
        eH = ein[3*(g - 1) + 1];
        ec = ein[3*(g - 1) + 2];
    }
    uint base = g ? g*CH - off : 0;
    uint n = g ? CH : CH - off;
    uint t = 0;
    if (g == 0 && tn != 0)
    {
        // exact in 64-bit integers: |tcb*e| < |c|*B^2
        long cy = 0;
        for (; t + 1 < tn; t++)
        {
            long e = (long)etop[t + 1];
            if (t != 0 && e == 0 && cy == 0)
                continue;
            long v = ld(d, t, dbl) - tcb*e + cy;
            if (t == 0)
                v -= tc*(long)etop[0];
            st(d, t, balanced(v, w, &cy, B, invB), dbl);
        }
        carry = cy;
    }
    for (; t < n && (carry != 0 || eL != 0 || eH != 0 || ec != 0) && base + t < len; t++)
    {
        long v = ld(d, base + t, dbl) + carry;
        if (eL != 0 || eH != 0 || ec != 0)
        {
            long dl = balanced(eL, w, &eL, B, invB), dh = balanced(eH, w, &eH, B, invB);
            v += ek*balanced(ek*dl - tcb*dh + ec, w, &ec, B, invB);
        }
        st(d, base + t, balanced(v, w, &carry, B, invB), dbl);
    }
    if ((g == nch - 1 && cin[g] != 0 && wrap == 0) || eL != 0 || eH != 0 || ec != 0)
        atomic_or(err, 1u);     // overflow out of the top of the buffer
    cout[g] = carry;
    if (last && carry != 0)
        atomic_or(err, 8u);     // long run of edge digits: carry_seq finishes the job
}

__kernel void carry_fix(__global digit* d, __global const long* cin, __global long* cout, uint w, uint CH, uint nch, __global uint* err, int last, long B, dparam invB, int dbl, uint len,
                        uint off, __global const long* ein, long ek, __global const digit* etop, uint tn, long tcb, long tc, int wrap)
{
    uint g = get_global_id(0);
    if (g < nch)
        carry_fix_g(g, d, cin, cout, w, CH, nch, err, last, B, invB, dbl, len, off, ein, ek, etop, tn, tcb, tc, wrap);
}

// Rare slow path: after two parallel passes a carry is still pending (many consecutive digits
// at the edge of the digit range, as in small or highly structured values). One work item adds
// the leftover chunk carries and propagates them through the whole buffer. No-op otherwise.
inline void carry_seq_1(__global digit* d, __global const long* left, __global uint* err, uint len, uint CH, uint w, long B, dparam invB, int dbl, uint off, int wrap)
{
    long carry = 0;
    for (uint i = 0; i < len; i++)
    {
        if (i != 0 && (i + off) % CH == 0)
            carry += left[(i + off)/CH - 1];
        if (carry != 0)
            st(d, i, balanced(ld(d, i, dbl) + carry, w, &carry, B, invB), dbl);
    }
    carry += left[(len + off + CH - 1)/CH - 1];
    // cyclic / negacyclic: the carry out of the top wraps around to digit 0
    for (int r = 0; r < 4 && wrap != 0 && carry != 0; r++)
    {
        carry *= wrap;
        for (uint i = 0; i < len && carry != 0; i++)
            st(d, i, balanced(ld(d, i, dbl) + carry, w, &carry, B, invB), dbl);
    }
    if (carry != 0)
        atomic_or(err, 1u);
    atomic_and(err, ~8u);
}

__kernel void carry_seq(__global digit* d, __global const long* left, __global uint* err, uint len, uint CH, uint w, long B, dparam invB, int dbl, uint off, int wrap)
{
    if ((*err & 8u) == 0)
        return;
    carry_seq_1(d, left, err, len, CH, w, B, invB, dbl, off, wrap);
}

// carry_fix (first pass) + carry_fix (last pass) + carry_seq in one launch: every work-group does the
// first pass for its chunks, and the last work-group to finish (atomic ticket) does the second pass
// for all chunks and, if needed, the sequential one. No work-group waits for another. cout2/left:
// carries out of the second pass. ticket: a counter that is 0 between launches.
__kernel void carry_fix3(__global digit* d, __global const long* cin, __global long* cout, uint w, uint CH, uint nch, __global uint* err, int unused, long B, dparam invB, int dbl, uint len,
                         uint off, __global const long* ein, long ek, __global const digit* etop, uint tn, long tcb, long tc, __global long* cout2, __global uint* ticket,
                         int wrap)
{
    __local uint is_last;
    uint g = get_global_id(0), lid = get_local_id(0), ls = get_local_size(0);
    if (g < nch)
        carry_fix_g(g, d, cin, cout, w, CH, nch, err, 0, B, invB, dbl, len, off, ein, ek, etop, tn, tcb, tc, wrap);
    // make this group's digits and carries visible before taking the ticket
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
        is_last = atomic_inc(ticket) == get_num_groups(0) - 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!is_last)
        return;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    for (uint h = lid; h < nch; h += ls)
        carry_fix_g(h, d, cout, cout2, w, CH, nch, err, 1, B, invB, dbl, len, off, ein, 0, etop, 0, tcb, tc, wrap);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0)
    {
        if ((atomic_or(err, 0u) & 8u) != 0)
            carry_seq_1(d, cout2, err, len, CH, w, B, invB, dbl, off, wrap);
        atomic_xchg(ticket, 0u);
    }
}

#ifdef FP64
// out = k*L - c*H where in = H*b^n + L, b^n = B^q * bs (bs = b^s, bmul = b^(e-s) = B/bs).
// Digits are normalized (|d| < B/2 < 2^31), so the split uses 32-bit floor division by bs.
inline void split_digit(long d, long bs, long* a, long* r)
{
    int di = (int)d, bi = (int)bs;
    int ai = di/bi, ri = di - ai*bi;
    if (ri < 0) { ri += bi; ai--; }
    *a = ai;
    *r = ri;
}

// floor division of a normalized digit (|d| < 2^31) by bs without an integer divide:
// reciprocal estimate, then exact correction.
inline void split_fast(long d, int bs, double inv_bs, int use24, int* a, int* r)
{
    int di = (int)d;
    int ai = (int)floor((double)di*inv_bs);
    int ri = di - (use24 ? mul24(ai, bs) : ai*bs);   // mul24: full rate on AMD, operands < 2^23
    if (ri < 0) { ri += bs; ai--; }
    else if (ri >= bs) { ri -= bs; ai++; }
    *a = ai;
    *r = ri;
}

// Fold and carry within chunks in one pass: out = k*L - c*H with in = H*b^n + L,
// b^n = B^q * bs, bmul = B/bs. len = significant digits of in, nch = output chunks.
// in[i] (low part) and in[q + i] (high part) are staged through local memory (coalesced).
// fast != 0 (non-binary base, k*B < 2^51): reciprocal splits and carries in double.
__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void fold_carry(__global const digit* in, __global digit* out, __global long* carries, uint q, long bs, long bmul, long k, long c, uint len,
                uint w, uint CH, long B, double invB, int dbl, int fast, double inv_bs, uint nch, int use24)
{
    __local digit Ls[PADL(CW*CHUNK) + 1];
    __local digit Hs[PADL(CW*CHUNK + 1) + 1];
    __local long cs[CW];
    __local int more;
    long lcarry;
    uint lid = get_local_id(0);
    uint gbase = get_group_id(0)*CW*CHUNK;
    uint total = nch*CHUNK;
    for (uint t = lid; t < CW*CHUNK; t += CW)
    {
        uint i = gbase + t;
        Ls[PADL(t)] = (i <= q && i < len) ? in[i] : 0;
    }
    for (uint t = lid; t <= CW*CHUNK; t += CW)
    {
        uint jq = q + gbase + t;
        Hs[PADL(t)] = jq < len ? in[jq] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint lb = lid*CHUNK;
    if (fast)
    {
        int ibs = (int)bs;
        double Bd = (double)B, lo = (double)(-(B >> 1)), hi = lo + Bd - 1.0;
        double kd = (double)k, cd = (double)c, bm = (double)bmul;
        double carry = 0.0;
        int a0, r0, a1, r1;
        split_fast(Hs[PADL(lb)], ibs, inv_bs, use24, &a0, &r0);
        for (uint t = 0; t < CHUNK; t++)
        {
            uint i = gbase + lb + t;
            double l = 0.0;
            if (i < q)
                l = (double)Ls[PADL(lb + t)];
            else if (i == q)
            {
                int aq, rq;
                split_fast(Ls[PADL(lb + t)], ibs, inv_bs, use24, &aq, &rq);
                l = (double)rq;
            }
            split_fast(Hs[PADL(lb + t + 1)], ibs, inv_bs, use24, &a1, &r1);
            double h = (double)a0 + (double)r1*bm;
            double v = kd*l - cd*h + carry;
            double qq = rint(v*invB);
            double d = fma(-qq, Bd, v);
            if (d > hi) { d -= Bd; qq += 1.0; }
            else if (d < lo) { d += Bd; qq -= 1.0; }
            Ls[PADL(lb + t)] = (digit)d;
            carry = qq;
            a0 = a1;
            r0 = r1;
        }
        lcarry = (long)carry;
    }
    else
    {
        long carry = 0;
        for (uint t = 0; t < CHUNK; t++)
        {
            uint i = gbase + lb + t;
            long a, r;
            long l = 0;
            if (i < q)
                l = Ls[PADL(lb + t)];
            else if (i == q)
            {
                split_digit(Ls[PADL(lb + t)], bs, &a, &r);
                l = r;
            }
            split_digit(Hs[PADL(lb + t)], bs, &a, &r);
            long h = a;
            split_digit(Hs[PADL(lb + t + 1)], bs, &a, &r);
            h += r*bmul;
            Ls[PADL(lb + t)] = (digit)balanced(k*l - c*h + carry, w, &carry, B, invB);
        }
        lcarry = carry;
    }
    long gc = group_carry(Ls, cs, &more, lcarry, lid, gbase + lb, total, w, B, invB);
    if (lid == CW - 1)
        carries[get_group_id(0)] = gc;
    for (uint t = lid; t < CW*CHUNK; t += CW)
        if (gbase + t < total)
            st(out, gbase + t, Ls[PADL(t)], dbl);
}
#endif

// Fused reduction of a product: rounding of the inverse FFT output, carry, both folds and the
// conversion to a slot, in one pass (replaces carry_d + 2 x fold_carry and their fix-up passes).
//
// P = sum r_i*B^i, r_i = rint(in[i]*scale)*m. With b^n = B^q*bs and B^(q+1) = b^n*bmul:
//   fold 1: P = L + b^n*H, L = digits 0..q, H digit j = bmul * digit q+1+j   =>  e = k*L - c*H
//   fold 2: e = L' + b^n*H', L' = digits 0..q-1 and (digit q) mod bs          =>  z = k*L' - c*H'
// so z == k^2*P (mod N) as with two fold_carry passes (the split differs, z is the same residue).
// Thread = one chunk of positions, read straight from global memory and kept in registers; both
// input streams are normalized on the fly (carries cL, cH), then u = k*dL - c*bmul*dH is normalized
// to e (carry cE). All of it in doubles: the host checks (k + |c|*bmul)*B < 2^51. The three
// carries of a chunk enter the next chunk of the work-group (exchanged through local memory);
// z = k*e follows the same way.
// Group g covers positions g*CW*CHUNK - off ..: the host picks off so that q and the top digits
// of e (q..U-1) sit inside one group, far from its ends. The carries out of a group go to zcar[]
// (z level) and ecar[] (e level, a large number: carry_fix spreads it over a few digits times k).
// H' is tiny (e digits q..U-1): those digits go to etop[] and carry_fix (chunk 0) applies them.
// Digits may exceed the balanced range by a unit (quotients are estimated); that is harmless.
// NTT input (ntt != 0): the inverse transform over GF(p^2), p = 2^61 - 1, left D times the product
// coefficients mod p. They are exact integers below 2^60 in magnitude: times dinv = 1/D, to signed,
// then normalized in exact integer arithmetic (the multiplier m is applied to the normalized digit,
// so nothing overflows); the rest is the same.
#define PM61 0x1FFFFFFFFFFFFFFFUL
inline ulong mred61(ulong x) { return (x & PM61) + (x >> 61); }
inline ulong mmul61(ulong a, ulong b) { ulong lo = a*b, hi = mul_hi(a, b); return mred61((hi << 3) + (lo >> 61) + (lo & PM61)); }
// Hybrid FP32 + GF(M31^2): f approximates the coefficient (times D), a is it mod 2^31 - 1 (times D).
// x = rint(f/D) + c with c the centered difference mod p: exact while the FP32 error is below p/2.
// *me gets the error as a fraction of p (0.5 = ambiguous), like the FP64 round-off.
#define PM31 0x7FFFFFFFu
inline uint mod31_64(ulong u)
{
    u = (u & PM31) + (u >> 31);
    u = (u & PM31) + (u >> 31);
    uint r = (uint)u;
    return r >= PM31 ? r - PM31 : r;
}
#ifdef FP64
inline long h32_value(float f, uint a, uint sh, double scale, double* me)
{
    double xd = (double)f*scale;
    long xl = (long)rint(xd);
    uint ar = ((a << sh) & PM31) | (a >> (31 - sh));
    uint m = mod31_64(xl < 0 ? (ulong)(-xl) : (ulong)xl);
    if (xl < 0 && m != 0)
        m = PM31 - m;
    uint d = ar >= m ? ar - m : ar + PM31 - m;
    long c = d > (PM31 >> 1) ? (long)d - (long)PM31 : (long)d;
    long x = xl + c;
    *me = fmax(*me, fabs((double)x - xd)*(1.0/2147483647.0));
    return x;
}

// Same, all in double arithmetic, returning the split x = q*B + r (r returned, q in *q).
// xd = rint(f/D) has at most 24 significant bits, so every fma below is exact (results are small).
// *me gets |correction| (scaled by 1/p by the caller).
// Returns t = (x - q1*B)*m + cin and q1 (x = rint(f/D) + correction): the caller normalizes t.
// q1 is only an estimate (single precision): x - q1*B stays far below 2^53, so the fma is exact.
inline double h32_split(float f, uint a, uint sh, float scalef, double Bd, float invBf, double m, double cin, double* q, uint* me)
{
    // xf = rint(f/D) in single precision (exact: D is a power of two); xf = +-M*2^E, M < 2^24
    float xf = rint(f*scalef);
    uint bits = as_uint(xf);
    int ex = (int)((bits >> 23) & 0xFF) - 150;
    uint M = (bits & 0x7FFFFF) | ((bits & 0x7F800000) ? 0x800000u : 0u);
    // xf mod p: M*2^E with 2^31 == 1 is a rotation of the 31-bit M (E < 0: xf is a small integer)
    uint mr;
    if (ex >= 0)
    {
        uint e = (uint)ex % 31;
        mr = e ? (((M << e) & PM31) | (M >> (31 - e))) : M;
    }
    else
        mr = ex > -24 ? M >> (uint)(-ex) : 0u;
    if ((bits >> 31) && mr != 0)
        mr = PM31 - mr;
    uint ar = ((a << sh) & PM31) | (a >> (31 - sh));
    uint d = ar >= mr ? ar - mr : ar + PM31 - mr;
    int c = d > (PM31 >> 1) ? (int)(d - PM31) : (int)d;
    *me = max(*me, (uint)abs(c));
    double q1 = (double)rint(xf*invBf);
    *q = q1;
    return fma(fma(-q1, Bd, (double)xf) + (double)c, m, cin);
}
#endif

// FP32 + GF(P3^2): as h32_split with p = P3 (not a Mersenne prime): x mod p from xf = +-M*2^E with
// a table of 2^E (Montgomery products), the NTT residue times R/D likewise.
inline uint mont3(uint a, uint b, uint P, uint PINV)
{
    uint lo = a*b, hi = mul_hi(a, b);
    uint mh = mul_hi(lo*PINV, P);
    return hi >= mh ? hi - mh : hi - mh + P;
}
#ifdef FP64
// As h32_value with p = P3, without a 64-bit division: xf = rint(f/D) in single precision is exactly
// +-M*2^E (M < 2^24), so xf mod p = M*2^E from the table pw (2^E*R mod p, Montgomery products), and the
// residue a times C3 (R^2/D) is the coefficient mod p. x = xf + the centered difference.
inline long h32_value3(float f, uint a, float scalef, double* me, uint P, uint PINV, uint C3, __global const uint* pw)
{
    float xf = rint(f*scalef);
    uint bits = as_uint(xf);
    int ex = (int)((bits >> 23) & 0xFF) - 150;
    uint M = (bits & 0x7FFFFF) | ((bits & 0x7F800000) ? 0x800000u : 0u);
    uint mr;
    if (ex >= 0)
        mr = mont3(M, pw[ex & 63], P, PINV);
    else
        mr = ex > -24 ? M >> (uint)(-ex) : 0u;
    if ((bits >> 31) && mr != 0)
        mr = P - mr;
    uint ar = mont3(a, C3, P, PINV);
    uint d = ar >= mr ? ar - mr : ar + P - mr;
    int c = d > (P >> 1) ? (int)(d - P) : (int)d;
    *me = fmax(*me, (double)abs(c)/(double)P);
    return (long)xf + c;
}

inline double h32_split3(float f, uint a, float scalef, double Bd, float invBf, double m, double cin, double* q, uint* me,
                         uint P, uint PINV, uint C3, __global const uint* pw)
{
    float xf = rint(f*scalef);
    uint bits = as_uint(xf);
    int ex = (int)((bits >> 23) & 0xFF) - 150;
    uint M = (bits & 0x7FFFFF) | ((bits & 0x7F800000) ? 0x800000u : 0u);
    uint mr;
    if (ex >= 0)
        mr = mont3(M, pw[ex & 63], P, PINV);
    else
        mr = ex > -24 ? M >> (uint)(-ex) : 0u;
    if ((bits >> 31) && mr != 0)
        mr = P - mr;
    uint ar = mont3(a, C3, P, PINV);
    uint d = ar >= mr ? ar - mr : ar + P - mr;
    int c = d > (P >> 1) ? (int)(d - P) : (int)d;
    *me = max(*me, (uint)abs(c));
    double q1 = (double)rint(xf*invBf);
    *q = q1;
    return fma(fma(-q1, Bd, (double)xf) + (double)c, m, cin);
}
#endif

// a (below 2^61 + 8) times 1/D = 2^sh mod p, as a signed integer: a rotation of the 61-bit value
inline long ntt_signed(ulong a, uint sh)
{
    ulong x = mred61(a);
    if (x >= PM61)
        x -= PM61;
    x = ((x << sh) & PM61) | (x >> (61 - sh));
    return x > (PM61 >> 1) ? (long)x - (long)PM61 : (long)x;
}

#ifdef FP64
__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void carry_fold(__global const double* in, __global double* out, __global long* zcar, __global long* ecar, __global digit* etop,
                __global uint* maxerr, __global uint* err, double scale, double m, int q, int len,
                double kd, double cb, double Bd, double invB, double bsd, double inv_bs, int U,
                long kl, long cbl, int off, int ntt, ulong dinv, uint w, long B, __global const uint* in31,
                uint P3v, uint PINV3, uint C3, __global const uint* pw3, int islot)
{
    __local double Ls[PADL(CW*CHUNK) + 1];
    __local double csL[CW], csH[CW], csE[CW];
    __local int more;
    uint lid = get_local_id(0);
    int p0 = (int)(get_group_id(0)*CW*CHUNK + lid*CHUNK) - off;
    double e[CHUNK];
    double me = 0.0;
    double cL = 0.0, cH = 0.0, cE = 0.0;
    if (ntt)
    {
        // s = q0*B + d0 with q0 = rint(s/B) (estimated in double: |d0| < 1.5*B, which is enough), then
        // d0*m + carry is normalized in double; the carries (q0*m + ...) stay below 2^53
        __global const ulong* inu = (__global const ulong*)in;
        uint sh = (uint)dinv;
        uint mi = 0;
#pragma unroll
        for (uint t = 0; t < CHUNK; t++)
        {
            int i = p0 + (int)t;
            int jq = q + 1 + i;
            double qL, qH, vL, vH;
            if (ntt == 3)
            {
                __global const float* inf = (__global const float*)in;
                bool vl = i >= 0 && i <= q, vh = i >= 0 && jq < len;
                vL = h32_split3(vl ? inf[i] : 0.0f, vl ? in31[i] : 0u, (float)scale, Bd, (float)invB, m, cL, &qL, &mi, P3v, PINV3, C3, pw3);
                vH = h32_split3(vh ? inf[jq] : 0.0f, vh ? in31[jq] : 0u, (float)scale, Bd, (float)invB, m, cH, &qH, &mi, P3v, PINV3, C3, pw3);
            }
            else if (ntt == 2)
            {
                __global const float* inf = (__global const float*)in;
                bool vl = i >= 0 && i <= q, vh = i >= 0 && jq < len;
                vL = h32_split(vl ? inf[i] : 0.0f, vl ? in31[i] : 0u, sh, (float)scale, Bd, (float)invB, m, cL, &qL, &mi);
                vH = h32_split(vh ? inf[jq] : 0.0f, vh ? in31[jq] : 0u, sh, (float)scale, Bd, (float)invB, m, cH, &qH, &mi);
            }
            else
            {
                long sL = ntt_signed(i >= 0 && i <= q ? inu[i] : 0UL, sh), sH = ntt_signed(i >= 0 && jq < len ? inu[jq] : 0UL, sh);
                qL = rint((double)sL*invB);
                qH = rint((double)sH*invB);
                vL = fma((double)(sL - (long)qL*B), m, cL);
                vH = fma((double)(sH - (long)qH*B), m, cH);
            }
            double c1 = rint(vL*invB), c2 = rint(vH*invB);
            double dL = fma(-c1, Bd, vL), dH = fma(-c2, Bd, vH);
            cL = fma(qL, m, c1);
            cH = fma(qH, m, c2);
            double u = fma(kd, dL, fma(-cb, dH, cE));
            cE = rint(u*invB);
            e[t] = fma(-cE, Bd, u);
        }
        me = (double)mi;
    }
    else
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
    {
        int i = p0 + (int)t;
        int jq = q + 1 + i;
        double xL = (i >= 0 && i <= q ? in[i] : 0.0)*scale, xH = (i >= 0 && jq < len ? in[jq] : 0.0)*scale;
        double rL = rint(xL), rH = rint(xH);
        me = fmax(me, fmax(fabs(xL - rL), fabs(xH - rH)));
        double vL = fma(rL, m, cL), vH = fma(rH, m, cH);
        cL = rint(vL*invB);
        cH = rint(vH*invB);
        double dL = fma(-cL, Bd, vL), dH = fma(-cH, Bd, vH);
        double u = fma(kd, dL, fma(-cb, dH, cE));
        cE = rint(u*invB);
        e[t] = fma(-cE, Bd, u);
    }
    // e level: the carries (cL, cH at the input level, cE) of chunk lid - 1 enter chunk lid, usually
    // settling within a few digits; whatever runs through a whole chunk goes on in another round.
    // oL/oH/oE (work item CW - 1): carries leaving the group.
    double oL = 0.0, oH = 0.0, oE = 0.0;
    csL[lid] = cL; csH[lid] = cH; csE[lid] = cE;
    for (;;)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        cL = lid ? csL[lid - 1] : 0.0;
        cH = lid ? csH[lid - 1] : 0.0;
        cE = lid ? csE[lid - 1] : 0.0;
        if (lid == CW - 1) { oL += csL[lid]; oH += csH[lid]; oE += csE[lid]; }
        if (lid == 0)
            more = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (uint t = 0; t < 4; t++)
            if (cL != 0.0 || cH != 0.0 || cE != 0.0)
            {
                double qL = rint(cL*invB), qH = rint(cH*invB);
                double dL = fma(-qL, Bd, cL), dH = fma(-qH, Bd, cH);
                double u = e[t] + fma(kd, dL, fma(-cb, dH, cE));
                cE = rint(u*invB);
                e[t] = fma(-cE, Bd, u);
                cL = qL;
                cH = qH;
            }
        if (cL != 0.0 || cH != 0.0 || cE != 0.0)
        {
#pragma unroll
            for (uint t = 4; t < CHUNK; t++)
                if (cL != 0.0 || cH != 0.0 || cE != 0.0)
                {
                    double qL = rint(cL*invB), qH = rint(cH*invB);
                    double dL = fma(-qL, Bd, cL), dH = fma(-qH, Bd, cH);
                    double u = e[t] + fma(kd, dL, fma(-cb, dH, cE));
                    cE = rint(u*invB);
                    e[t] = fma(-cE, Bd, u);
                    cL = qL;
                    cH = qH;
                }
        }
        csL[lid] = cL; csH[lid] = cH; csE[lid] = cE;
        if (cL != 0.0 || cH != 0.0 || cE != 0.0)
            more = 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (more == 0)
            break;
    }
    // z level
    uint bad = 0;
    double c3 = 0.0;
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
    {
        int p = p0 + (int)t;
        double v;
        if (p < q)
            v = fma(kd, e[t], c3);
        else if (p == q)
        {
            double a = floor(e[t]*inv_bs);
            etop[0] = (digit)a;
            v = fma(kd, fma(-a, bsd, e[t]), c3);
        }
        else
        {
            if (p < U)
                etop[p - q] = (digit)e[t];
            else if (e[t] != 0.0)
                bad = 1;
            v = c3;
        }
        c3 = rint(v*invB);
        e[t] = fma(-c3, Bd, v);
    }
    double oZ = 0.0;
    csE[lid] = c3;
    for (;;)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        c3 = lid ? csE[lid - 1] : 0.0;
        if (lid == CW - 1)
            oZ += csE[lid];
        if (lid == 0)
            more = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (uint t = 0; t < 3; t++)
            if (c3 != 0.0)
            {
                double v = e[t] + c3;
                c3 = rint(v*invB);
                e[t] = fma(-c3, Bd, v);
            }
        if (c3 != 0.0)
        {
#pragma unroll
            for (uint t = 3; t < CHUNK; t++)
                if (c3 != 0.0)
                {
                    double v = e[t] + c3;
                    c3 = rint(v*invB);
                    e[t] = fma(-c3, Bd, v);
                }
        }
        csE[lid] = c3;
        if (c3 != 0.0)
            more = 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (more == 0)
            break;
    }
    if (lid == CW - 1)
    {
        // the next group starts either well below q or above U
        bool low = p0 + (int)CHUNK <= q;
        uint gi = get_group_id(0);
        zcar[gi] = (long)oZ;
        if (ntt)
        {
            // L and H carries can be ~2^46 here: k times that would overflow, so they go on separately
            if (!low && (oL != 0.0 || oH != 0.0 || oE != 0.0))
                bad = 1;
            ecar[3*gi] = low ? (long)oL : 0;
            ecar[3*gi + 1] = low ? (long)oH : 0;
            ecar[3*gi + 2] = low ? (long)oE : 0;
        }
        else
        {
            long ge = kl*(long)oL - cbl*(long)oH + (long)oE;
            if (!low && ge != 0)
                bad = 1;
            ecar[3*gi] = 0;
            ecar[3*gi + 1] = 0;
            ecar[3*gi + 2] = low ? ge : 0;
        }
    }
    // staged through local memory so the global writes are contiguous over the work-group
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
        Ls[PADL(lid*CHUNK + t)] = e[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    int gbase = (int)(get_group_id(0)*CW*CHUNK) - off;
    for (uint t = lid; t < CW*CHUNK; t += CW)
        if (gbase + (int)t >= 0)
        {
            if (islot)
                ((__global int*)out)[gbase + (int)t] = (int)Ls[PADL(t)];
            else
                out[gbase + (int)t] = Ls[PADL(t)];
        }
    if (ntt == 2)
        me *= 1.0/2147483647.0;
    if (ntt == 3)
        me *= 1.0/(double)P3v;
    atomic_max(maxerr, as_uint((float)me));
    if (bad)
        atomic_or(err, 1u);
}
#endif

// ---------------------------------------------------------------------------------------------
// carry_fold for the hybrid engines in integer and single precision arithmetic only (no FP64,
// 1/16 rate on consumer GPUs). Same algorithm, layout and outputs as carry_fold (ntt 2 and 3).
// Divisions by B: a single precision estimate of the quotient (exact to +-1 while it is below 2^23),
// the remainder in 32-bit wraparound arithmetic (exact because it is known to be small), a second
// estimate and a final range fix. Larger quotients (up to 2^40) take one more estimate (f2l: top).
// binary base B = 2^W (W > 0, compile time): balanced digit by mask and shift, exact
#ifndef WB
#define WB 0
#endif
inline int divB_w(long v, long* q)
{
    int d = (int)(v & ((1L << WB) - 1));
    if (d >= (1 << (WB - 1)))
        d -= 1 << WB;
    *q = (v - d) >> WB;
    return d;
}
// |v/B| < 2^28: digit (balanced) and *q = quotient
inline int divB_s(long v, float Bf, float invBf, int B, int lo, int hi, long* q)
{
    if (WB > 0)
        return divB_w(v, q);
    float qf = rint(convert_float(v)*invBf);
    int qi = (int)qf;
    int d = (int)((uint)v - (uint)qi*(uint)B);
    int qc = (int)rint((float)d*invBf);
    d -= qc*B;
    qi += qc;
    if (d > hi) { d -= B; qi++; }
    else if (d < lo) { d += B; qi--; }
    *q = (long)qi;
    return d;
}
// |v| < 2^62
inline int divB_b(long v, float Bf, float invBf, int B, int lo, int hi, long* q)
{
    if (WB > 0)
        return divB_w(v, q);
    long qa = f2l(rint(convert_float(v)*invBf));
    long qb;
    int d = divB_s(v - qa*(long)B, Bf, invBf, B, lo, hi, &qb);
    *q = qa + qb;
    return d;
}
// One coefficient from the FP32 approximation f and the NTT residue a: x = Q*B + d2, |d2| < B
inline int hsplit(float f, uint a, int mode, uint sh, float scalef, float Bf, float invBf, int B, long* Q, uint* me,
                  uint P, uint PINV, uint C3, __global const uint* pw)
{
    float xf = rint(f*scalef);
    uint bits = as_uint(xf);
    int ex = (int)((bits >> 23) & 0xFF) - 150;
    uint M = (bits & 0x7FFFFF) | ((bits & 0x7F800000) ? 0x800000u : 0u);
    bool neg = (bits >> 31) != 0;
    uint mr, ar, Pm;
    if (mode == 3)
    {
        Pm = P;
        mr = ex >= 0 ? mont3(M, pw[ex & 63], P, PINV) : (ex > -24 ? M >> (uint)(-ex) : 0u);
        ar = mont3(a, C3, P, PINV);
    }
    else
    {
        Pm = PM31;
        if (ex >= 0)
        {
            uint e = (uint)ex % 31;
            mr = e ? (((M << e) & PM31) | (M >> (31 - e))) : M;
        }
        else
            mr = ex > -24 ? M >> (uint)(-ex) : 0u;
        ar = ((a << sh) & PM31) | (a >> (31 - sh));
    }
    if (neg && mr != 0)
        mr = Pm - mr;
    uint dd = ar >= mr ? ar - mr : ar + Pm - mr;
    int c = dd > (Pm >> 1) ? (int)(dd - Pm) : (int)dd;
    *me = max(*me, (uint)abs(c));
    // low 32 bits of the integer xf, then x = xf + c split by B
    uint xlo = ex >= 0 ? (ex < 32 ? M << (uint)ex : 0u) : (ex > -24 ? M >> (uint)(-ex) : 0u);
    if (neg)
        xlo = 0u - xlo;
    float q1f = rint(xf*invBf);
    float q2f = rint((fma(-q1f, Bf, xf) + (float)c)*invBf);
    long Qv = f2l(q1f) + (long)(int)q2f;
    *Q = Qv;
    return (int)(xlo + (uint)c - (uint)Qv*(uint)B);
}

__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void carry_fold_i(__global const float* inf, __global const uint* in31, __global int* out, __global long* zcar, __global long* ecar,
                  __global digit* etop, __global uint* maxerr, __global uint* err, float scalef, int m, int q, int len,
                  long kl, long cbl, int B, float invBf, int bs, int U, int off, int mode, uint sh,
                  uint P3v, uint PINV3, uint C3, __global const uint* pw3, int islot)
{
    __local int Ls[PADL(CW*CHUNK) + 1];
    __local long csL[CW], csH[CW], csE[CW];
    __local int more;
    uint lid = get_local_id(0);
    int p0 = (int)(get_group_id(0)*CW*CHUNK + lid*CHUNK) - off;
    float Bf = (float)B;
    int lo = -(B >> 1), hi = lo + B - 1;
    int e[CHUNK];
    uint me = 0;
    long cL = 0, cH = 0, cE = 0;
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
    {
        int i = p0 + (int)t;
        int jq = q + 1 + i;
        bool vl = i >= 0 && i <= q, vh = i >= 0 && jq < len;
        long QL, QH, qq;
        int dL0 = hsplit(vl ? inf[i] : 0.0f, vl ? in31[i] : 0u, mode, sh, scalef, Bf, invBf, B, &QL, &me, P3v, PINV3, C3, pw3);
        int dH0 = hsplit(vh ? inf[jq] : 0.0f, vh ? in31[jq] : 0u, mode, sh, scalef, Bf, invBf, B, &QH, &me, P3v, PINV3, C3, pw3);
        int dL = divB_s((long)dL0*m + cL, Bf, invBf, B, lo, hi, &qq);
        cL = QL*m + qq;
        int dH = divB_s((long)dH0*m + cH, Bf, invBf, B, lo, hi, &qq);
        cH = QH*m + qq;
        e[t] = divB_b(kl*dL - cbl*dH + cE, Bf, invBf, B, lo, hi, &cE);
    }
    // e level: carries of chunk lid - 1 enter chunk lid; rounds until none is left (see carry_fold)
    long oL = 0, oH = 0, oE = 0;
    csL[lid] = cL; csH[lid] = cH; csE[lid] = cE;
    for (;;)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        cL = lid ? csL[lid - 1] : 0;
        cH = lid ? csH[lid - 1] : 0;
        cE = lid ? csE[lid - 1] : 0;
        if (lid == CW - 1) { oL += csL[lid]; oH += csH[lid]; oE += csE[lid]; }
        if (lid == 0)
            more = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < CHUNK && (cL != 0 || cH != 0 || cE != 0); t++)
        {
            long qL, qH;
            int dL = divB_s(cL, Bf, invBf, B, lo, hi, &qL), dH = divB_s(cH, Bf, invBf, B, lo, hi, &qH);
            e[t] = divB_b((long)e[t] + kl*dL - cbl*dH + cE, Bf, invBf, B, lo, hi, &cE);
            cL = qL;
            cH = qH;
        }
        csL[lid] = cL; csH[lid] = cH; csE[lid] = cE;
        if (cL != 0 || cH != 0 || cE != 0)
            more = 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (more == 0)
            break;
    }
    // z level
    uint bad = 0;
    long c3 = 0;
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
    {
        int p = p0 + (int)t;
        long v;
        if (p < q)
            v = kl*e[t] + c3;
        else if (p == q)
        {
            int a = e[t]/bs, r = e[t] - a*bs;     // floor division
            if (r < 0) { r += bs; a--; }
            etop[0] = a;
            v = kl*r + c3;
        }
        else
        {
            if (p < U)
                etop[p - q] = e[t];
            else if (e[t] != 0)
                bad = 1;
            v = c3;
        }
        e[t] = divB_b(v, Bf, invBf, B, lo, hi, &c3);
    }
    long oZ = 0;
    csE[lid] = c3;
    for (;;)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        c3 = lid ? csE[lid - 1] : 0;
        if (lid == CW - 1)
            oZ += csE[lid];
        if (lid == 0)
            more = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < CHUNK && c3 != 0; t++)
            e[t] = divB_b((long)e[t] + c3, Bf, invBf, B, lo, hi, &c3);
        csE[lid] = c3;
        if (c3 != 0)
            more = 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (more == 0)
            break;
    }
    if (lid == CW - 1)
    {
        bool low = p0 + (int)CHUNK <= q;
        uint gi = get_group_id(0);
        zcar[gi] = oZ;
        if (!low && (oL != 0 || oH != 0 || oE != 0))
            bad = 1;
        ecar[3*gi] = low ? oL : 0;
        ecar[3*gi + 1] = low ? oH : 0;
        ecar[3*gi + 2] = low ? oE : 0;
    }
#pragma unroll
    for (uint t = 0; t < CHUNK; t++)
        Ls[PADL(lid*CHUNK + t)] = e[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    int gbase = (int)(get_group_id(0)*CW*CHUNK) - off;
    for (uint t = lid; t < CW*CHUNK; t += CW)
        if (gbase + (int)t >= 0)
        {
#ifdef FP64
            if (!islot)
                ((__global double*)out)[gbase + (int)t] = (double)Ls[PADL(t)];
            else
#endif
                out[gbase + (int)t] = Ls[PADL(t)];
        }
    float mef = (float)me*(1.0f/(float)(mode == 3 ? P3v : PM31));
    atomic_max(maxerr, as_uint(mef));
    if (bad)
        atomic_or(err, 1u);
}

#ifdef FP64
// Montgomery: out = low q digits of in (as doubles), zero elsewhere.
__kernel void lowpart(__global const digit* in, __global double* out, uint q)
{
    uint i = get_global_id(0);
    out[i] = i < q ? (double)in[i] : 0.0;
}

// Hybrid Montgomery: the same into a slot of 32-bit integers.
__kernel void lowpart_i(__global const digit* in, __global int* out, uint q)
{
    uint i = get_global_id(0);
    out[i] = i < q ? (int)in[i] : 0;
}

// Hybrid (FP32 + GF(p^2)) counterpart of carry_d for products without a fold (Montgomery): each
// coefficient exactly from its FP32 approximation and its residue mod p (M31: rotation by sh = 1/D,
// P3: times C3), times mulc, normalized into balanced int64 digits with carries within the group;
// the group carries go to carry_pass. maxerr: the FP32 error as a fraction of p (0.5 = ambiguous).
__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void carry_h(__global const float* inf, __global const uint* in31, __global digit* out, __global long* carries, double scale, uint w,
             __global uint* maxerr, long mulc, long B, double invB, uint nch, int f3, uint sh, uint P3v, uint PINV3, uint C3,
             __global const uint* pw3)
{
    __local digit ys[PADL(CW*CHUNK) + 1];
    __local long cs[CW];
    __local int more;
    uint lid = get_local_id(0);
    uint g = get_global_id(0);
    uint gbase = get_group_id(0)*CW*CHUNK;
    uint total = nch*CHUNK;
    uint lb = lid*CHUNK;
    double me = 0.0;
    long carry = 0;
    // (staging the loads through local memory for coalescing measured no faster)
    for (uint t = 0; t < CHUNK; t++)
    {
        uint i = gbase + lb + t;
        long x = 0;
        if (i < total)
            x = f3 ? h32_value3(inf[i], in31[i], (float)scale, &me, P3v, PINV3, C3, pw3) : h32_value(inf[i], in31[i], sh, scale, &me);
        ys[PADL(lb + t)] = (digit)balanced(x*mulc + carry, w, &carry, B, invB);
    }
    long gc = group_carry(ys, cs, &more, carry, lid, gbase + lb, total, w, B, invB);
    if (lid == CW - 1)
        carries[get_group_id(0)] = gc;
    for (uint k = lid; k < CW*CHUNK; k += CW)
        if (gbase + k < total)
            out[gbase + k] = (digit)ys[PADL(k)];
    if (g < nch)
        atomic_max(maxerr, as_uint((float)me));
}
#endif

// d = d + b with carry within chunks (Montgomery: P + m*N).
__kernel void carry_add(__global digit* d, __global const digit* b, __global long* carries, uint w, uint CH, long B, dparam invB)
{
    uint g = get_global_id(0);
    uint base = g*CH;
    long carry = 0;
    for (uint t = 0; t < CH; t++)
        d[base + t] = (digit)balanced((long)d[base + t] + b[base + t] + carry, w, &carry, B, invB);
    carries[g] = carry;
}

#ifdef FP64
// Montgomery: out = in / 2^(q*w) (digit shift); the low q digits must be zero.
__kernel void shiftdown(__global const digit* in, __global double* out, uint q, uint len, __global uint* err)
{
    uint i = get_global_id(0);
    out[i] = i + q < len ? (double)in[i + q] : 0.0;
    if (i < q && in[i] != 0)
        atomic_or(err, 4u);
}

// Hybrid Montgomery: the same into a slot of 32-bit integers.
__kernel void shiftdown_i(__global const digit* in, __global int* out, uint q, uint len, __global uint* err)
{
    uint i = get_global_id(0);
    out[i] = i + q < len ? (int)in[i + q] : 0;
    if (i < q && in[i] != 0)
        atomic_or(err, 4u);
}

// ---------------------------------------------------------------------------------------------
// Montgomery with a cyclic and a negacyclic half (MMGW): R = 2^K - 1, Q = 2^K + 1, K = 2D digits.
// x*y mod R is the hybrid's real cyclic product of length 2D (no zero padding), x*y mod Q a
// right-angle negacyclic product: z[j] = (x[j] + i*x[j + D])*zeta^j (zeta^D = i, zeta^(4D) = 1),
// a complex cyclic convolution of length D, then unweighted. With u = (x*y mod R)*(-N^-1) mod R,
// x*y + u*N = 0 mod R and z = (x*y + u*N)/R is exact; mod Q, 1/R = -1/2 (R = -2 mod Q).
// GF(p^2) products of the hybrid's field: P3 (Montgomery products, f3) or M31.
inline uint mul31(uint a, uint b) { return mod31_64((ulong)a*b); }
inline uint fmulq(uint a, uint b, int f3, uint P, uint PINV) { return f3 ? mont3(a, b, P, PINV) : mul31(a, b); }
inline uint faddq(uint a, uint b, uint P) { uint s = a + b; return s >= P ? s - P : s; }
inline uint fsubq(uint a, uint b, uint P) { return a >= b ? a - b : a + P - b; }
inline uint2 gmulq(uint2 a, uint2 b, int f3, uint P, uint PINV)
{
    uint xx = fmulq(a.x, b.x, f3, P, PINV), yy = fmulq(a.y, b.y, f3, P, PINV);
    uint xy = fmulq(a.x, b.y, f3, P, PINV), yx = fmulq(a.y, b.x, f3, P, PINV);
    return (uint2)(fsubq(xx, yy, P), faddq(xy, yx, P));
}
inline uint2 gmulcq(uint2 a, uint2 b, int f3, uint P, uint PINV)   // a*conj(b)
{
    uint xx = fmulq(a.x, b.x, f3, P, PINV), yy = fmulq(a.y, b.y, f3, P, PINV);
    uint xy = fmulq(a.x, b.y, f3, P, PINV), yx = fmulq(a.y, b.x, f3, P, PINV);
    return (uint2)(faddq(xx, yy, P), fsubq(yx, xy, P));
}
inline float2 cmulf(float2 a, float2 b) { return (float2)(fma(a.x, b.x, -a.y*b.y), fma(a.x, b.y, a.y*b.x)); }
inline float2 cmulcf(float2 a, float2 b) { return (float2)(fma(a.x, b.x, a.y*b.y), fma(a.y, b.x, -a.x*b.y)); }
// zeta^j = lo[j & 1023]*hi[j >> 10] (FP32, and the field in its product form)
#define ZF(j) cmulf(flo[(j) & 1023], fhi[(j) >> 10])
#define ZN(j) gmulq(nlo[(j) & 1023], nhi[(j) >> 10], f3, P, PINV)

// Negacyclic input: slot x (2D digits) -> of (FP32) and on (field) halves, D complex points.
__kernel void nega_weight(__global const int* x, __global float2* of, __global uint2* on,
                          __global const float2* flo, __global const float2* fhi, __global const uint2* nlo, __global const uint2* nhi,
                          uint D, int f3, uint P, uint PINV)
{
    uint j = get_global_id(0);
    if (j >= D)
        return;
    int a = x[j], b = x[j + D];
    of[j] = cmulf((float2)((float)a, (float)b), ZF(j));
    uint2 g = (uint2)(a < 0 ? (uint)(a + (int)P) : (uint)a, b < 0 ? (uint)(b + (int)P) : (uint)b);
    on[j] = gmulq(g, ZN(j), f3, P, PINV);
}

// Negacyclic pointwise product: u = (x*x or x*b) + u*n (FP32 times psc; P3: both terms carry 1/2^32).
__kernel void nega_mul(__global float2* uf, __global uint2* un, __global const float2* xf, __global const uint2* xn,
                       __global const float2* bf, __global const uint2* bn, __global const float2* nf, __global const uint2* nn,
                       int square, float psc, uint D, int f3, uint P, uint PINV)
{
    uint k = get_global_id(0);
    if (k >= D)
        return;
    float2 x = xf[k];
    float2 p = cmulf(x, square ? x : bf[k]);
    uf[k] = (p + cmulf(uf[k], nf[k]))*psc;
    uint2 xg = xn[k];
    uint2 pg = gmulq(xg, square ? xg : bn[k], f3, P, PINV), ug = gmulq(un[k], nn[k], f3, P, PINV);
    un[k] = (uint2)(faddq(pg.x, ug.x, P), faddq(pg.y, ug.y, P));
}

// c' = -(coefficients j and j + D) of the negacyclic product: unweighted, recovered exactly.
inline void nq_value(uint j, __global const float2* inf, __global const uint2* inn, __global const float2* flo, __global const float2* fhi,
                     __global const uint2* nlo, __global const uint2* nhi, int f3, uint P, uint PINV, double scale, uint sh, uint C3,
                     __global const uint* pw3, double* me, long* lo, long* hi)
{
    float2 c = cmulcf(inf[j], ZF(j));
    uint2 g = gmulcq(inn[j], ZN(j), f3, P, PINV);
    long a = f3 ? h32_value3(c.x, g.x, (float)scale, me, P, PINV, C3, pw3) : h32_value(c.x, g.x, sh, scale, me);
    long b = f3 ? h32_value3(c.y, g.y, (float)scale, me, P, PINV, C3, pw3) : h32_value(c.y, g.y, sh, scale, me);
    *lo = -a;
    *hi = -b;
}

// MMGW result: v = x*y + u*N mod Q (negacyclic coefficients c), z = v/R = -v/2 mod Q, times mulc.
// With c' = -c and b = c'[0] & 1, -v/2 = (V' + b*Q)/2: digit i = (c'[i] >> 1) + (c'[i+1] & 1)*2^(w-1),
// plus b at digit 0 and b*2^(w-1) at the top (the 'next' of digit 2D - 1 is digit 0). Work item: chunk
// j0..j0+CHUNK-1 of the low half (digits j) and of the high half (digits j + D), two carry chains.
// Group carries: carries[g] (low), carries[groups + g] (high), in digit order for carry_pass.
__kernel __attribute__((reqd_work_group_size(CW, 1, 1)))
void carry_q(__global const float2* inf, __global const uint2* inn, __global digit* out, __global long* carries, double scale, uint w,
             __global uint* maxerr, long mulc, uint D, __global const float2* flo, __global const float2* fhi,
             __global const uint2* nlo, __global const uint2* nhi, int f3, uint sh, uint P, uint PINV, uint C3, __global const uint* pw3)
{
    __local digit yl[PADL(CW*CHUNK) + 1];
    __local digit yh[PADL(CW*CHUNK) + 1];
    __local long cs[CW];
    __local int more;
    uint lid = get_local_id(0);
    uint gbase = get_group_id(0)*CW*CHUNK;
    uint lb = lid*CHUNK, j0 = gbase + lb;
    long B = 1L << w;
    double me = 0.0;
    long lo, hi, cl = 0, ch = 0;
    nq_value(j0, inf, inn, flo, fhi, nlo, nhi, f3, P, PINV, scale, sh, C3, pw3, &me, &lo, &hi);
    long b0 = 0;
    if (j0 == 0)
        b0 = lo & 1;
    for (uint t = 0; t < CHUNK; t++)
    {
        uint jn = j0 + t + 1;
        long nl, nh;
        if (jn < D)
            nq_value(jn, inf, inn, flo, fhi, nlo, nhi, f3, P, PINV, scale, sh, C3, pw3, &me, &nl, &nh);
        else
            nq_value(0, inf, inn, flo, fhi, nlo, nhi, f3, P, PINV, scale, sh, C3, pw3, &me, &nh, &nl);
        long el = (lo >> 1) + ((nl & 1) << (w - 1)) + (t == 0 ? b0 : 0);
        long eh = (hi >> 1) + ((nh & 1) << (w - 1));
        yl[PADL(lb + t)] = (digit)balanced(el*mulc + cl, w, &cl, B, 0);
        yh[PADL(lb + t)] = (digit)balanced(eh*mulc + ch, w, &ch, B, 0);
        lo = nl;
        hi = nh;
    }
    long gl = group_carry(yl, cs, &more, cl, lid, j0, D, w, B, 0);
    long gh = group_carry(yh, cs, &more, ch, lid, j0, D, w, B, 0);
    if (lid == CW - 1)
    {
        carries[get_group_id(0)] = gl;
        carries[get_num_groups(0) + get_group_id(0)] = gh;
    }
    for (uint k = lid; k < CW*CHUNK; k += CW)
    {
        out[gbase + k] = yl[PADL(k)];
        out[D + gbase + k] = yh[PADL(k)];
    }
    atomic_max(maxerr, as_uint((float)me));
}
#undef ZF
#undef ZN

// ---------------------------------------------------------------------------------------------
// IBDWT for N = k*2^n - 1 (hybrid): L = 2D digits, digit j at bit e_j = ceil(j*n/L), its width
// e_(j+1) - e_j = q0 + (f_j < r) with f_j = (-j*n) mod L and n = q0*L + r. The top digit's base is
// k*2^width: k*2^n = 1 mod N, so a carry out of it adds to digit 0. Weights a_j = 2^(f_j/L)*k^(-j/L)
// (field: theta^f_j*kappa^j with theta^L = 2, kappa^L = 1/k) make the cyclic convolution of length L
// a product mod X^L - 1/k: k times each output coefficient is an integer, and the digits of these
// integers hold k*x*y mod N (slots hold z with x = k*z, so a product needs no further reduction).
// Tables (two levels, lo: 1024 entries): .x the forward weight factor, .y the inverse one (the hi
// table of 2^(f/L) also carries the factor k of the output); field entries times 2^32 for P3.
// fl[t] = (-t*n) mod L, fh[s] = (-1024*s*n) mod L.
inline uint ib_f(uint j, __global const uint* fl, __global const uint* fh, uint L)
{
    uint f = fl[j & 1023] + fh[j >> 10];
    return f >= L ? f - L : f;
}

__kernel void ib_weight(__global const int* x, __global float* of, __global uint* on,
                        __global const float2* w2lo, __global const float2* w2hi, __global const float2* wklo, __global const float2* wkhi,
                        __global const uint2* n2lo, __global const uint2* n2hi, __global const uint2* nklo, __global const uint2* nkhi,
                        __global const uint* fl, __global const uint* fh, uint L, int f3, uint P, uint PINV)
{
    uint j = get_global_id(0);
    if (j >= L)
        return;
    uint f = ib_f(j, fl, fh, L);
    int a = x[j];
    float wf = (w2lo[f & 1023].x*w2hi[f >> 10].x)*(wklo[j & 1023].x*wkhi[j >> 10].x);
    of[j] = (float)a*wf;
    uint wn = fmulq(fmulq(n2lo[f & 1023].x, n2hi[f >> 10].x, f3, P, PINV), fmulq(nklo[j & 1023].x, nkhi[j >> 10].x, f3, P, PINV), f3, P, PINV);
    uint g = a < 0 ? (uint)(a + (int)P) : (uint)a;
    on[j] = fmulq(g, wn, f3, P, PINV);
}

// The coefficient from its FP32 approximation f and residue a, as h32_value / h32_value3 but without
// FP64 (1/16-1/64 rate on consumer GPUs): xf = rint(f*scale) = +-M*2^E is exact in single precision,
// xf mod p a rotation (M31) or a product with 2^E from the table pw (P3). *me: max |correction|.
inline long ib_value(float f, uint a, float scalef, uint* me, int f3, uint sh, uint P, uint PINV, uint C3, __global const uint* pw)
{
    float xf = rint(f*scalef);
    uint bits = as_uint(xf);
    int ex = (int)((bits >> 23) & 0xFF) - 150;
    uint M = (bits & 0x7FFFFF) | ((bits & 0x7F800000) ? 0x800000u : 0u);
    uint mr, ar;
    if (ex < 0)
        mr = ex > -24 ? M >> (uint)(-ex) : 0u;
    else if (f3)
        mr = mont3(M, pw[ex & 63], P, PINV);
    else
    {
        uint e = (uint)ex % 31;
        mr = e ? (((M << e) & PM31) | (M >> (31 - e))) : M;
    }
    ar = f3 ? mont3(a, C3, P, PINV) : (((a << sh) & PM31) | (a >> (31 - sh)));
    if ((bits >> 31) && mr != 0)
        mr = P - mr;
    uint d = ar >= mr ? ar - mr : ar + P - mr;
    int c = d > (P >> 1) ? (int)(d - P) : (int)d;
    *me = max(*me, (uint)abs(c));
    return f2l(xf) + c;
}

// Output: CHUNK digits per work item, each coefficient unweighted (times k) and recovered exactly,
// times mulc, normalized to balanced digits of its width. Work-groups of 64 (L/CHUNK a multiple of
// 64): the carry of chunk lid - 1 goes into the first two digits of chunk lid (normalized, the rest
// into the third), the work-group's last carry to carries[group] for ib_carry2.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void ib_carry(__global const float* inf, __global const uint* inn, __global int* out, __global long* carries,
              double scale, __global uint* maxerr, long mulc, uint L, uint q0, uint r, long kk,
              __global const float2* w2lo, __global const float2* w2hi, __global const float2* wklo, __global const float2* wkhi,
              __global const uint2* n2lo, __global const uint2* n2hi, __global const uint2* nklo, __global const uint2* nkhi,
              __global const uint* fl, __global const uint* fh, int f3, uint sh, uint P, uint PINV, uint C3, __global const uint* pw3)
{
    __local long lc[64];
    __local uint lme[64];
    uint g = get_global_id(0), lid = get_local_id(0);
    uint me = 0;
    float scalef = (float)scale;
    uint j = g*CHUNK;
    uint f = ib_f(j, fl, fh, L);
    uint f0 = f;
    long carry = 0;
    long d0 = 0, d1 = 0, d2 = 0;
    for (uint t = 0; t < CHUNK; t++, j++)
    {
        float wf = (w2lo[f & 1023].y*w2hi[f >> 10].y)*(wklo[j & 1023].y*wkhi[j >> 10].y);
        uint wn = fmulq(fmulq(n2lo[f & 1023].y, n2hi[f >> 10].y, f3, P, PINV), fmulq(nklo[j & 1023].y, nkhi[j >> 10].y, f3, P, PINV), f3, P, PINV);
        float c = inf[j]*wf;
        uint a = fmulq(inn[j], wn, f3, P, PINV);
        long x = ib_value(c, a, scalef, &me, f3, sh, P, PINV, C3, pw3);
        long v = x*mulc + carry;
        uint b = q0 + (f < r ? 1 : 0);
        long h, d;
        if (j == L - 1)
        {
            // top digit: base k*2^b, balanced
            long base = kk << b;
            long u = v + (base >> 1);
            h = u >= 0 ? u/base : -((-u + base - 1)/base);
            d = v - h*base;
        }
        else
        {
            h = (v + (1L << (b - 1))) >> b;
            d = v - (h << b);
        }
        if (t == 0) d0 = d; else if (t == 1) d1 = d; else if (t == 2) d2 = d; else out[j] = (int)d;
        carry = h;
        f = f >= r ? f - r : f + L - r;
    }
    lc[lid] = carry;
    lme[lid] = me;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid > 0)
    {
        long c = lc[lid - 1];
        uint b = q0 + (f0 < r ? 1 : 0);
        long v = d0 + c;
        long h = (v + (1L << (b - 1))) >> b;
        d0 = v - (h << b);
        f0 = f0 >= r ? f0 - r : f0 + L - r;
        b = q0 + (f0 < r ? 1 : 0);
        v = d1 + h;
        h = (v + (1L << (b - 1))) >> b;
        d1 = v - (h << b);
        d2 += h;
    }
    j = g*CHUNK;
    out[j] = (int)d0;
    out[j + 1] = (int)d1;
    out[j + 2] = (int)d2;
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (lid < s)
            lme[lid] = max(lme[lid], lme[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
    {
        carries[get_group_id(0)] = lc[63];
        atomic_max(maxerr, as_uint((float)lme[0]/(float)P));
    }
}

// The carry of group g - 1 (group 0: of the top group, k*2^n = 1) into the first two digits of
// group g (stride digits per group), normalized; the rest goes into the third digit unnormalized.
__kernel void ib_carry2(__global int* d, __global const long* carries, __global uint* err, uint L, uint q0, uint r,
                        __global const uint* fl, __global const uint* fh, uint stride)
{
    uint g = get_global_id(0);
    uint ng = L/stride;
    if (g >= ng)
        return;
    long c = carries[g == 0 ? ng - 1 : g - 1];
    if (c == 0)
        return;
    uint j = g*stride;
    uint f = ib_f(j, fl, fh, L);
    for (uint t = 0; t < 2; t++, j++)
    {
        uint b = q0 + (f < r ? 1 : 0);
        long v = (long)d[j] + c;
        long h = (v + (1L << (b - 1))) >> b;
        d[j] = (int)(v - (h << b));
        c = h;
        f = f >= r ? f - r : f + L - r;
    }
    long v = (long)d[j] + c;
    if (v > 0x3FFFFFFF || v < -0x3FFFFFFF)
        atomic_or(err, 1u);
    d[j] = (int)v;
}
#endif
)CLC";


// Local-memory Stockham pass of radix R = RA*RB (RA, RB in {1,2,4,8,16}), RG butterflies per
// work-group, RT work items. Compiled once per (RA, RB, RG, RT) with -D options.
static const char* lds_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double2 C;
// fma: 2 multiplies + 2 fused multiply-adds instead of 4 + 2, and one rounding fewer
inline C cmul(C a, C b) { return (C)(fma(a.x, b.x, -a.y*b.y), fma(a.x, b.y, a.y*b.x)); }
inline C cconj(C a) { return (C)(a.x, -a.y); }
inline C mul_mi(C a, int inv) { return inv ? (C)(-a.y, a.x) : (C)(a.y, -a.x); }   // forward: *(-i), inverse: *i

inline void dft2(C* v) { C a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }

inline void dft4(C* v, int inv)
{
    C s0 = v[0] + v[2], s1 = v[0] - v[2], s2 = v[1] + v[3], s3 = mul_mi(v[1] - v[3], inv);
    v[0] = s0 + s2; v[1] = s1 + s3; v[2] = s0 - s2; v[3] = s1 - s3;
}

// n = 2a + b, k = c + 4d
inline void dft8(C* v, int inv)
{
    C e[4], o[4];
    for (int a = 0; a < 4; a++) { e[a] = v[2*a]; o[a] = v[2*a + 1]; }
    dft4(e, inv);
    dft4(o, inv);
    const double h = 0.70710678118654752440;
    o[1] = cmul(o[1], (C)(h, inv ? h : -h));
    o[2] = mul_mi(o[2], inv);
    o[3] = cmul(o[3], (C)(-h, inv ? h : -h));
    for (int k = 0; k < 4; k++) { v[k] = e[k] + o[k]; v[k + 4] = e[k] - o[k]; }
}

// n = 4a + b, k = c + 4d
inline void dft16(C* v, int inv)
{
    C t[16];
    for (int b = 0; b < 4; b++)
    {
        C x[4] = { v[b], v[b + 4], v[b + 8], v[b + 12] };
        dft4(x, inv);
        for (int k = 0; k < 4; k++) t[b + 4*k] = x[k];
    }
    const double c1 = 0.92387953251128675613, s1 = 0.38268343236508977173, c2 = 0.70710678118654752440;
    const C w1 = (C)(c1, inv ? s1 : -s1), w2 = (C)(c2, inv ? c2 : -c2), w3 = (C)(s1, inv ? c1 : -c1);
    const C w6 = (C)(-c2, inv ? c2 : -c2), w9 = (C)(-c1, inv ? -s1 : s1);
    t[5] = cmul(t[5], w1);  t[9] = cmul(t[9], w2);  t[13] = cmul(t[13], w3);
    t[6] = cmul(t[6], w2);  t[10] = mul_mi(t[10], inv); t[14] = cmul(t[14], w6);
    t[7] = cmul(t[7], w3);  t[11] = cmul(t[11], w6); t[15] = cmul(t[15], w9);
    for (int k = 0; k < 4; k++)
    {
        C x[4] = { t[4*k], t[4*k + 1], t[4*k + 2], t[4*k + 3] };
        dft4(x, inv);
        for (int d = 0; d < 4; d++) v[k + 4*d] = x[d];
    }
}

#define DFT(v, n, inv) do { if ((n) == 16) dft16(v, inv); else if ((n) == 8) dft8(v, inv); else if ((n) == 4) dft4(v, inv); else if ((n) == 2) dft2(v); } while (0)

#define R (RA*RB)
#define P (R*RG)
#define PAD(i) ((i) + ((i) >> 4))

#ifdef FUSEMUL
// Real-FFT post-processing, pointwise product, pre-processing for the inverse (see realmul).
inline void unpack(C Zk, C Zm, C t, C* Xk, C* Xm)
{
    C E = 0.5*(Zk + cconj(Zm));
    C O = 0.5*(Zk - cconj(Zm));
    O = (C)(O.y, -O.x);
    C tO = cmul(t, O);
    *Xk = E + tO;
    *Xm = cconj(E - tO);
}

// Butterfly g's output k is frequency j + k*Ns. Processes the pair (g1,k1), (g2,k2) = (m, N - m).
inline void pair_product(__local C* buf, uint g1, uint k1, uint g2, uint k2, uint p0, uint N, uint Ns, uint lgNs,
                         __global const C* Bsp, __global const C* tw2, int square)
{
    uint H = RG/2;
    uint j1 = g1 < H ? p0 + g1 : (p0 + g1 - H == 0 ? Ns/2 : Ns - (p0 + g1 - H));
    uint m1 = j1 + (k1 << lgNs);
    uint i1 = PAD(g1*R + k1), i2 = PAD(g2*R + k2);
    bool first = m1 <= N - m1;           // element 1 holds the lower frequency kk <= N/2
    uint kk = first ? m1 : N - m1;
    C Zk = first ? buf[i1] : buf[i2];
    C Zm = first ? buf[i2] : buf[i1];
    C t = tw2[kk];
    C Xk, Xm, Yk, Ym;
    unpack(Zk, Zm, t, &Xk, &Xm);
    if (square)
    {
        Yk = cmul(Xk, Xk);
        Ym = cmul(Xm, Xm);
    }
    else
    {
        C Wk, Wm;
        unpack(Bsp[kk], Bsp[kk == 0 ? 0 : N - kk], t, &Wk, &Wm);
        Yk = cmul(Xk, Wk);
        Ym = cmul(Xm, Wm);
    }
    C E2 = 0.5*(Yk + cconj(Ym));
    C O2 = cmul(0.5*(Yk - cconj(Ym)), cconj(t));
    C Zk2 = E2 + (C)(-O2.y, O2.x);
    C cO2 = cconj(O2);
    C Zm2 = cconj(E2) + (C)(-cO2.y, cO2.x);
    if (i1 == i2)
        buf[i1] = Zk2;                   // kk = 0 or N/2
    else
    {
        buf[first ? i1 : i2] = Zk2;
        buf[first ? i2 : i1] = Zm2;
    }
}
#endif

__kernel __attribute__((reqd_work_group_size(RT, 1, 1)))
void fft_lds(__global const C* in, __global C* out, __global const C* tw, uint N, uint Ns, int inv,
             __global const C* twlo, __global const C* twhi, uint lbits, uint q, uint step, uint lgNs
#ifdef FUSEMUL
             , __global const C* Bsp, __global const C* tw2, int square
#endif
             )
{
    __local C buf[P + P/16 + 1];
    uint lid = get_local_id(0);
#ifdef FUSEMUL
    // Last forward pass with the pointwise product fused in. Output index m = j + k*Ns pairs with
    // N - m, which lies in butterfly Ns - j, so a work-group takes RG/2 butterflies p and their
    // mirrors (p = 0 pairs with Ns/2; both of those mirror onto themselves).
    uint p0 = get_group_id(0)*(RG/2);
#define BFLY(g) ((g) < RG/2 ? p0 + (g) : (p0 + (g) - RG/2 == 0 ? Ns/2 : Ns - (p0 + (g) - RG/2)))
#else
    uint j0 = get_group_id(0)*RG;
#define BFLY(g) (j0 + (g))
#endif
    // Ns and R are powers of two (N may be 3*2^k): q = N/R and step = N/(Ns*R) come from the host
    uint nsmask = Ns - 1;
    // load the RG butterflies (coalesced over g), apply the inter-pass twiddles
    for (uint idx = lid; idx < P; idx += RT)
    {
        uint r = idx/RG, g = idx%RG;
        uint j = BFLY(g);
        C v = in[j + r*q];
        if (Ns > 1 && r != 0)
        {
            // W^m = twlo[m mod 2^lbits] * twhi[m >> lbits]: both tables stay in cache
            uint jm = j & nsmask;
            uint m = (lgNs < 24 ? mul24(r, jm) : r*jm)*step;
            C w = cmul(twlo[m & ((1u << lbits) - 1)], twhi[m >> lbits]);
            if (inv) w = cconj(w);
            v = cmul(v, w);
        }
        buf[PAD(g*R + r)] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // DFT_R of each butterfly: n = a*RB + b, k = c + RA*d
#if RB > 1
    if (lid < RG*RB)
    {
        uint g = lid/RB, b = lid%RB;
        C v[RA];
        for (uint a = 0; a < RA; a++) v[a] = buf[PAD(g*R + a*RB + b)];
        DFT(v, RA, inv);
        for (uint cc = 0; cc < RA; cc++)
        {
            C x = v[cc];
            if (b*cc != 0)
            {
                C w = tw[mul24(b, cc)*q];
                if (inv) w = cconj(w);
                x = cmul(x, w);
            }
            buf[PAD(g*R + cc*RB + b)] = x;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        C v[RB];
        uint g = lid/RA, cc = lid%RA;
        bool act = lid < RG*RA;
        if (act)
        {
            for (uint b = 0; b < RB; b++) v[b] = buf[PAD(g*R + cc*RB + b)];
            DFT(v, RB, inv);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (act)
            for (uint d = 0; d < RB; d++) buf[PAD(g*R + cc + RA*d)] = v[d];
    }
#else
    if (lid < RG)
    {
        C v[RA];
        for (uint a = 0; a < RA; a++) v[a] = buf[PAD(lid*R + a)];
        DFT(v, RA, inv);
        for (uint k = 0; k < RA; k++) buf[PAD(lid*R + k)] = v[k];
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef FUSEMUL
    {
        uint H = RG/2;
        for (uint t = lid; t < H*R; t += RT)
        {
            uint g = t/R, k = t%R;
            if (p0 + g != 0)
                pair_product(buf, g, k, H + g, R - 1 - k, p0, N, Ns, lgNs, Bsp, tw2, square);
            else
            {
                if (k <= R/2)
                    pair_product(buf, 0, k, 0, (R - k)%R, p0, N, Ns, lgNs, Bsp, tw2, square);
                if (k < R/2)
                    pair_product(buf, H, k, H, R - 1 - k, p0, N, Ns, lgNs, Bsp, tw2, square);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    // store; order chosen so consecutive work items write consecutive addresses
    for (uint idx = lid; idx < P; idx += RT)
    {
        uint g, k;
        if (Ns >= RG) { k = idx/RG; g = idx%RG; }
        else { g = idx/R; k = idx%R; }
        uint j = BFLY(g);
        uint jm = j & nsmask;
        out[(j - jm)*R + jm + (k << lgNs)] = buf[PAD(g*R + k)];
    }
}
)CLC";

// FFT v2 ("four-step" layout, PRPLL style). D = W*H complex points, n = n1 + W*n2 (n1 < W, n2 < H).
// Forward: Stockham passes over the H dimension ("columns", element n1 + W*n2, thread per column,
// so every global access is a long contiguous run), then one pass per row: twiddle W_D^(n1*k2) and a
// W-point FFT of the contiguous row in local memory. The output is in transposed order: position
// k2*W + k1 holds frequency k2 + H*k1. The inverse runs the same steps backwards and returns natural
// order, so only the pointwise product (realmul2) needs to know the layout.

// NTT engine ("FFT61"): the v2 layout (column passes + row FFT in local memory + pair product)
// over GF(p^2), p = 2^61 - 1, i^2 = -1. All roots of unity come from the subgroup of norm-1
// elements (order p + 1 = 2^61), on which the Frobenius map a + bi -> a - bi is the inverse, just
// like complex conjugation on the unit circle: the real-input packing and the pair product work
// unchanged. Transform lengths are powers of two. Elements are kept weakly reduced (below p + 8).
// Constants (host): W4I (forward 4th root is i: 1, -i: 0), forward roots of order 8 and 16.
static const char* ntt_source = R"CLC(
#if defined(F3)
// GF(p^2), p = P3 = 57*2^25 - 1 < 2^31 (p + 1 = 3*19*2^25: roots of order 3*2^k for the
// mixed-radix lengths). Montgomery products (R = 2^32): twiddles and constants are stored as w*R,
// data stays in normal form; a product of two data values carries a factor 1/R, removed in the carry.
typedef uint E;
typedef uint2 G;
#define PM P3
inline uint madd(uint a, uint b) { uint s = a + b; return s >= PM ? s - PM : s; }
inline uint msub(uint a, uint b) { return a >= b ? a - b : a + PM - b; }
inline uint mneg(uint a) { return a ? PM - a : 0u; }
inline uint mmul(uint a, uint b)
{
    uint lo = a*b, hi = mul_hi(a, b);
    uint mh = mul_hi(lo*PINV, PM);
    return hi >= mh ? hi - mh : hi - mh + PM;
}
#elif defined(F31)
// GF(p^2), p = 2^31 - 1: canonical residues in [0, p), 32-bit arithmetic
typedef uint E;
typedef uint2 G;
#define PM 0x7FFFFFFFu
inline uint madd(uint a, uint b) { uint s = a + b; return s >= PM ? s - PM : s; }
inline uint msub(uint a, uint b) { return a >= b ? a - b : a + PM - b; }
inline uint mneg(uint a) { return a ? PM - a : 0u; }
// a*b = hi*2^32 + lo; 2^31 == 1 (mod p)
inline uint mmul(uint a, uint b) { uint lo = a*b, hi = mul_hi(a, b); uint r = (lo & PM) + ((hi << 1) | (lo >> 31)); return r >= PM ? r - PM : r; }
#define MSH8(x) ((((x) << 15) & PM) | ((x) >> 16))     // x*2^15 (x < p: a rotation of 31 bits)
#else
typedef ulong E;
typedef ulong2 G;
#define PM 0x1FFFFFFFFFFFFFFFUL
inline ulong mred(ulong x) { return (x & PM) + (x >> 61); }
inline ulong madd(ulong a, ulong b) { return mred(a + b); }
inline ulong msub(ulong a, ulong b) { return mred(a + (PM + PM) - b); }
inline ulong mneg(ulong a) { return mred((PM + PM) - a); }
// a, b < 2^61 + 8: a*b = hi*2^64 + lo, 2^64 == 8 and 2^61 == 1 (mod p)
inline ulong mmul(ulong a, ulong b) { ulong lo = a*b, hi = mul_hi(a, b); return mred((hi << 3) + (lo >> 61) + (lo & PM)); }
#define MSH8(x) mred((((x) << 30) & PM) + ((x) >> 31))  // x*2^30 for any x < 2^64
#endif
inline G gadd(G a, G b) { return (G)(madd(a.x, b.x), madd(a.y, b.y)); }
inline G gsub(G a, G b) { return (G)(msub(a.x, b.x), msub(a.y, b.y)); }
inline G gconj(G a) { return (G)(a.x, mneg(a.y)); }
// Karatsuba: 3 multiplications
inline G gmul(G a, G b)
{
    E m1 = mmul(a.x, b.x), m2 = mmul(a.y, b.y), s = mmul(madd(a.x, a.y), madd(b.x, b.y));
    return (G)(msub(m1, m2), msub(s, madd(m1, m2)));
}
inline G gmulc(G a, G b)   // a*conj(b)
{
    E m1 = mmul(a.x, b.x), m2 = mmul(a.y, b.y), s = mmul(madd(a.x, a.y), msub(b.x, b.y));
    return (G)(madd(m1, m2), msub(madd(s, m2), m1));
}
inline G gsqr(G a) { return (G)(mmul(madd(a.x, a.y), msub(a.x, a.y)), mmul(a.x, madd(a.y, a.y))); }
inline G mul_i(G a) { return (G)(mneg(a.y), a.x); }
inline G mul_mi(G a) { return (G)(a.y, mneg(a.x)); }
inline G mulw4(G a, int inv) { return (W4I ^ inv) ? mul_i(a) : mul_mi(a); }   // forward / inverse 4th root
inline G mulk(G a, G c, int inv) { return inv ? gmulc(a, c) : gmul(a, c); }
#if defined(F3)
inline G mul8(G z, int inv) { return mulk(z, (G)(W8X, W8Y), inv); }
#else
// times the 8th root w8 = 2^(bits/2)*(W8SX + i*W8SY) (forward) or its conjugate (inverse): no multiplication
inline G mul8(G z, int inv)
{
    int sy = inv ? -W8SY : W8SY;
    E re, im;
    if (W8SX > 0) { re = sy > 0 ? msub(z.x, z.y) : madd(z.x, z.y); im = sy > 0 ? madd(z.x, z.y) : msub(z.y, z.x); }
    else          { re = sy > 0 ? mneg(madd(z.x, z.y)) : msub(z.y, z.x); im = sy > 0 ? msub(z.x, z.y) : mneg(madd(z.x, z.y)); }
    return (G)(MSH8(re), MSH8(im));
}
#endif
inline G mul83(G z, int inv) { return mulw4(mul8(z, inv), inv); }
#define K8   ((G)(W8X, W8Y))
#define K83  ((G)(W83X, W83Y))
#define K16  ((G)(W161X, W161Y))
#define K163 ((G)(W163X, W163Y))
#define K169 ((G)(W169X, W169Y))

inline void dft2(G* v) { G a = v[0], b = v[1]; v[0] = gadd(a, b); v[1] = gsub(a, b); }
inline void dft4(G* v, int inv)
{
    G s0 = gadd(v[0], v[2]), s1 = gsub(v[0], v[2]), s2 = gadd(v[1], v[3]), s3 = mulw4(gsub(v[1], v[3]), inv);
    v[0] = gadd(s0, s2); v[1] = gadd(s1, s3); v[2] = gsub(s0, s2); v[3] = gsub(s1, s3);
}
inline void dft8(G* v, int inv)
{
    G e[4], o[4];
    for (int a = 0; a < 4; a++) { e[a] = v[2*a]; o[a] = v[2*a + 1]; }
    dft4(e, inv);
    dft4(o, inv);
    o[1] = mul8(o[1], inv);
    o[2] = mulw4(o[2], inv);
    o[3] = mul83(o[3], inv);
    for (int k = 0; k < 4; k++) { v[k] = gadd(e[k], o[k]); v[k + 4] = gsub(e[k], o[k]); }
}
inline void dft16(G* v, int inv)
{
    G t[16];
    for (int b = 0; b < 4; b++)
    {
        G x[4] = { v[b], v[b + 4], v[b + 8], v[b + 12] };
        dft4(x, inv);
        for (int k = 0; k < 4; k++) t[b + 4*k] = x[k];
    }
    t[5] = mulk(t[5], K16, inv);  t[9] = mul8(t[9], inv);       t[13] = mulk(t[13], K163, inv);
    t[6] = mul8(t[6], inv);       t[10] = mulw4(t[10], inv);    t[14] = mul83(t[14], inv);
    t[7] = mulk(t[7], K163, inv); t[11] = mul83(t[11], inv);    t[15] = mulk(t[15], K169, inv);
    for (int k = 0; k < 4; k++)
    {
        G x[4] = { t[4*k], t[4*k + 1], t[4*k + 2], t[4*k + 3] };
        dft4(x, inv);
        for (int d = 0; d < 4; d++) v[k + 4*d] = x[d];
    }
}
// 32 points: two 16-point DFTs and the 32nd roots W32_k (host constants)
inline void dft32(G* v, int inv)
{
    const G w[15] = { (G)(W32_1X, W32_1Y), (G)(W32_2X, W32_2Y), (G)(W32_3X, W32_3Y), (G)(W32_4X, W32_4Y), (G)(W32_5X, W32_5Y), (G)(W32_6X, W32_6Y), (G)(W32_7X, W32_7Y), (G)(W32_8X, W32_8Y), (G)(W32_9X, W32_9Y), (G)(W32_10X, W32_10Y), (G)(W32_11X, W32_11Y), (G)(W32_12X, W32_12Y), (G)(W32_13X, W32_13Y), (G)(W32_14X, W32_14Y), (G)(W32_15X, W32_15Y) };
    G e[16], o[16];
#pragma unroll
    for (int a = 0; a < 16; a++) { e[a] = v[2*a]; o[a] = v[2*a + 1]; }
    dft16(e, inv);
    dft16(o, inv);
#pragma unroll
    for (int k = 1; k < 16; k++)
        o[k] = mulk(o[k], w[k - 1], inv);
#pragma unroll
    for (int k = 0; k < 16; k++) { G x = e[k], y = o[k]; v[k] = gadd(x, y); v[k + 16] = gsub(x, y); }
}
#define DFT(v, n, inv) do { if ((n) == 32) dft32(v, inv); else if ((n) == 16) dft16(v, inv); else if ((n) == 8) dft8(v, inv); else if ((n) == 4) dft4(v, inv); else if ((n) == 2) dft2(v); } while (0)

#ifdef COLPASS
// Stockham radix-R pass over the H dimension, as v2's col_pass. CONV: the input is a slot of
// double digits (first forward pass), converted to field elements on load. HALFZ: upper half zero.
inline E tofield(long a) { return a < 0 ? (E)(a + (long)PM) : (E)a; }
#ifndef SLOTN
#define SLOTN double2                         // slot digits: double2, or int2 in the hybrid
#endif
#define R (R2*R3)
#if R3 == 3
// 3 points: w3 = x + i*y (forward), w3^2 = conj(w3); X1,2 = a + x*(b + c) +- i*y*(b - c)
inline void dft3(G* v, int inv)
{
    G s = gadd(v[1], v[2]), d = gsub(v[1], v[2]);
    E yy = inv ? mneg(K3Y) : K3Y;
    G h = gadd(v[0], (G)(mmul(s.x, K3X), mmul(s.y, K3X)));
    G iyd = mul_i((G)(mmul(d.x, yy), mmul(d.y, yy)));
    v[0] = gadd(v[0], s);
    v[1] = gadd(h, iyd);
    v[2] = gsub(h, iyd);
}
#endif
// DFT of R = R2*R3 points (as v2's dftR): radix-3 butterflies, twiddles W_R^(nm*k3), R2-point DFTs
inline void dftR(G* v, int inv, __global const G* tw, uint qR)
{
#if R3 == 1
    DFT(v, R2, inv);
#else
    G t[R];
#pragma unroll
    for (uint nm = 0; nm < R2; nm++)
    {
        G u[3] = { v[nm], v[R2 + nm], v[2*R2 + nm] };
        dft3(u, inv);
        if (nm != 0)
        {
            G w1 = tw[nm*qR], w2 = tw[2*nm*qR];
            u[1] = inv ? gmulc(u[1], w1) : gmul(u[1], w1);
            u[2] = inv ? gmulc(u[2], w2) : gmul(u[2], w2);
        }
        t[nm] = u[0]; t[R2 + nm] = u[1]; t[2*R2 + nm] = u[2];
    }
#pragma unroll
    for (uint k3 = 0; k3 < 3; k3++)
    {
        G u[R2];
#pragma unroll
        for (uint nm = 0; nm < R2; nm++) u[nm] = t[k3*R2 + nm];
        DFT(u, R2, inv);
#pragma unroll
        for (uint km = 0; km < R2; km++) v[k3 + 3*km] = u[km];
    }
#endif
}
#ifdef IBW
inline E ibw_n(uint j, __global const uint2* n2lo, __global const uint2* n2hi, __global const uint2* nklo, __global const uint2* nkhi,
               __global const uint* fl, __global const uint* fh, uint L)
{
    uint f = fl[j & 1023] + fh[j >> 10];
    if (f >= L)
        f -= L;
    return mmul(mmul(n2lo[f & 1023].x, n2hi[f >> 10].x), mmul(nklo[j & 1023].x, nkhi[j >> 10].x));
}
#define IBWP_N , __global const uint2* ibn2lo, __global const uint2* ibn2hi, __global const uint2* ibnklo, __global const uint2* ibnkhi, __global const uint* ibfl_n, __global const uint* ibfh_n, uint ibL_n
#define IBWN(j) ibw_n(j, ibn2lo, ibn2hi, ibnklo, ibnkhi, ibfl_n, ibfh_n, ibL_n)
#else
#define IBWP_N
#endif
__kernel void col_pass(__global const G* in, __global G* out, __global const G* tw, uint W, uint lgW, uint hq,
                       uint Ns, uint step, uint total, uint qR IBWP_N)
{
    uint id = get_global_id(0);
    if (id >= total)
        return;
    uint n1 = id & (W - 1), jb = id >> lgW;
    G v[R];
#ifdef HALFZ
#define RIN (R/2)
#pragma unroll
    for (uint r = R/2; r < R; r++)
        v[r] = (G)(0);
#else
#define RIN R
#endif
#ifdef CONV
    __global const SLOTN* ind = (__global const SLOTN*)in;
#pragma unroll
    for (uint r = 0; r < RIN; r++)
    {
        SLOTN x = ind[n1 + W*(jb + r*hq)];
        v[r] = (G)(tofield((long)x.x), tofield((long)x.y));
#ifdef IBW
        uint ibj = 2*(n1 + W*(jb + r*hq));
        v[r] = (G)(mmul(v[r].x, IBWN(ibj)), mmul(v[r].y, IBWN(ibj + 1)));
#endif
    }
#else
#pragma unroll
    for (uint r = 0; r < RIN; r++)
        v[r] = in[n1 + W*(jb + r*hq)];
#endif
    uint jm = jb & (Ns - 1);
    if (jm != 0)
    {
#pragma unroll
        for (uint r = 1; r < RIN; r++)
        {
            G w = tw[jm*r*step];
            v[r] = INV ? gmulc(v[r], w) : gmul(v[r], w);
        }
    }
    dftR(v, INV, tw, qR);
    uint ob = (jb - jm)*R + jm;
#pragma unroll
    for (uint k = 0; k < R; k++)
        out[n1 + W*(ob + k*Ns)] = v[k];
}
#endif

#ifdef ROWPASS
// RW-point FFT of each contiguous row k2 in local memory (one work-group per row, RT threads holding
// RV = RW/RT values each). The four-step twiddle W_D^(n1*k2) is merged into the stage twiddles:
//     y[k1] = sum x[n1]*E(n1*(k1 + k2/H)/RW),   E(x) = exp(-2*pi*i*x),
// is a Stockham chain whose stage (Ns, Rs) multiplies input r of butterfly jm by
// E(r*(jm + k2/H)/(Ns*Rs)). These RW - 1 factors per row come from a host table,
// rtw[k2*RW + Ns - 1 + jm*(Rs - 1) + r - 1], so a point costs one multiplication per stage and
// nothing else. The first stage works on the values as loaded, the last one stores to global memory.
// The inverse is the exact conjugate transpose: stages in reverse order, each one gather, inverse
// DFT, conjugate twiddle. STAGES_F / STAGES_I expand to FSTAGE / ISTAGE(Ns, Rs, first, last).
// Bank-conflict avoidance in local memory: padding (one slot per 16), or with SWZ an XOR swizzle
// within blocks of 16 (no extra space: two 1024-point rows then fit in 32 KB). LROW = row stride.
#ifdef SWZ
#define PADR(i) ((i) ^ (((i) >> 4) & 15))
#define LROW (RW)
#else
#define PADR(i) ((i) + ((i) >> 4))
#define LROW (PADR(RW) + 1)
#endif
#define FSTAGE(NS, RS, FIRST, LAST) \
    { \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            uint ti = NS - 1 + jm*(RS - 1); \
            _Pragma("unroll") for (uint r = 0; r < RS; r++) \
            { \
                G x = FIRST ? xs[u + r*(RV/RS)] : BUF(b + r*(RW/RS)); \
                if (r != 0) \
                    x = gmul(x, T[ti + r - 1]); \
                v[u*RS + r] = x; \
            } \
            DFT(v + u*RS, RS, 0); \
        } \
        if (!FIRST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            _Pragma("unroll") for (uint k = 0; k < RS; k++) \
            { \
                uint o = (b - jm)*RS + jm + k*NS; \
                if (LAST) out[base + o] = v[u*RS + k]; \
                else BUF(o) = v[u*RS + k]; \
            } \
        } \
        if (!LAST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
    }
#define ISTAGE(NS, RS, FIRST, LAST) \
    { \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            uint ti = NS - 1 + jm*(RS - 1); \
            _Pragma("unroll") for (uint k = 0; k < RS; k++) \
                v[u*RS + k] = FIRST ? xs[u + k*(RV/RS)] : BUF((b - jm)*RS + jm + k*NS); \
            DFT(v + u*RS, RS, 1); \
            _Pragma("unroll") for (uint r = 1; r < RS; r++) \
                v[u*RS + r] = gmulc(v[u*RS + r], T[ti + r - 1]); \
        } \
        if (!FIRST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            _Pragma("unroll") for (uint r = 0; r < RS; r++) \
            { \
                uint o = b + r*(RW/RS); \
                if (LAST) out[base + o] = v[u*RS + r]; \
                else BUF(o) = v[u*RS + r]; \
            } \
        } \
        if (!LAST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
    }

// RG rows per work-group (RG*RT threads), so short rows still fill whole waves.
#define BUF(i) buf[boff + PADR(i)]
#define ROW_KERNEL(NAME, STAGES) __kernel __attribute__((reqd_work_group_size(RG*RT, 1, 1))) void NAME(__global const G* in, __global G* out, __global const G* rtw) {     __local G buf[RG*(PADR(RW) + 1)];     uint lid = get_local_id(0);     uint lr = lid % RT;     uint boff = (lid/RT)*(PADR(RW) + 1);     uint base = (get_group_id(0)*RG + lid/RT)*RW;     __global const G* T = rtw + base;     G xs[RV], v[RV];     _Pragma("unroll") for (uint i = 0; i < RV; i++)         xs[i] = in[base + lr + i*RT];     STAGES }
ROW_KERNEL(row_fwd, STAGES_F)
ROW_KERNEL(row_inv, STAGES_I)
#ifdef TAILK   // the fused tail only in its own build: two rows can exceed the local memory of row-only builds

// Fused tail (field version of v2's row_tail, as gpuowl's tailSquare): forward row FFT, pointwise product and inverse row FFT of a
// row k2 and its partner row H - k2 in one work-group (group 0: rows 0 and H/2, each paired with
// itself), so the spectrum never goes to global memory. Pair product as in realmul2.
inline void tail_pair(__local G* buf, uint ia, uint ib, uint sa, uint sb, uint fa, uint D, G u, __global const G* Bsp, int square)
{
    bool low = 2*fa <= D;
    uint ik = low ? ia : ib, im = low ? ib : ia;
    G Zk = buf[ik], Zm = buf[im];
    G d = gsub(Zk, gconj(Zm));
    G g, Rk, Rm;
    if (square)
    {
        g = gmul(u, gsqr(d));
        Rk = gsqr(Zk);
        Rm = gsqr(Zm);
    }
    else
    {
        G Bk = Bsp[low ? sa : sb], Bm = Bsp[low ? sb : sa];
        g = gmul(u, gmul(d, gsub(Bk, gconj(Bm))));
        Rk = gmul(Zk, Bk);
        Rm = gmul(Zm, Bm);
    }
    buf[ik] = gadd(Rk, g);
    if (ik != im)
        buf[im] = gadd(Rm, gconj(g));
}

__kernel __attribute__((reqd_work_group_size(2*RT, 1, 1)))
void row_tail(__global const G* in, __global G* out, __global const G* rtw, __global const G* Bsp, __global const G* u2s,
              uint H, uint D, int square)
{
    __local G buf[2*LROW];
    const uint L = LROW;
    uint lid = get_local_id(0);
    uint lr = lid % RT, sel = lid/RT;
    uint boff = sel*L;
    uint g = get_group_id(0);
    uint rb = g == 0 ? H/2 : H - g;
    uint base = (sel == 0 ? g : rb)*RW;
    __global const G* T = rtw + base;
    G xs[RV], v[RV];
#pragma unroll
    for (uint i = 0; i < RV; i++)
        xs[i] = in[base + lr + i*RT];
    STAGES_FT
    if (g != 0)
    {
        // element k1 of row g pairs with element RW - 1 - k1 of row H - g
#pragma unroll
        for (uint j = 0; j < RV/2; j++)
        {
            uint k1 = lid + j*2*RT, k1b = RW - 1 - k1;
            tail_pair(buf, PADR(k1), L + PADR(k1b), g*RW + k1, rb*RW + k1b, g + H*k1, D, u2s[g*RW + k1], Bsp, square);
        }
    }
    else
    {
        // row 0: k1 <-> (RW - k1) mod RW, k1 = 0..RW/2; row H/2: k1 <-> RW - 1 - k1, k1 < RW/2
        uint r = sel == 0 ? 0 : H/2;
#pragma unroll
        for (uint j = 0; j < RV/2; j++)
        {
            uint k1 = lr + j*RT;
            uint k1b = sel == 0 ? (RW - k1) & (RW - 1) : RW - 1 - k1;
            tail_pair(buf, boff + PADR(k1), boff + PADR(k1b), r*RW + k1, r*RW + k1b, r + H*k1, D, u2s[r*RW + k1], Bsp, square);
        }
        if (sel == 0 && lr == 0)
            tail_pair(buf, PADR(RW/2), PADR(RW/2), RW/2, RW/2, H*(RW/2), D, u2s[RW/2], Bsp, square);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint i = 0; i < RV; i++)
        xs[i] = BUF(lr + i*RT);
    barrier(CLK_LOCAL_MEM_FENCE);
    STAGES_I
}
#endif   // TAILK
#endif

#ifdef REALMUL
// Pair product in the transposed layout, as v2's realmul2 (u = -(1 + t^2)/4 from the host).
__kernel void realmul(__global G* A, __global const G* B, __global const G* u2s, uint W, uint lgW, uint H, uint D, int square, uint total)
{
    uint s = get_global_id(0);
    if (s >= total)
        return;
    uint k2 = s >> lgW, k1 = s & (W - 1);
    uint p2, p1;
    if (k2 == 0)
    {
        if (2*k1 > W) return;
        p2 = 0; p1 = (W - k1) & (W - 1);
    }
    else if (2*k2 == H)
    {
        if (2*k1 >= W) return;
        p2 = k2; p1 = W - 1 - k1;
    }
    else
    {
        p2 = H - k2; p1 = W - 1 - k1;
    }
    uint f = k2 + H*k1;
    uint sp = p2*W + p1;
    bool low = 2*f <= D;
    uint ik = low ? s : sp, im = low ? sp : s;
    G u = u2s[s];
    G Zk = A[ik], Zm = A[im];
    G d = gsub(Zk, gconj(Zm));
    G g, Rk, Rm;
    if (square)
    {
        g = gmul(u, gsqr(d));
        Rk = gsqr(Zk);
        Rm = gsqr(Zm);
    }
    else
    {
        G Bk = B[ik], Bm = B[im];
        g = gmul(u, gmul(d, gsub(Bk, gconj(Bm))));
        Rk = gmul(Zk, Bk);
        Rm = gmul(Zm, Bm);
    }
    A[ik] = gadd(Rk, g);
    if (ik != im)
        A[im] = gadd(Rm, gconj(g));
}
#endif
)CLC";

static const char* v2_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double2 C;
inline C cmul(C a, C b) { return (C)(fma(a.x, b.x, -a.y*b.y), fma(a.x, b.y, a.y*b.x)); }
inline C cconj(C a) { return (C)(a.x, -a.y); }
inline C cmulc(C a, C b) { return (C)(fma(a.x, b.x, a.y*b.y), fma(a.y, b.x, -a.x*b.y)); }   // a*conj(b)
inline C mul_mi(C a, int inv) { return inv ? (C)(-a.y, a.x) : (C)(a.y, -a.x); }   // forward: *(-i), inverse: *i

inline void dft2(C* v) { C a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }
inline void dft4(C* v, int inv)
{
    C s0 = v[0] + v[2], s1 = v[0] - v[2], s2 = v[1] + v[3], s3 = mul_mi(v[1] - v[3], inv);
    v[0] = s0 + s2; v[1] = s1 + s3; v[2] = s0 - s2; v[3] = s1 - s3;
}
// Constant twiddles w = c*(1 + i*t) are applied as a rotation (x.x - t*x.y, x.y + t*x.x) and the
// factor c goes into the fused multiply-add of the butterfly that follows: 2 operations less
// than a full complex multiplication each. sg = -1 forward, +1 inverse (t = sg*tan(angle)).
inline C rotc(C x, double t) { return (C)(fma(-t, x.y, x.x), fma(t, x.x, x.y)); }
inline C fmac(double c, C x, C a) { return (C)(fma(c, x.x, a.x), fma(c, x.y, a.y)); }
inline void dft8(C* v, int inv)
{
    C e[4], o[4];
    for (int a = 0; a < 4; a++) { e[a] = v[2*a]; o[a] = v[2*a + 1]; }
    dft4(e, inv);
    dft4(o, inv);
    const double h = 0.70710678118654752440;
    const double sg = inv ? 1.0 : -1.0;
    C y1 = rotc(o[1], sg), y2 = mul_mi(o[2], inv), y3 = rotc(o[3], -sg);    // W8^1 = h*(1 + i*sg), W8^3 = -h*(1 - i*sg)
    v[0] = e[0] + o[0]; v[4] = e[0] - o[0];
    v[1] = fmac(h, y1, e[1]); v[5] = fmac(-h, y1, e[1]);
    v[2] = e[2] + y2; v[6] = e[2] - y2;
    v[3] = fmac(-h, y3, e[3]); v[7] = fmac(h, y3, e[3]);
}
// 4-point DFT of (x0, ca*(1 + i*ta)*x1, cb*y2, cc*(1 + i*tc)*x3) with rca = cc/ca; y2 is already rotated.
inline void dft4t(C x0, C x1, C y2, C x3, double ca, double ta, double cb, double rca, double tc, int inv, C* v, int k)
{
    C y1 = rotc(x1, ta), y3 = rotc(x3, tc);
    C s0 = fmac(cb, y2, x0), s1 = fmac(-cb, y2, x0);
    C s2 = fmac(rca, y3, y1), s3 = mul_mi(fmac(-rca, y3, y1), inv);
    v[k] = fmac(ca, s2, s0); v[k + 8] = fmac(-ca, s2, s0);
    v[k + 4] = fmac(ca, s3, s1); v[k + 12] = fmac(-ca, s3, s1);
}
inline void dft16(C* v, int inv)
{
    C t[16];
    for (int b = 0; b < 4; b++)
    {
        C x[4] = { v[b], v[b + 4], v[b + 8], v[b + 12] };
        dft4(x, inv);
        for (int k = 0; k < 4; k++) t[b + 4*k] = x[k];
    }
    // row k holds t[4k + b], to be multiplied by W16^(b*k) before the 4-point DFT over b
    const double C1 = 0.92387953251128675613, C2 = 0.70710678118654752440, C3 = 0.38268343236508977173;
    const double T1 = 0.41421356237309504880, T3 = 2.41421356237309504880;
    const double sg = inv ? 1.0 : -1.0;
    {
        C x[4] = { t[0], t[1], t[2], t[3] };
        dft4(x, inv);
        v[0] = x[0]; v[4] = x[1]; v[8] = x[2]; v[12] = x[3];
    }
    // k = 1: W^1 = C1*(1 + i*T1), W^2 = C2*(1 + i), W^3 = C3*(1 + i*T3)   (tangents times sg)
    dft4t(t[4], t[5], rotc(t[6], sg), t[7], C1, sg*T1, C2, C3/C1, sg*T3, inv, v, 1);
    // k = 2: W^2 = C2*(1 + i), W^4 = i*sg, W^6 = -C2*(1 - i)
    dft4t(t[8], t[9], mul_mi(t[10], inv), t[11], C2, sg, 1.0, -1.0, -sg, inv, v, 2);
    // k = 3: W^3 = C3*(1 + i*T3), W^6 = -C2*(1 - i), W^9 = -C1*(1 + i*T1)
    dft4t(t[12], t[13], rotc(t[14], -sg), t[15], C3, sg*T3, -C2, -C1/C3, sg*T1, inv, v, 3);
}
// 32 points: two 16-point DFTs (even / odd inputs), twiddles W32^k, butterflies
inline void dft32(C* v, int inv)
{
    const double cs[9] = { 1.0, 0.98078528040323044913, 0.92387953251128675613, 0.83146961230254523708,
                           0.70710678118654752440, 0.55557023301960222474, 0.38268343236508977173, 0.19509032201612826785, 0.0 };
    C e[16], o[16];
#pragma unroll
    for (int a = 0; a < 16; a++) { e[a] = v[2*a]; o[a] = v[2*a + 1]; }
    dft16(e, inv);
    dft16(o, inv);
#pragma unroll
    for (int k = 1; k < 16; k++)
    {
        // W32^k = cos(pi*k/16) -+ i*sin(pi*k/16)
        double c = k <= 8 ? cs[k] : -cs[16 - k];
        double sn = k <= 8 ? cs[8 - k] : cs[k - 8];
        o[k] = cmul(o[k], (C)(c, inv ? sn : -sn));
    }
#pragma unroll
    for (int k = 0; k < 16; k++) { C x = e[k], y = o[k]; v[k] = x + y; v[k + 16] = x - y; }
}
#define DFT(v, n, inv) do { if ((n) == 32) dft32(v, inv); else if ((n) == 16) dft16(v, inv); else if ((n) == 8) dft8(v, inv); else if ((n) == 4) dft4(v, inv); else if ((n) == 2) dft2(v); } while (0)

// W_D^m from the two-level table (both parts stay in cache)
inline C twid(__global const C* twlo, __global const C* twhi, uint lbits, uint m)
{
    return cmul(twlo[m & ((1u << lbits) - 1)], twhi[m >> lbits]);
}

typedef double REALT;
#ifdef COLPASS
#define R (R2*R3)
// DFT of R = R3*R2 points in registers, natural order. tw[m*qR] = W_R^m.
inline void dftR(C* v, int inv, __global const C* tw, uint qR)
{
#if R3 == 1
    DFT(v, R2, inv);
#else
    // n = R2*n3 + nm, k = k3 + 3*km: radix-3 butterflies, twiddles W_R^(nm*k3), then R2-point DFTs
    const double s3 = 0.86602540378443864676;
    C t[R];
#pragma unroll
    for (uint nm = 0; nm < R2; nm++)
    {
        C v0 = v[nm], v1 = v[R2 + nm], v2 = v[2*R2 + nm];
        C s = v1 + v2, d = v1 - v2;
        C h = v0 - 0.5*s;
        C id = (C)(-d.y, d.x)*(inv ? s3 : -s3);
        C x1 = h + id, x2 = h - id;
        if (nm != 0)
        {
            C w1 = tw[nm*qR], w2 = tw[2*nm*qR];
            if (inv) { w1 = cconj(w1); w2 = cconj(w2); }
            x1 = cmul(x1, w1);
            x2 = cmul(x2, w2);
        }
        t[nm] = v0 + s;
        t[R2 + nm] = x1;
        t[2*R2 + nm] = x2;
    }
#pragma unroll
    for (uint k3 = 0; k3 < 3; k3++)
    {
        C u[R2];
#pragma unroll
        for (uint nm = 0; nm < R2; nm++) u[nm] = t[k3*R2 + nm];
        DFT(u, R2, inv);
#pragma unroll
        for (uint km = 0; km < R2; km++) v[k3 + 3*km] = u[km];
    }
#endif
}

// One Stockham radix-R pass over the H dimension. Thread (n1, jb): n1 is the column (fastest, so
// global accesses are contiguous over the work-group), jb the butterfly. hq = H/R, step = D/(Ns*R).
// INV: direction (compile time). HALFZ: the upper half of the input (n2 >= H/2) is known to be zero
// (first forward pass of a value: only the low D of the 2D reals are used).
#ifdef FANCY
// Radix-8 butterfly with its input twiddles in tangent form (as gpuowl's "fancy" trig): w_r =
// i^m_r * c_r*(1 + i*t_r), |angle| <= 45 degrees (c_r >= 0.707, |t_r| <= 1). The quadrant is a free
// swap, the rotation 2 FMAs, and the scales c_r go into the FMAs of the butterfly, which were
// additions anyway: 66 instead of 80 FP64 operations. F (8 per jm): t1..t7, c4, c2, c6/c2, c1,
// c5/c1, c7/c3, c3/c1, h*c1, quadrant bits. Inverse: conjugate twiddles (t -> -t, m -> -m).
inline C qrot(C x, uint m) { C a = (m & 1) ? (C)(-x.y, x.x) : x; return (m & 2) ? -a : a; }
inline void fancy8(C* v, __global const C* F, int inv)
{
    C f0 = F[0], f1 = F[1], f2 = F[2], f3 = F[3], f4 = F[4], f5 = F[5], f6 = F[6], f7 = F[7];
    uint fl = (uint)as_ulong(f7.y);
    double ts = inv ? -1.0 : 1.0;
    double t[8] = { 0.0, f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x };
    C y[8];
    y[0] = v[0];
#pragma unroll
    for (uint r = 1; r < 8; r++)
    {
        uint m = (fl >> (2*(r - 1))) & 3;
        if (inv) m = (4 - m) & 3;
        y[r] = rotc(qrot(v[r], m), ts*t[r]);
    }
    double c4 = f3.y, c2 = f4.x, r62 = f4.y, c1 = f5.x, r51 = f5.y, r73 = f6.x, r31 = f6.y, hc1 = f7.x;
    C s0 = fmac(c4, y[4], y[0]), s1 = fmac(-c4, y[4], y[0]);
    C s2 = fmac(r62, y[6], y[2]), s3 = mul_mi(fmac(-r62, y[6], y[2]), inv);
    C e0 = fmac(c2, s2, s0), e2 = fmac(-c2, s2, s0), e1 = fmac(c2, s3, s1), e3 = fmac(-c2, s3, s1);
    C p0 = fmac(r51, y[5], y[1]), p1 = fmac(-r51, y[5], y[1]);
    C q2 = fmac(r73, y[7], y[3]), q3 = mul_mi(fmac(-r73, y[7], y[3]), inv);
    C o0 = fmac(r31, q2, p0), o2 = fmac(-r31, q2, p0), o1 = fmac(r31, q3, p1), o3 = fmac(-r31, q3, p1);
    double sg = inv ? 1.0 : -1.0;
    C g1 = rotc(o1, sg), g2 = mul_mi(o2, inv), g3 = rotc(o3, -sg);
    v[0] = fmac(c1, o0, e0); v[4] = fmac(-c1, o0, e0);
    v[1] = fmac(hc1, g1, e1); v[5] = fmac(-hc1, g1, e1);
    v[2] = fmac(c1, g2, e2); v[6] = fmac(-c1, g2, e2);
    v[3] = fmac(-hc1, g3, e3); v[7] = fmac(hc1, g3, e3);
}
#endif

// IBW (hybrid IBDWT, first forward pass from a slot): digit j times its weight (see ib_weight)
#ifdef IBW
inline float ibw_f(uint j, __global const float2* w2lo, __global const float2* w2hi, __global const float2* wklo, __global const float2* wkhi,
                   __global const uint* fl, __global const uint* fh, uint L)
{
    uint f = fl[j & 1023] + fh[j >> 10];
    if (f >= L)
        f -= L;
    return (w2lo[f & 1023].x*w2hi[f >> 10].x)*(wklo[j & 1023].x*wkhi[j >> 10].x);
}
#define IBWP_F , __global const float2* ibw2lo, __global const float2* ibw2hi, __global const float2* ibwklo, __global const float2* ibwkhi, __global const uint* ibfl, __global const uint* ibfh, uint ibL
#define IBWF(j) ibw_f(j, ibw2lo, ibw2hi, ibwklo, ibwkhi, ibfl, ibfh, ibL)
#else
#define IBWP_F
#endif
__kernel void col_pass(__global const C* in, __global C* out, __global const C* tw, uint W, uint lgW, uint hq,
                       uint Ns, uint step, uint qR, uint total, __global const C* F IBWP_F)
{
    uint id = get_global_id(0);
    if (id >= total)
        return;
    uint n1 = id & (W - 1), jb = id >> lgW;
    C v[R];
#ifdef HALFZ
#define RIN (R/2)
#pragma unroll
    for (uint r = R/2; r < R; r++)
        v[r] = (C)(0.0, 0.0);
#else
#define RIN R
#endif
#ifdef CONVD
    // FP32 transform, first pass: the input is a slot of double digits (SLOTT, not converted)
    __global const SLOTT* ind = (__global const SLOTT*)in;
#pragma unroll
    for (uint r = 0; r < RIN; r++)
    {
        SLOTT x = ind[n1 + W*(jb + r*hq)];
        v[r] = (C)((REALT)x.x, (REALT)x.y);
#ifdef IBW
        uint ibj = 2*(n1 + W*(jb + r*hq));
        v[r].x *= IBWF(ibj);
        v[r].y *= IBWF(ibj + 1);
#endif
    }
#else
#pragma unroll
    for (uint r = 0; r < RIN; r++)
        v[r] = in[n1 + W*(jb + r*hq)];
#endif
    uint jm = jb & (Ns - 1);
#ifdef FANCY
    fancy8(v, F + 8*jm, INV);
#else
    if (jm != 0)
    {
#pragma unroll
        for (uint r = 1; r < RIN; r++)
        {
            C w = tw[jm*r*step];
            v[r] = INV ? cmulc(v[r], w) : cmul(v[r], w);
        }
    }
    dftR(v, INV, tw, qR);
#endif
    uint ob = (jb - jm)*R + jm;
#pragma unroll
    for (uint k = 0; k < R; k++)
        out[n1 + W*(ob + k*Ns)] = v[k];
}
#endif

#ifdef ROWPASS
// RW-point FFT of each contiguous row k2 in local memory (one work-group per row, RT threads holding
// RV = RW/RT values each). The four-step twiddle W_D^(n1*k2) is merged into the stage twiddles:
//     y[k1] = sum x[n1]*E(n1*(k1 + k2/H)/RW),   E(x) = exp(-2*pi*i*x),
// is a Stockham chain whose stage (Ns, Rs) multiplies input r of butterfly jm by
// E(r*(jm + k2/H)/(Ns*Rs)). These RW - 1 factors per row come from a host table,
// rtw[k2*RW + Ns - 1 + jm*(Rs - 1) + r - 1], so a point costs one multiplication per stage and
// nothing else. The first stage works on the values as loaded, the last one stores to global memory.
// The inverse is the exact conjugate transpose: stages in reverse order, each one gather, inverse
// DFT, conjugate twiddle. STAGES_F / STAGES_I expand to FSTAGE / ISTAGE(Ns, Rs, first, last).
// Bank-conflict avoidance in local memory: padding (one slot per 16), or with SWZ an XOR swizzle
// within blocks of 16 (no extra space: two 1024-point rows then fit in 32 KB). LROW = row stride.
#ifdef SWZ
#define PADR(i) ((i) ^ (((i) >> 4) & 15))
#define LROW (RW)
#else
#define PADR(i) ((i) + ((i) >> 4))
#define LROW (PADR(RW) + 1)
#endif
// Row twiddles from the per-row table T, or with TWC as one product of a stage table shared by all
// rows, ST[Ns - 1 + jm*(Rs - 1) + r - 1] = W^(r*jm*H*q), and a small per-row table,
// RTr[16*stage + r - 1] = W^(r*k2*q). TWC is for the FP32 transform: the full table was as large as
// the data, the extra product is nearly free.
#ifdef TWC
#define TWINIT(NS, jm)
#define TW(ti, r, NS) cmul(ST[(ti) + (r) - 1], RTr[16*((31 - clz((uint)(NS)))/LGRV) + (r) - 1])
#define TWSTEP
#define TWPTRS __global const C* ST = rtw; __global const C* RTr = rtr + 64*(base/RW);
#else
#define TWINIT(NS, jm)
#define TW(ti, r, NS) T[(ti) + (r) - 1]
#define TWSTEP
#define TWPTRS
#endif
#define FSTAGE(NS, RS, FIRST, LAST) \
    { \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            uint ti = NS - 1 + jm*(RS - 1); \
            TWINIT(NS, jm) \
            _Pragma("unroll") for (uint r = 0; r < RS; r++) \
            { \
                C x = FIRST ? xs[u + r*(RV/RS)] : BUF(b + r*(RW/RS)); \
                if (r != 0) \
                { \
                    x = cmul(x, TW(ti, r, NS)); \
                    TWSTEP \
                } \
                v[u*RS + r] = x; \
            } \
            DFT(v + u*RS, RS, 0); \
        } \
        if (!FIRST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            _Pragma("unroll") for (uint k = 0; k < RS; k++) \
            { \
                uint o = (b - jm)*RS + jm + k*NS; \
                if (LAST) out[base + o] = v[u*RS + k]; \
                else BUF(o) = v[u*RS + k]; \
            } \
        } \
        if (!LAST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
    }
#define ISTAGE(NS, RS, FIRST, LAST) \
    { \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            uint jm = b & (NS - 1); \
            uint ti = NS - 1 + jm*(RS - 1); \
            TWINIT(NS, jm) \
            _Pragma("unroll") for (uint k = 0; k < RS; k++) \
                v[u*RS + k] = FIRST ? xs[u + k*(RV/RS)] : BUF((b - jm)*RS + jm + k*NS); \
            DFT(v + u*RS, RS, 1); \
            _Pragma("unroll") for (uint r = 1; r < RS; r++) \
            { \
                v[u*RS + r] = cmulc(v[u*RS + r], TW(ti, r, NS)); \
                TWSTEP \
            } \
        } \
        if (!FIRST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") for (uint u = 0; u < RV/RS; u++) \
        { \
            uint b = lr + u*RT; \
            _Pragma("unroll") for (uint r = 0; r < RS; r++) \
            { \
                uint o = b + r*(RW/RS); \
                if (LAST) out[base + o] = v[u*RS + r]; \
                else BUF(o) = v[u*RS + r]; \
            } \
        } \
        if (!LAST) \
            barrier(CLK_LOCAL_MEM_FENCE); \
    }

// RG rows per work-group (RG*RT threads), so short rows still fill whole waves.
#define BUF(i) buf[boff + PADR(i)]
#define ROW_KERNEL(NAME, STAGES) __kernel __attribute__((reqd_work_group_size(RG*RT, 1, 1))) void NAME(__global const C* in, __global C* out, __global const C* rtw, __global const C* rtr) {     __local C buf[RG*(PADR(RW) + 1)];     uint lid = get_local_id(0);     uint lr = lid % RT;     uint boff = (lid/RT)*(PADR(RW) + 1);     uint base = (get_group_id(0)*RG + lid/RT)*RW;     __global const C* T = rtw + base;     TWPTRS     C xs[RV], v[RV];     _Pragma("unroll") for (uint i = 0; i < RV; i++)         xs[i] = in[base + lr + i*RT];     STAGES }
ROW_KERNEL(row_fwd, STAGES_F)
ROW_KERNEL(row_inv, STAGES_I)
#ifdef TAILK   // the fused tail only in its own build: two rows can exceed the local memory of row-only builds

// Fused tail (as gpuowl's tailSquare): forward row FFT, pointwise product and inverse row FFT of a
// row k2 and its partner row H - k2 in one work-group (group 0: rows 0 and H/2, each paired with
// itself), so the spectrum never goes to global memory. Pair product as in realmul2.
inline C csqr(C a) { return (C)(fma(a.x, a.x, -a.y*a.y), (a.x + a.x)*a.y); }
inline void tail_pair(__local C* buf, uint ia, uint ib, uint sa, uint sb, uint fa, uint D, C u, __global const C* Bsp, int square)
{
    bool low = 2*fa <= D;
    uint ik = low ? ia : ib, im = low ? ib : ia;
    C Zk = buf[ik], Zm = buf[im];
#ifdef PSCS
    { double f = square ? PSCS : PSCM; Zk *= f; Zm *= f; }
#endif
    C d = Zk - cconj(Zm);
    C g, Rk, Rm;
    if (square)
    {
        g = cmul(u, csqr(d));
        Rk = csqr(Zk);
        Rm = csqr(Zm);
    }
    else
    {
        C Bk = Bsp[low ? sa : sb], Bm = Bsp[low ? sb : sa];
        g = cmul(u, cmul(d, Bk - cconj(Bm)));
        Rk = cmul(Zk, Bk);
        Rm = cmul(Zm, Bm);
    }
    buf[ik] = Rk + g;
    if (ik != im)
        buf[im] = Rm + cconj(g);
}

__kernel __attribute__((reqd_work_group_size(2*RT, 1, 1)))
void row_tail(__global const C* in, __global C* out, __global const C* rtw, __global const C* Bsp, __global const C* u2s,
              uint H, uint D, int square, __global const C* rtr)
{
    __local C buf[2*LROW];
    const uint L = LROW;
    uint lid = get_local_id(0);
    uint lr = lid % RT, sel = lid/RT;
    uint boff = sel*L;
    uint g = get_group_id(0);
    uint rb = g == 0 ? H/2 : H - g;
    uint base = (sel == 0 ? g : rb)*RW;
    __global const C* T = rtw + base;
    TWPTRS
    C xs[RV], v[RV];
#pragma unroll
    for (uint i = 0; i < RV; i++)
        xs[i] = in[base + lr + i*RT];
    STAGES_FT
    if (g != 0)
    {
        // element k1 of row g pairs with element RW - 1 - k1 of row H - g
#pragma unroll
        for (uint j = 0; j < RV/2; j++)
        {
            uint k1 = lid + j*2*RT, k1b = RW - 1 - k1;
            tail_pair(buf, PADR(k1), L + PADR(k1b), g*RW + k1, rb*RW + k1b, g + H*k1, D, u2s[g*RW + k1], Bsp, square);
        }
    }
    else
    {
        // row 0: k1 <-> (RW - k1) mod RW, k1 = 0..RW/2; row H/2: k1 <-> RW - 1 - k1, k1 < RW/2
        uint r = sel == 0 ? 0 : H/2;
#pragma unroll
        for (uint j = 0; j < RV/2; j++)
        {
            uint k1 = lr + j*RT;
            uint k1b = sel == 0 ? (RW - k1) & (RW - 1) : RW - 1 - k1;
            tail_pair(buf, boff + PADR(k1), boff + PADR(k1b), r*RW + k1, r*RW + k1b, r + H*k1, D, u2s[r*RW + k1], Bsp, square);
        }
        if (sel == 0 && lr == 0)
            tail_pair(buf, PADR(RW/2), PADR(RW/2), RW/2, RW/2, H*(RW/2), D, u2s[RW/2], Bsp, square);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint i = 0; i < RV; i++)
        xs[i] = BUF(lr + i*RT);
    barrier(CLK_LOCAL_MEM_FENCE);
    STAGES_I
}
#endif   // TAILK
#endif

#ifdef REALMUL2
// Real-FFT post-processing, pointwise product and pre-processing for the inverse, in the transposed
// layout: position s = k2*W + k1 holds frequency f = k2 + H*k1; its partner D - f sits in row
// H - k2 (row 0 and row H/2 pair with themselves), reversed.
// With Zk = Z[f], Zm = Z[D - f], t = exp(-i*pi*f/D) the spectrum of the real sequence is
// Xk = E + t*O, Xm = conj(E - t*O), E = (Zk + conj(Zm))/2, O = (Zk - conj(Zm))/2i, and packing the
// product back collapses to
//     Zk' = Zk*Bk + u*d*e,  Zm' = Zm*Bm + conj(u*d*e),  d = Zk - conj(Zm), e = Bk - conj(Bm),
// with u = -(1 + t^2)/4 (table u2s, indexed by the position of the lower frequency).
inline C csqr(C a) { return (C)(fma(a.x, a.x, -a.y*a.y), (a.x + a.x)*a.y); }
__kernel void realmul2(__global C* A, __global const C* B, __global const C* u2s, uint W, uint lgW, uint H, uint D, int square, uint total)
{
    uint s = get_global_id(0);
    if (s >= total)
        return;
    uint k2 = s >> lgW, k1 = s & (W - 1);
    uint p2, p1;
    if (k2 == 0)
    {
        if (2*k1 > W) return;
        p2 = 0; p1 = (W - k1) & (W - 1);
    }
    else if (2*k2 == H)
    {
        if (2*k1 >= W) return;
        p2 = k2; p1 = W - 1 - k1;
    }
    else
    {
        p2 = H - k2; p1 = W - 1 - k1;
    }
    uint f = k2 + H*k1;
    uint sp = p2*W + p1;
    // the element with the lower frequency plays "k", the other "D - k"
    bool low = 2*f <= D;
    uint ik = low ? s : sp, im = low ? sp : s;
    C u = u2s[s];
    C Zk = A[ik], Zm = A[im];
#ifdef PSCS
    // FP32 hybrid with D = 3*2^k: the product takes the factor 2^k/D, so the carry's scale is a power of two
    { double f = square ? PSCS : PSCM; Zk *= f; Zm *= f; }
#endif
    C d = Zk - cconj(Zm);
    C g, Rk, Rm;
    if (square)
    {
        g = cmul(u, csqr(d));
        Rk = csqr(Zk);
        Rm = csqr(Zm);
    }
    else
    {
        C Bk = B[ik], Bm = B[im];
        g = cmul(u, cmul(d, Bk - cconj(Bm)));
        Rk = cmul(Zk, Bk);
        Rm = cmul(Zm, Bm);
    }
    A[ik] = Rk + g;
    if (ik != im)
        A[im] = Rm + cconj(g);
}
#endif
)CLC";

static void check(cl_int err, const char* what)
{
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err) + " in " + what);
}

struct Engine::Impl
{
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel k_r3 = nullptr;
    cl_kernel k_r4 = nullptr, k_r2 = nullptr, k_realmul = nullptr, k_carry_d = nullptr, k_carry_l = nullptr, k_fix = nullptr, k_fold = nullptr;
    cl_kernel k_lowpart = nullptr, k_addl = nullptr, k_shiftdown = nullptr, k_r16 = nullptr, k_seq = nullptr;
    cl_kernel k_fix3 = nullptr;               // both carry_fix passes + carry_seq in one launch
    cl_kernel k_carry_h = nullptr, k_lowpart_i = nullptr, k_shiftdown_i = nullptr;   // hybrid Montgomery
    // Montgomery with cyclic and negacyclic halves (see carry_q): fN holds the negacyclic spectrum of N,
    // fQ that of x, fW the weighted input; slots have a second pair of halves (negacyclic spectrum of
    // transform()); mspec: both spectra of mul()'s operand; q*: zeta^j tables (FP32, field)
    bool mmgw = false;
    cl_kernel k_qweight = nullptr, k_qmul = nullptr, k_qcarry = nullptr;
    cl_mem fQ = nullptr, fW = nullptr, mspec = nullptr, qflo = nullptr, qfhi = nullptr, qnlo = nullptr, qnhi = nullptr;
    std::map<cl_mem, std::pair<cl_mem, cl_mem>> halves2;
    cl_kernel kcol2f = nullptr;               // col_pass2_h forward from a complex buffer (MMGW, IBDWT)
    // IBDWT for k*2^n - 1 (see ib_carry): weight tables (FP32: float2, field: uint2), f_j tables
    bool ibdwt = false;
    cl_kernel k_ibw = nullptr, k_ibc = nullptr, k_ibc2 = nullptr;
    cl_kernel kcib = nullptr, kcol2ib = nullptr;   // first forward column pass(es) from a slot, weighted (IBW)
    cl_mem ibt[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    cl_mem ibfl = nullptr, ibfh = nullptr;
    cl_mem ticket = nullptr;                  // its work-group counter (0 between launches)
    // fused carry + folds (carry_fold), see setup_fused()
    cl_kernel k_cfold = nullptr, k_cfoldi = nullptr;
    cl_mem etop = nullptr, car3 = nullptr;
    bool fused = false;
    uint32_t fU = 0, fA1 = 0, foff = 0;       // e digits end at fU, z digits (slot) at fA1; group grid offset
    cl_mem fC = nullptr, fN = nullptr, fNinv = nullptr;  // Montgomery: scratch, FFT(N), FFT(-N^-1 mod R)
    cl_mem tw = nullptr, tw2 = nullptr;
    cl_mem twlo = nullptr, twhi = nullptr;   // W^m = twlo[m & (2^lbits-1)] * twhi[m >> lbits]
    uint32_t lbits = 0;
    cl_mem fA = nullptr, fB = nullptr, fT = nullptr;   // complex work buffers (D)
    cl_mem P = nullptr, Q = nullptr;                    // long work buffers (2D)
    cl_mem car1 = nullptr, car2 = nullptr;
    cl_mem maxerr = nullptr, errflag = nullptr;
    std::vector<cl_mem> slots;
    std::string device_name;
    uint32_t CH = 16;                         // digits per carry chunk (CHUNK), see carry_chunk()
    cl_uint wk = 0;          // digit bits if the digit base is 2^wk, else 0
    cl_long B = 0;           // digit base
    cl_double invB = 0;
    uint32_t ops_since_sync = 0;
    uint32_t batch = 32;                      // operations per queued batch, grows if the GPU runs dry
    std::deque<cl_event> pending;

    ~Impl()
    {
        i32.reset();                          // inner engines first: they use this context
        i31.reset();
        for (auto ev : pending) clReleaseEvent(ev);
        if (sevent) { clWaitForEvents(1, &sevent); clReleaseEvent(sevent); }
        for (auto k : lds_kernels) clReleaseKernel(k);
        for (auto p : lds_progs) clReleaseProgram(p);
        for (auto m : slots) clReleaseMemObject(m);
        for (auto& cp : col_plan) if (cp.F) clReleaseMemObject(cp.F);
        for (auto& h : halves2) { clReleaseMemObject(h.second.first); clReleaseMemObject(h.second.second); }
        for (auto m : ibt) if (m) clReleaseMemObject(m);
        cl_mem mems[] = { ibfl, ibfh, fQ, fW, mspec, qflo, qfhi, qnlo, qnhi, ticket, rowtr, n3pw, ntw, nrowtw, nu2s, etop, car3, rowtw, tw2s, tw, tw2, twlo, twhi, fA, fB, fT, P, Q, car1, car2, maxerr, errflag, fC, fN, fNinv };
        for (auto m : mems) if (m) clReleaseMemObject(m);
        cl_kernel ks[] = { k_cfoldi, k_cfold, k_r4, k_r2, k_realmul, k_carry_d, k_carry_l, k_fix, k_fold, k_lowpart, k_addl, k_shiftdown, k_r16, k_seq, k_fix3, k_carry_h, k_lowpart_i, k_shiftdown_i, k_qweight, k_qmul, k_qcarry, k_ibw, k_ibc, k_ibc2 };
        for (auto k : ks) if (k) clReleaseKernel(k);
        if (program) clReleaseProgram(program);
        for (auto& h : halves) { clReleaseMemObject(h.second.first); clReleaseMemObject(h.second.second); }
        if (owner) return;                   // borrowed context and queue
        if (queue) clReleaseCommandQueue(queue);
        if (ctx) clReleaseContext(ctx);
    }

    std::map<cl_kernel, size_t> max_local;   // CL_KERNEL_WORK_GROUP_SIZE per kernel

    struct LdsPass { cl_kernel k; uint32_t R, Ns, G, T; };
    std::vector<LdsPass> lds_plan;            // FFT passes for this engine's length
    LdsPass lds_fused = { nullptr, 0, 0, 0, 0 };   // last forward pass with the pointwise product
    bool radix3 = false;                      // D = 3*2^k: a final radix-3 pass (fft_r3)
    // FFT v2 (default; PRST_GPU_FFT=1 selects the original Stockham passes): D = W*H
    bool v2 = false;
    // Hybrid engine (PRST_GPU_H32=1): an FP32 FFT and an NTT over GF(M31^2) of the same length; the
    // coefficients are recovered exactly from the FP32 approximation and the residue mod 2^31 - 1.
    // The two transforms are inner Impls sharing context and queue; spectra live in the two halves
    // of each complex buffer (sub-buffers, FP32 first).
    bool h32 = false, fp32 = false, f31 = false, f3 = false;
    bool islot = false;
    // device has FP64; without it only the hybrid engine with the integer carry and 32-bit slots runs
    bool fp64 = true;
    // asynchronous download (snapshot): host copies of two slots and of the error flags
    std::vector<char> sraw[2];
    cl_uint smaxerr = 0, serrflag = 0;
    cl_event sevent = nullptr;
    bool twc = false;                         // row twiddles computed in the kernel (FP32)                       // slots hold 32-bit integer digits (hybrid): half the slot traffic
    std::string nconsts;                      // NTT build constants (fused hybrid kernels)
    struct HPass { cl_kernel kf, ki, kc, kcz; uint32_t R, Ns; };   // fused FP32 + NTT column pass
    std::vector<HPass> hcol;
    // both column passes in one launch (col_pass2_h): forward from a slot, ditto half-zero, inverse
    cl_kernel kcol2c = nullptr, kcol2z = nullptr, kcol2i = nullptr;
    uint32_t col2_cw = 0;
    uint32_t n3pinv = 0, n3C = 0;             // P3 field: p^-1 mod 2^32, R^2/D (carry)
    cl_mem n3pw = nullptr;                    // P3 field: 2^E*R mod p, E < 64 (carry)
    Impl* owner = nullptr;                    // inner Impl: context/queue borrowed, kernels enqueued through owner
    std::unique_ptr<Impl> i32, i31;
    std::map<cl_mem, std::pair<cl_mem, cl_mem>> halves;
    // NTT engine (PRST_GPU_NTT=1): GF(p^2) transform, see ntt_source
    bool ntt = false;
    struct NttCol { cl_kernel kf, ki, kc, kcz; uint32_t R, Ns; };   // forward, inverse, forward from a slot, ditto half-zero
    std::vector<NttCol> ntt_col;
    cl_kernel k_nrow = nullptr, k_nrowi = nullptr, k_nmul = nullptr, k_ntail = nullptr;
    uint32_t ntail_threads = 0;
    uint32_t nrow_threads = 0, nrow_groups = 0;
    cl_mem ntw = nullptr, nrowtw = nullptr, nu2s = nullptr;
    uint64_t ndinv = 0;
    uint32_t W = 0, H = 0, lgW = 0;
    struct ColPass { cl_kernel kf, ki, kz; uint32_t R, Ns; cl_mem F; };   // forward, inverse, forward with half-zero input; F: fancy twiddles
    std::vector<ColPass> col_plan;
    std::vector<std::pair<uint32_t, uint32_t>> row_stages;   // (Ns, Rs) of the row FFT
    bool halfzero = false;                    // values occupy only the low D reals of a slot
    cl_kernel k_row = nullptr, k_rowi = nullptr, k_realmul2 = nullptr;
    cl_kernel k_tail = nullptr;               // fused row FFT + product + inverse row FFT (row_tail)
    uint32_t tail_threads = 0;
    std::string tail_opts, tail_prefix;       // build options / stage macros of row_tail (or the NTT's)
    cl_kernel k_htail = nullptr;              // hybrid: both row tails in one launch (row_tail_h)
    uint32_t row_threads = 0, row_groups = 0;
    cl_mem tw2s = nullptr, rowtw = nullptr, rowtr = nullptr;   // rowtr: per-row factors (TWC)
    std::vector<cl_program> lds_progs;
    std::vector<cl_kernel> lds_kernels;

    // Optional profiling (environment variable PRST_GPU_PROFILE=1): GPU time per kernel over the
    // first PROFILE_OPS operations, printed once to stderr. Waits after every kernel, so only then slower.
    static const uint32_t PROFILE_OPS = 200;
    bool profile = false;
    uint32_t prof_ops = 0;
    std::map<std::string, std::pair<double, uint32_t>> prof;

    void enqueue(cl_kernel k, size_t global, const size_t* local)
    {
        if (owner)
        {
            owner->enqueue(k, global, local);
            return;
        }
        if (!profile || prof_ops >= PROFILE_OPS)
        {
            check(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
            return;
        }
        cl_event ev;
        check(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, local, 0, nullptr, &ev), "clEnqueueNDRangeKernel");
        clWaitForEvents(1, &ev);
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        char name[64] = { 0 };
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name) - 1, name, nullptr);
        std::string key = std::string(name) + (local ? " wg=" + std::to_string(*local) : "") + " n=" + std::to_string(global);
        auto& p = prof[key];
        p.first += (t1 - t0)*1e-6;
        p.second++;
    }

    void op_done()
    {
        if (!profile || ++prof_ops != PROFILE_OPS)
            return;
        double total = 0;
        for (auto& p : prof)
            total += p.second.first;
        fprintf(stderr, "GPU profile, %u operations, %.3f ms/op kernel time:\n", PROFILE_OPS, total/PROFILE_OPS);
        for (auto& p : prof)
            fprintf(stderr, "  %-40s %7.3f ms/op %5.1f%%  %5.1f launches/op\n", p.first.c_str(), p.second.first/PROFILE_OPS, 100*p.second.first/total, (double)p.second.second/PROFILE_OPS);
        prof.clear();
    }

    void run_local(cl_kernel k, size_t global, size_t local)
    {
        enqueue(k, global, &local);
    }

    // Work-group size 64 (or the kernel's limit, if lower) when the global size allows it.
    void run(cl_kernel k, size_t global)
    {
        size_t local = 64;
        auto it = max_local.find(k);
        if (it != max_local.end())
            while (local > it->second && local > 1)
                local >>= 1;
        enqueue(k, global, global % local == 0 ? &local : nullptr);
    }
};

static void run_fft(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half = false);
static void run_col_passes(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half);
static void h32_cols(Engine::Impl& I, cl_mem src32, cl_mem src31, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half);
static void run_row(Engine::Impl& I, cl_mem in, cl_mem out, int inv);
static void ntt_row(Engine::Impl& I, cl_mem in, cl_mem out, int inv);
static void build_hybrid_fused(Engine::Impl& I);
static void build_hybrid_tail(Engine::Impl& I);
static void wait_event(cl_event ev);
// Hybrid inner transforms have 8-byte elements: rows up to twice as long fit the same local memory.
// Default rows stay at the 16-byte limit (measured faster); PRST_GPU_INNERP=2 allows twice as long.
static uint32_t inner_p(uint32_t P) { const char* e = getenv("PRST_GPU_INNERP"); return e != nullptr && atoi(e) == 2 ? 2*P : P; }
// Row length limit of the hybrid's transforms. Rows of 2048 points (devices with 48 KB of local
// memory) only when they save a column pass: otherwise 1024 measured faster on an NVIDIA T600
// (8-17%: a two-row tail of 2048 points leaves room for one work-group per SM), with 2048 up to 15%
// faster when the column passes drop from 3 to 2. Same pass count rule as the plans (radix <= 16,
// factor 3 in the last pass).
static uint32_t hybrid_rows(uint32_t P, uint32_t D)
{
    if (getenv("PRST_GPU_INNERP") != nullptr || P < 2048)
        return inner_p(P);
    auto passes = [&](uint32_t W)
    {
        uint32_t H = D/W, r3 = H % 3 == 0 ? 3 : 1, hb = 0;
        while ((1u << hb) < H/r3) hb++;
        return (hb + (r3 == 3 ? 2 : 0) + 3)/4;
    };
    uint32_t P2 = (D & (D - 1)) == 0 ? D : D/3;
    if (P2 < 4096)
        return P;
    return passes(2048) < passes(1024) ? 2048 : 1024;
}
// Slot contents as doubles on the host; on the device doubles, or 32-bit integers (islot).
static void write_slot(Engine::Impl& I, cl_mem m, const std::vector<double>& v)
{
    if (!I.islot)
    {
        check(clEnqueueWriteBuffer(I.queue, m, CL_TRUE, 0, sizeof(double)*v.size(), v.data(), 0, nullptr, nullptr), "write_slot");
        return;
    }
    std::vector<cl_int> b(v.size());
    for (size_t i = 0; i < v.size(); i++)
        b[i] = (cl_int)v[i];
    check(clEnqueueWriteBuffer(I.queue, m, CL_TRUE, 0, sizeof(cl_int)*b.size(), b.data(), 0, nullptr, nullptr), "write_slot");
}
static void read_slot(Engine::Impl& I, cl_mem m, std::vector<double>& v)
{
    if (!I.islot)
    {
        check(clEnqueueReadBuffer(I.queue, m, CL_TRUE, 0, sizeof(double)*v.size(), v.data(), 0, nullptr, nullptr), "read_slot");
        return;
    }
    std::vector<cl_int> b(v.size());
    check(clEnqueueReadBuffer(I.queue, m, CL_TRUE, 0, sizeof(cl_int)*b.size(), b.data(), 0, nullptr, nullptr), "read_slot");
    for (size_t i = 0; i < v.size(); i++)
        v[i] = (double)b[i];
}
static uint32_t ilog2(uint32_t x) { uint32_t k = 0; while ((1u << k) < x) k++; return k; }
static bool is_pow2(uint32_t x) { return (x & (x - 1)) == 0; }
// FFT lengths 16, 24, 32, 48, 64, 96, ...: powers of two and 3*2^k (fewer wasted bits per digit).
// PRST_GPU_POW2=1 restricts to powers of two.
static uint32_t next_fft_len(uint32_t D)
{
    static int pow2_only = -1;
    if (pow2_only < 0)
    {
        const char* e = getenv("PRST_GPU_POW2");
        pow2_only = e != nullptr && e[0] == '1';
    }
    if (pow2_only)
        return is_pow2(D) ? D*2 : (D/3)*4;
    return is_pow2(D) ? D + D/2 : (D/3)*4;
}
static void build_lds_plan(Engine::Impl& I, uint32_t D);
// Local memory a kernel may declare statically. NVIDIA reports the per-SM size (64 KB or more on
// Turing and later), but a statically sized __local array is limited to 48 KB per work-group.
static cl_ulong local_mem(Engine::Impl& I)
{
    cl_ulong lmem = 0;
    clGetDeviceInfo(I.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    return lmem < 49152 ? lmem : 49152;
}
static void build_v2_plan(Engine::Impl& I, uint32_t D, uint32_t P);
static void make_v2_tables_f32(Engine::Impl& I, uint32_t D);
// Two sub-buffers over the halves of a complex buffer (hybrid engine): FP32 data first, M31 second.
static void add_halves(Engine::Impl& I, cl_mem m, size_t size)
{
    cl_int err;
    cl_buffer_region r0 = { 0, size/2 }, r1 = { size/2, size/2 };
    cl_mem a = clCreateSubBuffer(m, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r0, &err); check(err, "sub-buffer");
    cl_mem b = clCreateSubBuffer(m, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r1, &err); check(err, "sub-buffer");
    I.halves[m] = { a, b };
}
// MMGW: the second spectrum of a slot (offset size), as add_halves.
static void add_halves2(Engine::Impl& I, cl_mem m, size_t size)
{
    cl_int err;
    cl_buffer_region r0 = { size, size/2 }, r1 = { size + size/2, size/2 };
    cl_mem a = clCreateSubBuffer(m, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r0, &err); check(err, "sub-buffer");
    cl_mem b = clCreateSubBuffer(m, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r1, &err); check(err, "sub-buffer");
    I.halves2[m] = { a, b };
}
static void setup_mmgw(Engine::Impl& I, uint32_t D);
static bool ib_magnitude(const std::vector<double>& digits, uint64_t n, uint64_t L, std::vector<uint32_t>& magnitude);
static void ib_forward(Engine::Impl& I, cl_mem slot, const std::pair<cl_mem, cl_mem>& dst);
static void nega_forward(Engine::Impl& I, cl_mem slot, const std::pair<cl_mem, cl_mem>& dst);
static void build_ntt_plan(Engine::Impl& I, uint32_t D, uint32_t P);
static void ntt_fft(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, int inv, bool half);
static void ntt_product(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, cl_mem Bsp, int square, bool half);
// Devices without FP64 run only the hybrid with the integer carry (carry_fold_i) and 32-bit slots;
// the environment switches that select an FP64 variant are ignored there.
// PRST_GPU_NTT=1 selects the GF(p^2) NTT engine where it applies (fold engines). Its carry is FP64.
static bool ntt_requested(bool fp64) { const char* e = getenv("PRST_GPU_NTT"); return fp64 && e != nullptr && e[0] == '1'; }
// PRST_GPU_ICARRY=0: the hybrid's carry in FP64 (carry_fold) instead of carry_fold_i.
static bool icarry_requested(bool fp64) { const char* e = getenv("PRST_GPU_ICARRY"); return !fp64 || !(e != nullptr && e[0] == '0'); }
// The hybrid is the default for the fold engines (measured 6-33% faster than FP64 on RDNA3, also
// where it needs the next power-of-two length). PRST_GPU_H32=0 selects the FP64 FFT.
// Digits per work item in the carry kernels. The hybrid's integer carry (carry_fold_i) is limited by
// its serial carry chain, not by arithmetic: 4 digits per work item measured 16-31% faster per
// squaring than 16 (the FP64 path is FP64 bound and stays at 16). At least 4: the fused carry needs
// work-groups of 256 digits or more (q and the top digits of e in one group). PRST_GPU_CH overrides.
static uint32_t carry_chunk(bool h32, bool fp64)
{
    const char* e = getenv("PRST_GPU_CH");
    uint32_t ch = h32 && icarry_requested(fp64) ? 4 : 16;
    if (e != nullptr)
    {
        uint32_t v = (uint32_t)atoi(e);
        if (v >= 4 && v <= 64 && (v & (v - 1)) == 0)
            ch = v;
    }
    return ch;
}
static bool h32_requested(bool fp64)
{
    const char* e = getenv("PRST_GPU_H32");
    const char* f = getenv("PRST_GPU_FUSED");
    return !fp64 || (!(e != nullptr && e[0] == '0') && !(f != nullptr && f[0] == '0'));
}
// Group grid of the fused carry (groups of unit = 64*CH digits, group g starts at g*unit - off).
// q should sit 128 digits below the end of its group: the top digits q..U-1 (U - q <= 64) stay in
// that group and the e-level carries of the groups below are absorbed before q. Chunk 0 is short
// (unit - off digits) and takes the top digits in carry_fix, so off <= unit - 128. When q is already
// in the top 128 digits of a natural group, this leaves q either where it is (if the top digits
// still fit) or 128 digits lower. The slot (fA1) ends at the end of q's group.
static void fold_grid(uint32_t unit, uint32_t fq, uint32_t& off, uint32_t& fA1)
{
    uint32_t r = fq%unit;
    if (r <= unit - 128)
        off = unit - 128 - r;
    else if (r <= unit - 64)
        off = 0;
    else
        off = unit - 128;
    uint32_t pos = (r + off)%unit;
    fA1 = fq + unit - pos;
}
// The checks of setup_fused, without allocating anything: the hybrid needs the fused carry.
static bool fused_feasible(int64_t B, uint32_t CH, uint32_t D, uint32_t fq, int64_t fbs, int64_t fbmul, uint32_t k, int32_t c)
{
    double lb = std::log2((double)B);
    double cb = std::fabs((double)c)*(double)fbmul;
    uint32_t kd = (uint32_t)std::ceil(32/lb) + 3;
    uint32_t m = fq + 1 + kd;
    uint32_t U = std::max(fq + (uint32_t)std::ceil(53/lb) + 1, 2*m + 1 - fq) + (uint32_t)std::ceil(52/lb) + 2;
    uint32_t off, fA1;
    fold_grid(64*CH, fq, off, fA1);
    bool ok = ((double)k + cb)*(double)B < 0x1p51 && (double)k + cb < 512.0*(double)B;
    return ok && U - fq <= 64 && m <= fA1 && fA1 <= 2*D && (double)fbs <= 0x1p31;
}
// Hybrid digit size above the FP64 limit: the FP32 error only has to stay below 2^30. Measured at
// 2^21 with 18.5-bit digits: error ~0.05 of the limit. PRST_GPU_H32BITS overrides (tuning).
// Hybrid lengths 3*2^k (GF(P3^2) NTT) as well as 2^k (GF(M31^2)). PRST_GPU_P3=0: powers of two only.
static bool h32_mixed() { const char* e = getenv("PRST_GPU_P3"); return !(e != nullptr && e[0] == '0'); }
// Hybrid lengths: 3*2^k only from 3*2^17 on (below that the power of two measured faster).
static uint32_t h32_next_len(uint32_t D)
{
    uint32_t c = next_fft_len(D);
    static uint32_t cmin = getenv("PRST_GPU_P3MIN") ? (uint32_t)atoi(getenv("PRST_GPU_P3MIN")) : 393216;   // tuning / tests
    if (!is_pow2(c) && (c < cmin || !h32_mixed()))
        c = is_pow2(D) ? 2*D : (D/3)*4;
    return c;
}
static double h32_extra_bits() { const char* e = getenv("PRST_GPU_H32BITS"); return e != nullptr ? atof(e) : 0.5; }
// Base-b digits (b^e, not whole bits) get more room: with 0.5 a digit size just above the limit falls
// back a whole factor b (101^3 = 19.9 bits -> 101^2 = 13.3 bits, a transform twice as long). Measured
// with 1.5 on an RX 7900 XTX (max round-off over a whole run, every operation of the test): GCW
// 4436026*101^4436026+1 101^3 digits, 0.18-0.20 (0.889 -> 0.493 ms per squaring); 7*10^9000000+1
// 10^6 digits, 0.18; bases whose next digit does not fit are unchanged. A round-off above 0.4
// restarts with one bit less (check_errors), i.e. the old digits.
static double h32_extra_bits_b() { const char* e = getenv("PRST_GPU_H32BITS"); return e != nullptr ? atof(e) : 1.5; }
// Exact products: |coefficient| <= D*(B/2 + 1)^2 must stay below p/2 = 2^60 (with margin). B >= 2^12
// keeps the carries of carry_fold's first stage below 2^53.
static bool ntt_digits_ok(uint32_t D, double B) { return std::log2((double)D) + 2*std::log2(B/2 + 1) < 59.5 && B >= 4096.0; }
static void setup_fused(Engine::Impl& I, uint32_t D, uint32_t fq, int64_t fbs, int64_t fbmul, uint32_t k, int32_t c);
static void fold_lengths(Engine::Impl& I, uint32_t fq, uint32_t len, uint32_t& A0, uint32_t& A1);
static void build_v2_plan(Engine::Impl& I, uint32_t D, uint32_t P);
static cl_kernel build_variant_impl(Engine::Impl& I, std::map<std::string, cl_kernel>& cache, const std::string& opts);

static std::string device_string(cl_device_id dev, cl_device_info param)
{
    size_t len = 0;
    clGetDeviceInfo(dev, param, 0, nullptr, &len);
    std::string s(len, '\0');
    clGetDeviceInfo(dev, param, len, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static std::vector<std::pair<cl_platform_id, cl_device_id>> all_devices()
{
    std::vector<std::pair<cl_platform_id, cl_device_id>> res;
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
        return res;
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    for (auto p : plats)
    {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
            continue;
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (auto d : devs)
            res.emplace_back(p, d);
    }
    return res;
}

// PRST_GPU_FP64=0: treat every device as one without FP64 (tests the hybrid-only path)
static bool has_fp64(cl_device_id d)
{
    static const bool off = getenv("PRST_GPU_FP64") != nullptr && getenv("PRST_GPU_FP64")[0] == '0';
    if (off)
        return false;
    std::string ext = device_string(d, CL_DEVICE_EXTENSIONS);
    return ext.find("cl_khr_fp64") != std::string::npos || ext.find("cl_amd_fp64") != std::string::npos;
}

// Index into all_devices(): the given one, or (device < 0) the first GPU with FP64, else the first
// GPU (hybrid engine only), else the first device with FP64.
static int pick_device(const std::vector<std::pair<cl_platform_id, cl_device_id>>& devs, int device)
{
    if (devs.empty())
        throw std::runtime_error("No OpenCL devices found.");
    if (device >= 0)
    {
        if (device >= (int)devs.size())
            throw std::runtime_error("OpenCL device index out of range.");
        return device;
    }
    auto is_gpu = [](cl_device_id d)
    {
        cl_device_type type = 0;
        clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
        return (type & CL_DEVICE_TYPE_GPU) != 0;
    };
    for (size_t i = 0; i < devs.size(); i++)
        if (is_gpu(devs[i].second) && has_fp64(devs[i].second))
            return (int)i;
    for (size_t i = 0; i < devs.size(); i++)
        if (is_gpu(devs[i].second))
            return (int)i;
    for (size_t i = 0; i < devs.size(); i++)
        if (has_fp64(devs[i].second))
            return (int)i;
    throw std::runtime_error("No OpenCL GPU or device with double precision (cl_khr_fp64) found.");
}
static bool device_fp64(int device)
{
    auto devs = all_devices();
    return has_fp64(devs[pick_device(devs, device)].second);
}

std::vector<DeviceInfo> Engine::list_devices()
{
    std::vector<DeviceInfo> res;
    auto devs = all_devices();
    for (size_t i = 0; i < devs.size(); i++)
    {
        DeviceInfo info;
        info.index = (int)i;
        info.name = device_string(devs[i].second, CL_DEVICE_NAME);
        size_t len = 0;
        clGetPlatformInfo(devs[i].first, CL_PLATFORM_NAME, 0, nullptr, &len);
        info.platform.resize(len);
        clGetPlatformInfo(devs[i].first, CL_PLATFORM_NAME, len, &info.platform[0], nullptr);
        while (!info.platform.empty() && info.platform.back() == '\0') info.platform.pop_back();
        info.fp64 = has_fp64(devs[i].second);
        cl_device_type type = 0;
        clGetDeviceInfo(devs[i].second, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
        info.is_gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
        res.push_back(info);
    }
    return res;
}

int Engine::find_gpu(const std::string& vendor, int index)
{
    std::string key;
    for (char ch : vendor)
        key += (char)std::tolower((unsigned char)ch);
    if (key == "ati" || key == "amd")
        key = "amd";
    else if (key.find("nvidia") != std::string::npos)
        key = "nvidia";
    else if (key.find("intel") != std::string::npos)
        key = "intel";
    else if (key.find("apple") != std::string::npos)
        key = "apple";
    int count = 0;
    for (auto& d : list_devices())
    {
        if (!d.is_gpu)
            continue;
        std::string plat;
        for (char ch : d.platform)
            plat += (char)std::tolower((unsigned char)ch);
        bool match = key.empty() || plat.find(key) != std::string::npos ||
            (key == "amd" && plat.find("advanced micro devices") != std::string::npos);
        if (match && count++ == index)
            return d.index;
    }
    return -1;
}

// Largest safe digit size for a real FFT of length 2D with balanced digits.
// Product coefficients grow like 2^(2w-2)*sqrt(digits); measured: 2^18 reals at 20 bits -> 0.09, 21 bits -> 0.33; keep one bit below that in the 53-bit mantissa.
static double max_digit_bits_real(uint32_t D)
{
    double lg = std::log2(2.0*D);
    return (53.0 - 5.0 - 0.6*lg + 2.0)/2.0;
}
static uint32_t max_digit_bits(uint32_t D) { return (uint32_t)std::floor(max_digit_bits_real(D)); }
static const char* no_fp64_error = "GPU engine: this number needs the FP64 FFT, but the OpenCL device has no double precision support.";

Engine::Engine(uint32_t k, uint32_t n, int32_t c, int device, int bits_reduce) : _impl(new Impl()), _k(k), _n(n), _c(c)
{
    if (k == 0 || (k & 1) == 0)
        throw std::runtime_error("GPU engine requires odd k.");
    if (k >= (1u << 31) || std::abs((int64_t)c) >= (1 << 20) || c == 0)
        throw std::runtime_error("GPU engine requires k < 2^31 and 0 < |c| < 2^20.");
    // k*2^n - 1: the weighted transform (no zero padding, about half the length) when it applies
    if (c == -1 && ib_setup(k, n, device, bits_reduce))
        return;

    // Choose FFT size: smallest power of two D (>= 16) such that the value fits in D digits.
    double bits = n + std::log2((double)k) + 3.0;
    _impl->fp64 = device_fp64(device);
    if (ntt_requested(_impl->fp64))
        for (uint32_t D = 4096; D <= (1u << 27); D *= 2)
        {
            uint32_t w = (uint32_t)std::ceil(bits/D);
            if (w < 12) w = 12;
            // as large as the product bound and carry_fold's double arithmetic allow
            while (w < 30 && ntt_digits_ok(D, std::ldexp(1.0, (int)w + 1)) &&
                   ((double)k + std::abs((double)c)*std::ldexp(1.0, (int)w + 1))*std::ldexp(1.0, (int)w + 1) < 0x1p50)
                w++;
            if (w <= 30 && ntt_digits_ok(D, std::ldexp(1.0, (int)w)))
            {
                _D = D;
                _w = w;
                _impl->ntt = true;
                break;
            }
        }
    // hybrid first; if the fused carry does not apply to its digits, the FP64 FFT
    for (int attempt = 0; attempt < 2; attempt++)
    {
    _impl->h32 = !_impl->ntt && attempt == 0 && h32_requested(_impl->fp64);
    _impl->CH = carry_chunk(_impl->h32, _impl->fp64);
    for (_D = _impl->ntt ? _D : 16; !_impl->ntt; _D = _impl->h32 ? h32_next_len(_D) : next_fft_len(_D))
    {
        uint32_t wmax = _impl->h32 ? (uint32_t)std::floor(max_digit_bits_real(_D) + h32_extra_bits()) : max_digit_bits(_D);
        if ((int)wmax - bits_reduce < 4)
            throw std::runtime_error("GPU engine: number too large for double precision FFT.");
        uint32_t w = (uint32_t)std::ceil(bits/_D);
        if (w < 4) w = 4;
        if (w <= wmax - bits_reduce)
        {
            // hybrid: the length is a power of two anyway, so take the largest digits that fit
            if (_impl->h32)
                while (w + 1 <= wmax - bits_reduce && w + 1 <= 24 &&
                       ((double)k + std::abs((double)c)*std::ldexp(1.0, (int)w + 1))*std::ldexp(1.0, (int)w + 1) < 0x1p50)
                    w++;
            _w = w;
            break;
        }
        if (_D >= (1u << 28))
            throw std::runtime_error("GPU engine: number too large.");
    }
    _nd = (uint32_t)std::ceil(bits/_w);
    _b = 2;
    _e = _w;
    _fq = _n/_w;
    _fbs = (int64_t)1 << (_n%_w);
    _fbmul = (int64_t)1 << (_w - _n%_w);
    _impl->wk = _w;
    _impl->B = (int64_t)1 << _w;
    _impl->invB = 1.0/_impl->B;
    // the hybrid NTT needs at least two rows of up to 2048 points
    if (!_impl->h32 || (_D >= 4096 && fused_feasible(_impl->B, _impl->CH, _D, _fq, _fbs, _fbmul, _k, _c)))
        break;
    if (!_impl->fp64)
        throw std::runtime_error(no_fp64_error);
    }
    init_device(device, bits_reduce);
    setup_fused(*_impl, _D, _fq, _fbs, _fbmul, _k, _c);
    if ((_impl->ntt || _impl->h32) && !_impl->fused)
        throw std::runtime_error("GPU NTT engine: the fused carry does not apply to this number.");
    if (_impl->h32)
        _impl->i32->halfzero = _impl->i31->halfzero = _impl->halfzero;
}

Engine::Engine(BaseB, uint32_t k, uint32_t b, uint32_t n, int32_t c, int device, int bits_reduce) : _impl(new Impl()), _k(k), _n(n), _c(c)
{
    if (b < 3 || (b & (b - 1)) == 0)
        throw std::runtime_error("GPU base-b engine requires b >= 3 and not a power of two.");
    if (k == 0 || k >= (1u << 31) || std::abs((int64_t)c) >= (1 << 20) || c == 0)
        throw std::runtime_error("GPU engine requires k < 2^31 and 0 < |c| < 2^20.");
    _b = b;
    double lb = std::log2((double)b);
    // value size in base-b units: |z| <= ~1.5*k*b^n, plus slack
    double units = n + std::log2(1.5*k)/lb + 3.0;
    _impl->fp64 = device_fp64(device);
    if (ntt_requested(_impl->fp64))
        for (uint32_t D = 4096; D <= (1u << 27); D *= 2)
        {
            uint32_t e = (uint32_t)std::ceil(units/(D - 2));
            uint32_t emin = (uint32_t)std::ceil(12/lb);
            if (e < emin) e = emin;
            // as large as the product bound and carry_fold's double arithmetic allow
            for (;;)
            {
                double B1 = std::pow((double)b, (double)(e + 1));
                if (!(ntt_digits_ok(D, B1) && ((double)k + std::abs((double)c)*B1)*B1 < 0x1p50))
                    break;
                e++;
            }
            if (e*lb < 40 && ntt_digits_ok(D, std::pow((double)b, (double)e)))
            {
                _D = D;
                _e = e;
                _impl->ntt = true;
                break;
            }
        }
    // hybrid first; if the fused carry does not apply to its digits, the FP64 FFT
    for (int attempt = 0; attempt < 2; attempt++)
    {
    _impl->h32 = !_impl->ntt && attempt == 0 && h32_requested(_impl->fp64);
    _impl->CH = carry_chunk(_impl->h32, _impl->fp64);
    for (_D = _impl->ntt ? _D : 16; !_impl->ntt; _D = _impl->h32 ? h32_next_len(_D) : next_fft_len(_D))
    {
        // base-B digits are not whole bits (13^5 = 18.5 bits): compare against the unrounded limit
        double wmax = max_digit_bits_real(_D) + (_impl->h32 ? h32_extra_bits_b() : 0.0) - bits_reduce;
        int emax = (int)std::floor(wmax/lb);
        uint32_t e = (uint32_t)std::ceil(units/(_D - 2));
        if (e < 1) e = 1;
        if (emax >= 1 && (int)e <= emax)
        {
            if (_impl->h32)
                while ((int)e + 1 <= emax && ((double)k + std::abs((double)c)*std::pow((double)b, e + 1.0))*std::pow((double)b, e + 1.0) < 0x1p50)
                    e++;
            _e = e;
            break;
        }
        if (wmax < 4 || _D >= (1u << 28))
            throw std::runtime_error("GPU engine: number too large for double precision FFT.");
    }
    int64_t B = 1;
    for (uint32_t i = 0; i < _e; i++)
        B *= b;
    _w = (uint32_t)std::ceil(_e*lb);       // for display only
    _nd = (uint32_t)std::ceil(units/_e);
    _fq = n/_e;
    uint32_t s = n%_e;
    _fbs = 1;
    for (uint32_t i = 0; i < s; i++)
        _fbs *= b;
    _fbmul = B/_fbs;
    _impl->wk = 0;
    _impl->B = B;
    _impl->invB = 1.0/(double)B;
    // the hybrid NTT needs at least two rows of up to 2048 points
    if (!_impl->h32 || (_D >= 4096 && fused_feasible(_impl->B, _impl->CH, _D, _fq, _fbs, _fbmul, _k, _c)))
        break;
    if (!_impl->fp64)
        throw std::runtime_error(no_fp64_error);
    }
    init_device(device, bits_reduce);
    setup_fused(*_impl, _D, _fq, _fbs, _fbmul, _k, _c);
    if ((_impl->ntt || _impl->h32) && !_impl->fused)
        throw std::runtime_error("GPU NTT engine: the fused carry does not apply to this number.");
    if (_impl->h32)
        _impl->i32->halfzero = _impl->i31->halfzero = _impl->halfzero;
}

int64_t Engine::digit_base() const
{
    return _impl->B;
}

void Engine::set_digits(int slot, const std::vector<int64_t>& digits)
{
    if (digits.size() > _D)
        throw std::runtime_error("GPU engine: too many digits for FFT size.");
    std::vector<double> buf(2*(size_t)_D, 0.0);
    for (size_t i = 0; i < digits.size(); i++)
        buf[i] = (double)digits[i];
    write_slot(*_impl, _impl->slots[slot], buf);
}

void Engine::get_digits(int slot, std::vector<int64_t>& digits)
{
    std::vector<double> buf(2*(size_t)_D);
    read_slot(*_impl, _impl->slots[slot], buf);
    digits.resize(buf.size());
    for (size_t i = 0; i < buf.size(); i++)
        digits[i] = (int64_t)buf[i];
    while (!digits.empty() && digits.back() == 0)
        digits.pop_back();
}

Engine::Engine(const uint32_t* N, int size, int device, int bits_reduce, bool mmgw) : _impl(new Impl()), _k(0), _n(0), _c(0)
{
    _mont = true;
    if (!device_fp64(device))
        throw std::runtime_error(no_fp64_error);
    while (size > 0 && N[size - 1] == 0)
        size--;
    if (size == 0 || (N[0] & 1) == 0)
        throw std::runtime_error("GPU Montgomery engine requires an odd modulus.");
    uint32_t top = N[size - 1];
    double nbits = 32.0*(size - 1);
    while (top) { nbits += 1; top >>= 1; }
    // R = 2^(q*w) >= 2^14 * N leaves room for |z| < N and mulc < 2^13.
    double bits = nbits + 14.0;
    // Hybrid (FP32 + GF(p^2), carry_h) by default, as the fold engines: RX 7900 XTX, 1572864 reals,
    // 0.242 vs 0.406 ms per product. PRST_GPU_H32=0: the FP64 FFT. The hybrid needs D >= 4096 (two
    // rows of up to 2048 points); its digits are capped at 24 bits like the fold engine's.
    _impl->fp64 = true;
    _impl->h32 = h32_requested(true);
    _impl->CH = carry_chunk(_impl->h32, true);
    // Hybrid: by default R = 2^K - 1 and Q = 2^K + 1, K = 2D*w (MMGW, see carry_q): products of
    // length 2D without zero padding instead of 4D. PRST_GPU_MMGW=0 disables.
    {
        const char* e = getenv("PRST_GPU_MMGW");
        _mmgw = _impl->mmgw = mmgw && _impl->h32 && !(e != nullptr && e[0] == '0');
    }
    for (_D = 16; ; _D = _impl->h32 ? h32_next_len(_D) : next_fft_len(_D))
    {
        // MMGW: full-length cyclic products, and the negacyclic one a sum of two products
        double wreal = _impl->h32 ? max_digit_bits_real(_D) + h32_extra_bits() - (_mmgw ? 0.75 : 0.0) : max_digit_bits_real(_D);
        int wmax = std::min(24, (int)std::floor(wreal)) - bits_reduce;
        if (wmax < 4)
            throw std::runtime_error("GPU engine: number too large for double precision FFT.");
        // products of two q-digit numbers need 2q+2 digits <= 2D; MMGW: |z| < 2N (times mulc < 2^13) < Q/2
        uint32_t w = _mmgw ? (uint32_t)std::ceil((nbits + 16.0)/(2.0*_D)) : (uint32_t)std::ceil(bits/(_D - 2));
        if (w < 4) w = 4;
        if ((int)w <= wmax && (!_impl->h32 || _D >= 4096) && (!_mmgw || _D % (64*_impl->CH) == 0))
        {
            _w = w;
            break;
        }
        if (_D >= (1u << 28))
            throw std::runtime_error("GPU engine: number too large.");
    }
    _q = _mmgw ? 2*_D : (uint32_t)std::ceil(bits/_w);
    _nd = _q;
    _impl->wk = _w;
    _impl->B = (int64_t)1 << _w;
    _impl->invB = 1.0/_impl->B;
    init_device(device, bits_reduce);

    Impl& I = *_impl;
    cl_int err;
    size_t csz = sizeof(double)*2*(size_t)_D;
    I.fC = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fC");
    I.fN = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fN");
    I.fNinv = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fNinv");
    if (I.h32)
    {
        // spectra of N and N' as the hybrid's two halves (FP32, residues), like fA/fB/fT
        add_halves(I, I.fN, csz);
        add_halves(I, I.fNinv, csz);
    }
    std::vector<double> digits;
    to_digits(N, size, digits);
    write_slot(I, I.fC, digits);
    if (_mmgw)
    {
        // fN: the negacyclic spectrum of N (fNinv: the cyclic one of -N^-1 mod R, set_mont_inverse)
        setup_mmgw(I, _D);
        nega_forward(I, I.fC, I.halves.at(I.fN));
    }
    else
        run_fft(I, I.fC, I.fN, I.fT, _D, 0);
    finish();
}

void Engine::set_mont_inverse(const uint32_t* data, int size)
{
    Impl& I = *_impl;
    std::vector<double> digits;
    to_digits(data, size, digits);
    write_slot(I, I.fC, digits);
    run_fft(I, I.fC, I.fNinv, I.fT, _D, 0);
    finish();
}

void Engine::init_device(int device, int bits_reduce)
{
    (void)bits_reduce;   // digit size was already chosen by the caller
    Impl& I = *_impl;
    auto devs = all_devices();
    int pick = pick_device(devs, device);
    I.platform = devs[pick].first;
    I.device = devs[pick].second;
    I.fp64 = has_fp64(I.device);
    // the constructors checked this already (hybrid engine with the integer carry only)
    if (!I.fp64 && (!I.h32 || _mont))
        throw std::runtime_error(no_fp64_error);
    I.device_name = device_string(I.device, CL_DEVICE_NAME);
    {
        cl_ulong max_alloc = 0, global_mem = 0;
        clGetDeviceInfo(I.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
        clGetDeviceInfo(I.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
        cl_ulong buf = sizeof(double)*2*(cl_ulong)_D;
        cl_ulong total = buf*16;   // ~6 complex + 2 integer work buffers, tables, 5 slots
        if ((max_alloc != 0 && buf > max_alloc) || (global_mem != 0 && total > global_mem))
            throw std::runtime_error("GPU " + I.device_name + " has too little memory: needs " + std::to_string(total >> 20) + " MB in buffers of " + std::to_string(buf >> 20) + " MB.");
    }

    cl_int err;
    I.ctx = clCreateContext(nullptr, 1, &I.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    const char* prof_env = getenv("PRST_GPU_PROFILE");
    I.profile = prof_env != nullptr && prof_env[0] != '\0' && prof_env[0] != '0';
    I.queue = clCreateCommandQueue(I.ctx, I.device, I.profile ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    check(err, "clCreateCommandQueue");
    I.program = clCreateProgramWithSource(I.ctx, 1, &kernel_source, nullptr, &err);
    check(err, "clCreateProgramWithSource");
    std::string mopts = "-DCHUNK=" + std::to_string(I.CH) + (I.fp64 ? "" : " -DNO_FP64");
    err = clBuildProgram(I.program, 1, &I.device, mopts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len = 0;
        clGetProgramBuildInfo(I.program, I.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(I.program, I.device, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }
    if (I.fp64)
    {
        // FP64 FFT, its carries, Montgomery (not compiled without FP64)
        I.k_r4 = clCreateKernel(I.program, "fft_r4", &err); check(err, "fft_r4");
        I.k_r2 = clCreateKernel(I.program, "fft_r2", &err); check(err, "fft_r2");
        I.k_r16 = clCreateKernel(I.program, "fft_r16", &err); check(err, "fft_r16");
        I.k_realmul = clCreateKernel(I.program, "realmul", &err); check(err, "realmul");
        I.k_r3 = clCreateKernel(I.program, "fft_r3", &err); check(err, "fft_r3");
        I.k_carry_d = clCreateKernel(I.program, "carry_d", &err); check(err, "carry_d");
        I.k_fold = clCreateKernel(I.program, "fold_carry", &err); check(err, "fold_carry");
        I.k_lowpart = clCreateKernel(I.program, "lowpart", &err); check(err, "lowpart");
        I.k_shiftdown = clCreateKernel(I.program, "shiftdown", &err); check(err, "shiftdown");
        I.k_cfold = clCreateKernel(I.program, "carry_fold", &err); check(err, "carry_fold");
        I.k_carry_h = clCreateKernel(I.program, "carry_h", &err); check(err, "carry_h");
        I.k_lowpart_i = clCreateKernel(I.program, "lowpart_i", &err); check(err, "lowpart_i");
        I.k_shiftdown_i = clCreateKernel(I.program, "shiftdown_i", &err); check(err, "shiftdown_i");
        if (I.ibdwt)
        {
            I.k_ibw = clCreateKernel(I.program, "ib_weight", &err); check(err, "ib_weight");
            I.k_ibc = clCreateKernel(I.program, "ib_carry", &err); check(err, "ib_carry");
            I.k_ibc2 = clCreateKernel(I.program, "ib_carry2", &err); check(err, "ib_carry2");
        }
        if (I.mmgw)
        {
            I.k_qweight = clCreateKernel(I.program, "nega_weight", &err); check(err, "nega_weight");
            I.k_qmul = clCreateKernel(I.program, "nega_mul", &err); check(err, "nega_mul");
            I.k_qcarry = clCreateKernel(I.program, "carry_q", &err); check(err, "carry_q");
        }
    }
    I.k_carry_l = clCreateKernel(I.program, "carry_l", &err); check(err, "carry_l");
    I.k_fix = clCreateKernel(I.program, "carry_fix", &err); check(err, "carry_fix");
    I.k_addl = clCreateKernel(I.program, "carry_add", &err); check(err, "carry_add");
    I.k_seq = clCreateKernel(I.program, "carry_seq", &err); check(err, "carry_seq");
    // PRST_GPU_FIX3=0: the three separate carry launches instead of carry_fix3
    {
        const char* e = getenv("PRST_GPU_FIX3");
        if (!(e != nullptr && e[0] == '0'))
        {
            I.k_fix3 = clCreateKernel(I.program, "carry_fix3", &err); check(err, "carry_fix3");
            cl_uint zero = 0;
            I.ticket = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &zero, &err); check(err, "ticket");
        }
    }
    if (I.wk != 0)
    {
        // binary digits: carry_fold_i with shifts (WB) from a separate build
        cl_program pw = clCreateProgramWithSource(I.ctx, 1, &kernel_source, nullptr, &err);
        check(err, "clCreateProgramWithSource(carry_fold_i)");
        std::string o = "-DWB=" + std::to_string(I.wk) + " -DCHUNK=" + std::to_string(I.CH) + (I.fp64 ? "" : " -DNO_FP64");
        check(clBuildProgram(pw, 1, &I.device, o.c_str(), nullptr, nullptr), "clBuildProgram(carry_fold_i)");
        I.k_cfoldi = clCreateKernel(pw, "carry_fold_i", &err); check(err, "carry_fold_i");
        I.lds_progs.push_back(pw);
    }
    else
    {
        I.k_cfoldi = clCreateKernel(I.program, "carry_fold_i", &err); check(err, "carry_fold_i");
    }
    for (cl_kernel k : { I.k_r4, I.k_r2, I.k_r16, I.k_realmul, I.k_carry_d, I.k_carry_l, I.k_fix, I.k_fold, I.k_lowpart, I.k_addl, I.k_shiftdown, I.k_fix3 })
    {
        size_t wg = 0;
        if (k != nullptr && clGetKernelWorkGroupInfo(k, I.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr) == CL_SUCCESS && wg > 0)
            I.max_local[k] = wg;
    }
    {
        const char* e = getenv("PRST_GPU_FFT");
        if (I.h32)
        {
            const char* se = getenv("PRST_GPU_ISLOT");
            I.islot = !I.fp64 || !(se != nullptr && se[0] == '0');   // double slots need FP64
            cl_ulong lmem = 0;
            lmem = local_mem(I);
            uint32_t P = 2048;
            while (P > 64 && ((cl_ulong)P + P/16 + 1)*16 + 1024 > lmem)
                P >>= 1;
            for (int f = 0; f < 2; f++)
            {
                std::unique_ptr<Impl> in(new Impl());
                in->owner = &I;
                in->islot = I.islot;
                in->platform = I.platform; in->device = I.device; in->ctx = I.ctx; in->queue = I.queue;
                in->CH = I.CH;
                if (f == 0)
                {
                    in->fp32 = true;
                    build_v2_plan(*in, _D, hybrid_rows(P, _D));
                    make_v2_tables_f32(*in, _D);
                    I.i32 = std::move(in);
                }
                else
                {
                    // powers of two: M31 (cheaper products); 3*2^k: P3
                    in->f31 = is_pow2(_D);
                    in->f3 = !in->f31;
                    build_ntt_plan(*in, _D, hybrid_rows(P, _D));
                    I.i31 = std::move(in);
                }
            }
            I.W = I.i32->W; I.H = I.i32->H; I.lgW = I.i32->lgW;
            try { build_hybrid_fused(I); }                // optional: separate column passes otherwise
            catch (const std::exception&) { I.hcol.clear(); }
            if (!I.hcol.empty())
            {
                try { build_hybrid_tail(I); }             // optional: two tail launches otherwise
                catch (const std::exception& ex) { I.k_htail = nullptr; if (getenv("PRST_GPU_HTAIL")) fprintf(stderr, "%s\n", ex.what()); }
            }
        }
        else if (I.ntt)
        {
            cl_ulong lmem = 0;
            lmem = local_mem(I);
            uint32_t P = 2048;
            while (P > 64 && ((cl_ulong)P + P/16 + 1)*16 + 1024 > lmem)
                P >>= 1;
            build_ntt_plan(I, _D, P);
        }
        else if (e != nullptr && e[0] == '1')
            build_lds_plan(I, _D);
        else
        {
            cl_ulong lmem = 0;
            lmem = local_mem(I);
            uint32_t P = 2048;
            while (P > 64 && ((cl_ulong)P + P/16 + 1)*16 + 1024 > lmem)
                P >>= 1;
            build_v2_plan(I, _D, P);
        }
    }

    // Twiddle tables, computed in long double on the host.
    const long double PI = 3.141592653589793238462643383279502884L;
    std::vector<double> tw(2*(size_t)_D), tw2(2*((size_t)_D/2 + 1));
    for (uint32_t m = 0; m < _D; m++)
    {
        long double a = 2*PI*m/_D;
        tw[2*m] = (double)cosl(a);
        tw[2*m + 1] = (double)-sinl(a);
    }
    for (uint32_t m = 0; m <= _D/2; m++)
    {
        long double a = PI*m/_D;
        tw2[2*m] = (double)cosl(a);
        tw2[2*m + 1] = (double)-sinl(a);
    }
    size_t csz = sizeof(double)*2*(size_t)_D;
    size_t lsz = sizeof(int64_t)*2*(size_t)_D;
    I.tw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, csz, tw.data(), &err); check(err, "tw");
    {
        uint32_t lg = 0;
        while ((1u << lg) < _D) lg++;
        I.lbits = (lg + 1)/2;
        uint32_t nlo = 1u << I.lbits, nhi = (_D >> I.lbits) + 1;
        std::vector<double> lo(2*(size_t)nlo), hi(2*(size_t)nhi);
        for (uint32_t m = 0; m < nlo; m++)
        {
            long double a = 2*PI*m/_D;
            lo[2*m] = (double)cosl(a);
            lo[2*m + 1] = (double)-sinl(a);
        }
        for (uint32_t m = 0; m < nhi; m++)
        {
            long double a = 2*PI*((long double)m*nlo)/_D;
            hi[2*m] = (double)cosl(a);
            hi[2*m + 1] = (double)-sinl(a);
        }
        I.twlo = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*lo.size(), lo.data(), &err); check(err, "twlo");
        I.twhi = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*hi.size(), hi.data(), &err); check(err, "twhi");
    }
    I.tw2 = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*tw2.size(), tw2.data(), &err); check(err, "tw2");
    if (I.v2)
    {
        // tw2 in the transposed layout, for positions s = k2*W + k1 with k2 <= H/2
        size_t n = (size_t)(I.H/2 + 1)*I.W;
        std::vector<double> t2(2*n);
        for (size_t s = 0; s < n; s++)
        {
            uint32_t k2 = (uint32_t)(s >> I.lgW), k1 = (uint32_t)(s & (I.W - 1));
            uint64_t f = k2 + (uint64_t)I.H*k1;
            if (2*f > _D) f = _D - f;
            // u = -(1 + t^2)/4, t = exp(-i*pi*f/D)
            long double a = 2*PI*f/_D;
            t2[2*s] = (double)(-(1 + cosl(a))/4);
            t2[2*s + 1] = (double)(sinl(a)/4);
        }
        I.tw2s = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*t2.size(), t2.data(), &err); check(err, "tw2s");
        // row FFT twiddles: E(r*(jm + k2/H)/(Ns*Rs)) = W_D^(r*(jm*H + k2)*W/(Ns*Rs))
        std::vector<double> rt(2*(size_t)_D, 0.0);
        for (uint32_t k2 = 0; k2 < I.H; k2++)
        {
            double* row = rt.data() + 2*(size_t)k2*I.W;
            for (auto& st : I.row_stages)
            {
                uint32_t Ns = st.first, Rs = st.second;
                uint64_t q = I.W/(Ns*Rs);
                for (uint32_t jm = 0; jm < Ns; jm++)
                    for (uint32_t r = 1; r < Rs; r++)
                    {
                        uint64_t m = (r*((uint64_t)jm*I.H + k2)*q) % _D;
                        size_t idx = Ns - 1 + jm*(Rs - 1) + r - 1;
                        row[2*idx] = tw[2*m];
                        row[2*idx + 1] = tw[2*m + 1];
                    }
            }
        }
        I.rowtw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*rt.size(), rt.data(), &err); check(err, "rowtw");
    }
    I.fA = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fA");
    I.fB = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fB");
    I.fT = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fT");
    if (I.h32)
        for (cl_mem m : { I.fA, I.fB, I.fT })
            add_halves(I, m, csz);
    I.P = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, lsz, nullptr, &err); check(err, "P");
    I.Q = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, lsz, nullptr, &err); check(err, "Q");
    uint32_t nch = 2*_D/I.CH;
    I.car1 = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, sizeof(int64_t)*nch, nullptr, &err); check(err, "car1");
    I.car2 = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, sizeof(int64_t)*nch, nullptr, &err); check(err, "car2");
    I.car3 = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, 3*sizeof(int64_t)*nch, nullptr, &err); check(err, "car3");
    cl_uint zero = 0;
    I.maxerr = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &zero, &err); check(err, "maxerr");
    I.errflag = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &zero, &err); check(err, "errflag");
}

Engine::~Engine()
{
    if (_impl && _impl->queue)
        clFinish(_impl->queue);
}

std::string Engine::description() const
{
    std::string mont = _mmgw ? ", Montgomery (cyclic/negacyclic)" : ", Montgomery";
    char ibd[64];
    snprintf(ibd, sizeof(ibd), "%.2f bits/digit, IBDWT", _n/(2.0*_D));
    std::string digits = _ibdwt ? std::string(ibd) : _impl->wk ? std::to_string(_w) + " bits/digit" : "base " + std::to_string(_b) + "^" + std::to_string(_e) + " digits (" + std::to_string(_e*std::log2((double)_b)).substr(0, 4) + " bits)";
    std::string layout = _impl->v2 || _impl->ntt || _impl->h32 ? " (" + std::to_string(_impl->W) + "x" + std::to_string(_impl->H) + ")" : "";
    return _impl->device_name + (_impl->ntt ? ", NTT GF(M61^2) " : _impl->h32 ? (_impl->i31->f3 ? ", FP32+GF(P3^2) " : ", FP32+GF(M31^2) ") : ", FFT ") + std::to_string(2*_D) + " reals" + layout + ", " + digits + (_mont ? mont : "");
}

int Engine::alloc_slot()
{
    cl_int err;
    size_t csz = sizeof(double)*2*(size_t)_D;
    // MMGW: room for a second (negacyclic) spectrum after the first
    size_t asz = _impl->mmgw ? 2*csz : csz;
    cl_mem m = clCreateBuffer(_impl->ctx, CL_MEM_READ_WRITE, asz, nullptr, &err);
    check(err, "alloc_slot");
    double zero = 0.0;
    check(clEnqueueFillBuffer(_impl->queue, m, &zero, sizeof(double), 0, asz, 0, nullptr, nullptr), "fill");
    _impl->slots.push_back(m);
    if (_impl->h32)
        add_halves(*_impl, m, csz);
    if (_impl->mmgw)
        add_halves2(*_impl, m, csz);
    return (int)_impl->slots.size() - 1;
}

void Engine::set(int slot, const uint32_t* data, int size)
{
    if (_impl->wk == 0)
        throw std::runtime_error("GPU engine: use set_digits() with a non-binary digit base.");
    std::vector<double> digits;
    if (_ibdwt)
        ib_to_digits(data, size, digits);
    else
        to_digits(data, size, digits);
    write_slot(*_impl, _impl->slots[slot], digits);
}

void Engine::to_digits(const uint32_t* data, int size, std::vector<double>& digits)
{
    digits.assign(2*(size_t)_D, 0.0);
    uint64_t total_bits = (uint64_t)size*32;
    int64_t carry = 0;
    int64_t half = (int64_t)1 << (_w - 1);
    for (uint32_t i = 0; i < 2*_D; i++)
    {
        uint64_t pos = (uint64_t)i*_w;
        int64_t v = 0;
        if (pos < total_bits)
        {
            uint64_t limb = pos/32, sh = pos%32;
            uint64_t window = data[limb];
            if (limb + 1 < (uint64_t)size) window |= (uint64_t)data[limb + 1] << 32;
            v = (int64_t)((window >> sh) & (((uint64_t)1 << _w) - 1));
        }
        else if (carry == 0)
            break;
        v += carry;
        if (v >= half) { v -= (int64_t)1 << _w; carry = 1; }
        else carry = 0;
        if (i >= _D && v != 0 && !_mmgw)
            throw std::runtime_error("GPU engine: value too large for FFT size.");
        digits[i] = (double)v;
    }
    // MMGW: all 2D digits hold the value (mod 2^K - 1 for -N^-1: the carry out of the top wraps around)
    if (_mmgw && carry != 0)
        digits[0] += (double)carry;
}

bool Engine::get(int slot, std::vector<uint32_t>& magnitude)
{
    if (_impl->wk == 0)
        throw std::runtime_error("GPU engine: use get_digits() with a non-binary digit base.");
    std::vector<double> digits(2*(size_t)_D);
    read_slot(*_impl, _impl->slots[slot], digits);
    return this->magnitude(digits, magnitude);
}

// Binary digits (2D of them) -> sign and magnitude.
bool Engine::magnitude(const std::vector<double>& digits, std::vector<uint32_t>& magnitude) const
{
    if (_ibdwt)
        return ib_magnitude(digits, _n, 2*(uint64_t)_D, magnitude);
    // Sum positive and negative digits separately (carries stay short), then subtract.
    size_t limbs = ((size_t)2*_D*_w + 63)/32 + 2;
    std::vector<uint32_t> pos(limbs, 0), neg(limbs, 0);
    for (uint32_t i = 0; i < 2*_D; i++)
    {
        int64_t d = (int64_t)digits[i];
        if (d == 0)
            continue;
        std::vector<uint32_t>& acc = d > 0 ? pos : neg;
        uint64_t v = (uint64_t)(d > 0 ? d : -d);     // < 2^24
        uint64_t bit = (uint64_t)i*_w;
        size_t j = bit/32;
        v <<= bit%32;                                  // < 2^56
        uint64_t carry = 0;
        for (; j < limbs && (v != 0 || carry != 0); j++)
        {
            uint64_t s = (uint64_t)acc[j] + (uint32_t)v + carry;
            acc[j] = (uint32_t)s;
            carry = s >> 32;
            v >>= 32;
        }
    }
    int cmp = 0;
    for (size_t j = limbs; j-- > 0 && cmp == 0; )
        if (pos[j] != neg[j])
            cmp = pos[j] > neg[j] ? 1 : -1;
    bool negative = cmp < 0;
    std::vector<uint32_t>& big = negative ? neg : pos;
    std::vector<uint32_t>& small = negative ? pos : neg;
    int64_t borrow = 0;
    for (size_t j = 0; j < limbs; j++)
    {
        int64_t s = (int64_t)big[j] - small[j] - borrow;
        borrow = s < 0;
        big[j] = (uint32_t)(s + (borrow << 32));
    }
    while (!big.empty() && big.back() == 0)
        big.pop_back();
    magnitude.swap(big);
    return negative;
}

void Engine::copy(int dst, int src)
{
    if (dst == src)
        return;
    check(clEnqueueCopyBuffer(_impl->queue, _impl->slots[src], _impl->slots[dst], 0, 0, sizeof(double)*2*(size_t)_D, 0, nullptr, nullptr), "copy");
}

// FFT plan: ceil(log2(D)/8) local-memory passes of up to radix 256, bits split evenly.
static void build_lds_plan(Engine::Impl& I, uint32_t D)
{
    cl_ulong lmem = 0;
    size_t maxwg = 256;
    lmem = local_mem(I);
    clGetDeviceInfo(I.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    if (maxwg > 256) maxwg = 256;
    // points per work-group: power of two with (P + P/16 + 1)*16 bytes fitting in local memory (max 2048)
    uint32_t P = 2048;
    while (P > 64 && ((cl_ulong)P + P/16 + 1)*16 + 1024 > lmem)
        P >>= 1;
    // D = P2 or 3*P2: local-memory passes cover the power-of-two factor, a radix-3 pass follows
    I.radix3 = !is_pow2(D);
    uint32_t P2 = I.radix3 ? D/3 : D;
    uint32_t lg = ilog2(P2);
    uint32_t np = (lg + 7)/8;
    // PRST_GPU_PASSES=n: split the power-of-two part into at least n passes (smaller radix,
    // larger work-groups). For tuning; the default uses the fewest passes (radix up to 256).
    if (const char* e = getenv("PRST_GPU_PASSES"))
    {
        uint32_t want = (uint32_t)atoi(e);
        if (want > np && want <= lg)
            np = want;
    }
    uint32_t Ns = 1;
    std::map<std::string, cl_kernel> cache;
    auto build_variant = [&](const std::string& opts) { return build_variant_impl(I, cache, opts); };
    for (uint32_t t = 0; t < np; t++)
    {
        uint32_t m = lg/np + (t < lg%np ? 1 : 0);
        uint32_t RA = 1u << ((m + 1)/2), RB = 1u << (m/2);
        uint32_t R = RA*RB;
        uint32_t G = P/R;
        if (G > P2/R) G = P2/R;
        while (G > 1 && G*RA > maxwg) G >>= 1;
        if (G < 1) G = 1;
        uint32_t T = G*RA;
        std::string opts = "-DRA=" + std::to_string(RA) + " -DRB=" + std::to_string(RB) + " -DRG=" + std::to_string(G) + " -DRT=" + std::to_string(T) + " -DLR=" + std::to_string(m);
        cl_kernel k = build_variant(opts);
        I.lds_plan.push_back({ k, R, Ns, G, T });
        // fused variant of the last pass (pointwise product in the epilogue): needs at least two
        // passes (Ns > 1) and an even group. Off by default: it measured no faster than the separate
        // realmul pass on RDNA3. PRST_GPU_FUSE=1 enables it for comparison on other GPUs.
        const char* fuse_env = getenv("PRST_GPU_FUSE");
        bool fuse = fuse_env != nullptr && fuse_env[0] == '1';
        if (fuse && !I.radix3 && t == np - 1 && np > 1 && G >= 2 && G/2 <= Ns/2)
            I.lds_fused = { build_variant(opts + " -DFUSEMUL"), R, Ns, G, T };
        Ns *= R;
    }
}

static cl_kernel build_variant_impl(Engine::Impl& I, std::map<std::string, cl_kernel>& cache, const std::string& opts)
{
        cl_kernel k;
        auto it = cache.find(opts);
        if (it != cache.end())
            k = it->second;
        else
        {
            cl_int err;
            cl_program prog = clCreateProgramWithSource(I.ctx, 1, &lds_source, nullptr, &err);
            check(err, "clCreateProgramWithSource(lds)");
            err = clBuildProgram(prog, 1, &I.device, opts.c_str(), nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                size_t len = 0;
                clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
                std::string log(len, '\0');
                clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
                throw std::runtime_error("OpenCL build failed (" + opts + "):\n" + log);
            }
            k = clCreateKernel(prog, "fft_lds", &err);
            check(err, "fft_lds");
            I.lds_progs.push_back(prog);
            I.lds_kernels.push_back(k);
            cache[opts] = k;
        }
        return k;
}

void Engine::snapshot(int a, int b)
{
    Impl& I = *_impl;
    snapshot_wait();
    size_t bytes = (I.islot ? sizeof(cl_int) : sizeof(double))*2*(size_t)_D;
    int sl[2] = { a, b };
    for (int k = 0; k < 2; k++)
    {
        if (sl[k] < 0)
            continue;
        I.sraw[k].resize(bytes);
        check(clEnqueueReadBuffer(I.queue, I.slots[sl[k]], CL_FALSE, 0, bytes, I.sraw[k].data(), 0, nullptr, nullptr), "snapshot");
    }
    check(clEnqueueReadBuffer(I.queue, I.maxerr, CL_FALSE, 0, sizeof(cl_uint), &I.smaxerr, 0, nullptr, nullptr), "snapshot");
    check(clEnqueueReadBuffer(I.queue, I.errflag, CL_FALSE, 0, sizeof(cl_uint), &I.serrflag, 0, nullptr, nullptr), "snapshot");
    // reset after the reads (in-order queue); the pattern is copied at enqueue time
    cl_uint zero = 0;
    check(clEnqueueFillBuffer(I.queue, I.maxerr, &zero, sizeof(zero), 0, sizeof(zero), 0, nullptr, nullptr), "snapshot");
    check(clEnqueueFillBuffer(I.queue, I.errflag, &zero, sizeof(zero), 0, sizeof(zero), 0, nullptr, nullptr), "snapshot");
    check(clEnqueueMarkerWithWaitList(I.queue, 0, nullptr, &I.sevent), "snapshot");
    clFlush(I.queue);
}

void Engine::snapshot_wait()
{
    Impl& I = *_impl;
    if (I.sevent == nullptr)
        return;
    wait_event(I.sevent);
    clReleaseEvent(I.sevent);
    I.sevent = nullptr;
}

double Engine::snapshot_roundoff() const
{
    float f;
    std::memcpy(&f, &_impl->smaxerr, sizeof(f));
    return f;
}

bool Engine::snapshot_carry_error() const
{
    return _impl->serrflag != 0;
}

static void raw_digits(const Engine::Impl& I, const std::vector<char>& raw, std::vector<double>& d)
{
    if (I.islot)
    {
        d.resize(raw.size()/sizeof(cl_int));
        const cl_int* p = (const cl_int*)raw.data();
        for (size_t i = 0; i < d.size(); i++)
            d[i] = (double)p[i];
    }
    else
    {
        d.resize(raw.size()/sizeof(double));
        std::memcpy(d.data(), raw.data(), raw.size());
    }
}

void Engine::snapshot_digits(int k, std::vector<int64_t>& digits) const
{
    std::vector<double> d;
    raw_digits(*_impl, _impl->sraw[k], d);
    digits.resize(d.size());
    for (size_t i = 0; i < d.size(); i++)
        digits[i] = (int64_t)d[i];
    while (!digits.empty() && digits.back() == 0)
        digits.pop_back();
}

bool Engine::snapshot_get(int k, std::vector<uint32_t>& mag) const
{
    std::vector<double> d;
    raw_digits(*_impl, _impl->sraw[k], d);
    return magnitude(d, mag);
}

void Engine::finish()
{
    check(clFinish(_impl->queue), "clFinish");
}

// Wait without busy-spinning: NVIDIA's clWaitForEvents/clFinish spin a CPU core at 100%.
static void wait_event(cl_event ev)
{
    for (int spins = 0; ; spins++)
    {
        cl_int status = CL_COMPLETE;
        if (clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr) != CL_SUCCESS || status <= CL_COMPLETE)
            return;
        if (spins < 100)
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static bool event_done(cl_event ev)
{
    cl_int status = CL_COMPLETE;
    return clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr) != CL_SUCCESS || status <= CL_COMPLETE;
}

static void throttle(Engine::Impl& I)
{
    // Keep the command queue bounded but never empty: up to 3 batches in flight, and the batch
    // grows whenever the GPU had already finished the previous batch by the time the next one
    // was submitted (small FFTs, or a long OS sleep in wait_event). Waiting sleeps instead of
    // spinning, so the host thread stays nearly idle on every vendor.
    if (++I.ops_since_sync < I.batch)
        return;
    I.ops_since_sync = 0;
    cl_event ev;
    check(clEnqueueMarkerWithWaitList(I.queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList");
    clFlush(I.queue);
    if (!I.pending.empty() && event_done(I.pending.back()) && I.batch < 4096)
        I.batch *= 2;                          // queue had drained: feed more work per batch
    I.pending.push_back(ev);
    while (I.pending.size() > 3)
    {
        wait_event(I.pending.front());
        clReleaseEvent(I.pending.front());
        I.pending.pop_front();
    }
}

namespace
{
    template<class T>
    void arg(cl_kernel k, cl_uint i, const T& v) { check(clSetKernelArg(k, i, sizeof(T), &v), "clSetKernelArg"); }
}

static void set_lds_args(Engine::Impl& I, Engine::Impl::LdsPass& p, cl_mem in, cl_mem out, uint32_t D, int inv)
{
    arg(p.k, 0, in); arg(p.k, 1, out); arg(p.k, 2, I.tw); arg(p.k, 3, (cl_uint)D); arg(p.k, 4, (cl_uint)p.Ns); arg(p.k, 5, (cl_int)inv);
    arg(p.k, 6, I.twlo); arg(p.k, 7, I.twhi); arg(p.k, 8, (cl_uint)I.lbits);
    arg(p.k, 9, (cl_uint)(D/p.R)); arg(p.k, 10, (cl_uint)(D/(p.Ns*p.R))); arg(p.k, 11, (cl_uint)ilog2(p.Ns));
}

// prefix: extra source lines in front of v2_source (macros that cannot go through -D: some
// drivers keep the quotes of a quoted -D value).
// v2_source in single precision (hybrid engine): double -> float, literals -> float literals.
static const std::string& v2_source_f32()
{
    static std::string src;
    if (src.empty())
    {
        std::string t = v2_source;
        // FP64 only for double slots (SLOTT=double2): also builds on devices without it
        t = std::regex_replace(t, std::regex("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"),
                               "#ifdef cl_khr_fp64\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#endif");
        t = std::regex_replace(t, std::regex("double2"), "float2");
        t = std::regex_replace(t, std::regex("\\bdouble\\b"), "float");
        t = std::regex_replace(t, std::regex("(\\b[0-9]+\\.[0-9]+)(?![0-9fFeE])"), "$1f");
        src = t;
    }
    return src;
}

static cl_kernel build_v2(Engine::Impl& I, const std::string& opts, const char* name, const std::string& prefix = "")
{
    cl_int err;
    const char* srcs[2] = { prefix.c_str(), I.fp32 ? v2_source_f32().c_str() : v2_source };
    cl_program prog = clCreateProgramWithSource(I.ctx, 2, srcs, nullptr, &err);
    check(err, "clCreateProgramWithSource(v2)");
    // PRST_GPU_V2OPTS: extra compiler options for the FFT kernels (tuning)
    std::string o2 = opts;
    if (const char* e = getenv("PRST_GPU_V2OPTS")) o2 += std::string(" ") + e;
    err = clBuildProgram(prog, 1, &I.device, o2.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len = 0;
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        throw std::runtime_error("OpenCL build failed (" + opts + "):\n" + log);
    }
    cl_kernel k = clCreateKernel(prog, name, &err);
    check(err, name);
    I.lds_progs.push_back(prog);
    I.lds_kernels.push_back(k);
    return k;
}

// FFT v2 plan: rows of W points (power of two, local memory), H = D/W columns-dimension length
// split into Stockham passes of radix <= 16 (times 3 in the last pass when D = 3*2^k).
static void build_v2_plan(Engine::Impl& I, uint32_t D, uint32_t P)
{
    uint32_t P2 = is_pow2(D) ? D : D/3;
    uint32_t W = P2 < P ? P2 : P;
    if (const char* e = getenv("PRST_GPU_ROW"))
    {
        uint32_t w = (uint32_t)atoi(e);
        if (w >= 2 && is_pow2(w) && w <= P2 && w <= P)
            W = w;
    }
    uint32_t H = D/W;
    I.v2 = true;
    I.W = W; I.H = H; I.lgW = ilog2(W);
    // column passes of radix <= 16 (registers): the factor 3 counts as 2 bits and goes to the last
    // pass, the power-of-two bits are spread evenly (H = 3*512: radices 16, 16, 6)
    uint32_t r3 = is_pow2(H) ? 1 : 3;
    uint32_t hb = ilog2(H/r3);
    // mode 1: radix <= 8 per pass and the factor 3 in a pass of its own. Measured faster (RDNA3)
    // when H = 3*8^j (radix-8 passes 8,8,...,3), slower otherwise. PRST_GPU_COLMODE=0/1 overrides.
    const char* cm = getenv("PRST_GPU_COLMODE");
    uint32_t mode = cm != nullptr ? (uint32_t)atoi(cm) : (r3 == 3 && hb % 3 == 0 && !I.fp32 ? 1 : 0);
    uint32_t maxb = mode == 1 ? 3 : 4;
    uint32_t np = mode == 1 ? (hb + 2)/3 + (r3 == 3 ? 1 : 0) : (hb + (r3 == 3 ? 2 : 0) + 3)/4;
    std::vector<uint32_t> cap(np, maxb), bits(np, 0);
    if (np > 0 && r3 == 3)
        cap[np - 1] = mode == 1 ? 0 : 2;
    for (uint32_t b = 0; b < hb; b++)
    {
        uint32_t best = 0;
        for (uint32_t t = 1; t < np; t++)
            if (cap[t] - bits[t] > cap[best] - bits[best])
                best = t;
        bits[best]++;
    }
    // PRST_GPU_COLS=b1,b2,...: bits of the power-of-two part per column pass (tuning; must add up)
    if (const char* e = getenv("PRST_GPU_COLS"))
    {
        std::vector<uint32_t> user;
        uint32_t sum = 0;
        for (const char* p = e; *p; )
        {
            uint32_t v = (uint32_t)strtoul(p, (char**)&p, 10);
            user.push_back(v);
            sum += v;
            if (*p == ',') p++;
        }
        bool ok = sum == hb && !user.empty();
        for (size_t i = 0; i < user.size(); i++)
            ok = ok && user[i] <= 5 && (user[i] > 0 || (i + 1 == user.size() && r3 == 3));
        if (ok)
        {
            bits = user;
            np = (uint32_t)bits.size();
        }
    }
    std::map<std::string, cl_kernel> cache;
    auto variant = [&](const std::string& opts)
    {
        auto it = cache.find(opts);
        return it != cache.end() ? it->second : (cache[opts] = build_v2(I, opts, "col_pass"));
    };
    uint32_t Ns = 1;
    for (uint32_t t = 0; t < np; t++)
    {
        uint32_t R2 = 1u << bits[t], R3 = t + 1 == np ? r3 : 1;
        std::string opts = "-DCOLPASS -DR2=" + std::to_string(R2) + " -DR3=" + std::to_string(R3);
        // radix-8 passes after the first (which has no twiddles): tangent-form twiddles (fancy8)
        const char* fe = getenv("PRST_GPU_FANCY");
        bool fancy = R2 == 8 && R3 == 1 && Ns > 1 && !(fe != nullptr && fe[0] == '0') && !I.fp32;
        cl_mem F = nullptr;
        if (fancy)
        {
            const long double PI = 3.141592653589793238462643383279502884L;
            uint32_t step = D/(Ns*8);
            std::vector<double> f(16*(size_t)Ns);
            for (uint32_t jm = 0; jm < Ns; jm++)
            {
                double c[8], tt[8];
                uint64_t bits = 0;
                c[0] = 1; tt[0] = 0;
                for (uint32_t r = 1; r < 8; r++)
                {
                    uint64_t m = ((uint64_t)jm*r*step) % D;
                    long double th = -2*PI*(long double)m/D;
                    long double k = roundl(th/(PI/2));
                    long double tp = th - k*(PI/2);
                    int q = ((int)k % 4 + 4) % 4;
                    c[r] = (double)cosl(tp);
                    tt[r] = (double)tanl(tp);
                    bits |= (uint64_t)q << (2*(r - 1));
                }
                double* e = f.data() + 16*(size_t)jm;
                for (int r = 1; r < 8; r++) e[r - 1] = tt[r];
                e[7] = c[4]; e[8] = c[2]; e[9] = c[6]/c[2]; e[10] = c[1]; e[11] = c[5]/c[1]; e[12] = c[7]/c[3]; e[13] = c[3]/c[1];
                e[14] = 0.70710678118654752440*c[1];
                std::memcpy(&e[15], &bits, sizeof(double));
            }
            cl_int err;
            F = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*f.size(), f.data(), &err); check(err, "fancy");
        }
        std::string fo = fancy ? " -DFANCY" : "";
        // FP32 transform: the first forward pass reads the slot (double digits)
        std::string cv = I.fp32 && t == 0 ? std::string(" -DCONVD -DSLOTT=") + (I.islot ? "int2" : "double2") : "";
        cl_kernel kf = variant(opts + fo + cv + " -DINV=0"), ki = variant(opts + fo + " -DINV=1");
        // first pass, upper half of the input zero: lets the compiler drop half of the butterfly inputs
        cl_kernel kz = t == 0 && (R2*R3) % 2 == 0 && H % 2 == 0 ? variant(opts + cv + " -DINV=0 -DHALFZ -cl-no-signed-zeros -cl-finite-math-only") : nullptr;
        I.col_plan.push_back({ kf, ki, kz, R2*R3, Ns, F });
        Ns *= R2*R3;
    }
    // row FFT: RW/RV threads holding RV values each, stages of radix RV (the last one smaller).
    // RV = 8 by default (no register spills, good occupancy); PRST_GPU_ROWV=4/16 for tuning.
    uint32_t RV = 8;
    if (const char* e = getenv("PRST_GPU_ROWV"))
    {
        uint32_t v = (uint32_t)atoi(e);
        if (v == 4 || v == 8 || v == 16)
            RV = v;
    }
    if (RV > W)
        RV = W;
    uint32_t RT = W/RV;
    for (uint32_t ns = 1; ns < W; )
    {
        uint32_t rs = W/ns < RV ? W/ns : RV;
        I.row_stages.emplace_back(ns, rs);
        ns *= rs;
    }
    std::string fwd, inv, fwdt;
    size_t nst = I.row_stages.size();
    for (size_t i = 0; i < nst; i++)
    {
        std::string a = "(" + std::to_string(I.row_stages[i].first) + "u," + std::to_string(I.row_stages[i].second) + "u,";
        fwd += "FSTAGE" + a + (i == 0 ? "1," : "0,") + (i + 1 == nst ? "1)" : "0)");
        fwdt += "FSTAGE" + a + (i == 0 ? "1," : "0,") + "0)";
        inv = "ISTAGE" + a + (i + 1 == nst ? "1," : "0,") + (i == 0 ? "1)" : "0)") + inv;
    }
    // rows per work-group: at least 64 threads, within the local memory budget (P points)
    uint32_t RG = 1;
    while (RG*RT < 64 && 2*RG*W <= P && H % (2*RG) == 0)
        RG *= 2;
    if (const char* e = getenv("PRST_GPU_ROWG"))
    {
        uint32_t g = (uint32_t)atoi(e);
        if (g >= 1 && is_pow2(g) && g*W <= P && H % g == 0 && g*RT <= 256)
            RG = g;
    }
    // FP32 (hybrid): row twiddles computed (TWC). PRST_GPU_TWC=0 keeps the table.
    {
        const char* te = getenv("PRST_GPU_TWC");
        I.twc = I.fp32 && !(te != nullptr && te[0] == '0');
    }
    std::string twco = I.twc ? " -DTWC -DLGRV=" + std::to_string(ilog2(RV)) : "";
    std::string opts = "-DROWPASS -DRW=" + std::to_string(W) + "u -DRT=" + std::to_string(RT) + " -DRV=" + std::to_string(RV) + "u -DRG=" + std::to_string(RG) + twco;
    std::string prefix = "#define STAGES_F " + fwd + "\n#define STAGES_I " + inv + "\n#define STAGES_FT " + fwdt + "\n";
    I.k_row = build_v2(I, opts, "row_fwd", prefix);
    I.k_rowi = build_v2(I, opts, "row_inv", prefix);
    // fused tail: needs column passes (the row FFT is not the whole transform), an even H, room for
    // two rows in local memory and 2*RT work items. On by default only with the radix-8 column plan
    // (H = 3*8^j): measured 1-2% faster there and up to 7% slower otherwise (RDNA3, FP64 bound).
    // PRST_GPU_TAIL=0/1 forces it off/on (2: on, swizzled local memory).
    {
        const char* te = getenv("PRST_GPU_TAIL");
        cl_ulong lmem = 0;
        size_t maxwg = 0;
        lmem = local_mem(I);
        clGetDeviceInfo(I.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
        cl_ulong es = I.fp32 ? 8 : 16;
        cl_ulong need = 2*((cl_ulong)W + W/16 + 1)*es;
        // padded rows when they fit, else the swizzled layout (2*W*16 bytes); PRST_GPU_TAIL=2 forces it
        bool swz = need > lmem || (te != nullptr && te[0] == '2');
        if (swz)
            need = 2*(cl_ulong)W*es;
        bool want = te != nullptr ? te[0] != '0' : (mode == 1 || I.fp32);
        if (want && np > 0 && H % 2 == 0 && RV >= 2 && need <= lmem && 2*RT <= maxwg)
        {
            std::string topts = "-DROWPASS -DRW=" + std::to_string(W) + "u -DRT=" + std::to_string(RT) + " -DRV=" + std::to_string(RV) + "u -DRG=1 -DTAILK" + (swz ? " -DSWZ" : "") + twco;
            // optional: if it does not build on this device, the rows and the product run separately
            topts += I.fp32 && !is_pow2(D) ? " -DPSCS=0.57735026918962576451 -DPSCM=0.33333333333333333333" : "";
            try { I.k_tail = build_v2(I, topts, "row_tail", prefix); }
            catch (const std::exception&) { I.k_tail = nullptr; }
            I.tail_threads = 2*RT;
            I.tail_opts = topts;
            I.tail_prefix = prefix;
        }
    }
    I.row_threads = RT*RG;
    I.row_groups = H/RG;
    // FP32 hybrid, D = 3*2^k: scale the product by 1/3 (sqrt(1/3) per operand when squaring)
    std::string psc = I.fp32 && !is_pow2(D) ? " -DPSCS=0.57735026918962576451 -DPSCM=0.33333333333333333333" : "";
    I.k_realmul2 = build_v2(I, "-DREALMUL2" + psc, "realmul2");
}

// FP32 tables of the inner FP32 transform (hybrid): column twiddles, pair product u, row twiddles.
static void make_v2_tables_f32(Engine::Impl& I, uint32_t D)
{
    const long double PI = 3.141592653589793238462643383279502884L;
    cl_int err;
    auto w = [&](uint64_t m, float* o) { long double a = 2*PI*(long double)(m % D)/D; o[0] = (float)cosl(a); o[1] = (float)-sinl(a); };
    std::vector<float> tw(2*(size_t)D);
    for (uint32_t m = 0; m < D; m++)
        w(m, &tw[2*m]);
    I.tw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*tw.size(), tw.data(), &err); check(err, "tw32");
    size_t n = (size_t)(I.H/2 + 1)*I.W;
    std::vector<float> t2(2*n);
    for (size_t s = 0; s < n; s++)
    {
        uint32_t k2 = (uint32_t)(s >> I.lgW), k1 = (uint32_t)(s & (I.W - 1));
        uint64_t f = k2 + (uint64_t)I.H*k1;
        if (2*f > D) f = D - f;
        long double a = 2*PI*f/D;
        t2[2*s] = (float)(-(1 + cosl(a))/4);
        t2[2*s + 1] = (float)(sinl(a)/4);
    }
    I.tw2s = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*t2.size(), t2.data(), &err); check(err, "tw2s32");
    if (I.twc)
    {
        // st[Ns - 1 + jm*(Rs - 1) + r - 1] = W_D^(r*jm*H*q), rr[64*k2 + 16*stage + r - 1] = W_D^(r*k2*q), q = W/(Ns*Rs)
        std::vector<float> st(2*(size_t)I.W, 0.0f), rr(2*64*(size_t)I.H, 0.0f);
        uint32_t stage = 0;
        for (auto& sg : I.row_stages)
        {
            uint32_t Ns = sg.first, Rs = sg.second;
            uint64_t q = I.W/(Ns*Rs);
            for (uint32_t jm = 0; jm < Ns; jm++)
                for (uint32_t r = 1; r < Rs; r++)
                    w((uint64_t)r*jm*I.H*q, &st[2*(Ns - 1 + jm*(Rs - 1) + r - 1)]);
            for (uint32_t k2 = 0; k2 < I.H; k2++)
                for (uint32_t r = 1; r < Rs; r++)
                    w((uint64_t)r*k2*q, &rr[2*(64*(size_t)k2 + 16*stage + r - 1)]);
            stage++;
        }
        I.rowtw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*st.size(), st.data(), &err); check(err, "rowst32");
        I.rowtr = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*rr.size(), rr.data(), &err); check(err, "rowtr32");
        return;
    }
    std::vector<float> rt(2*(size_t)D, 0.0f);
    for (uint32_t k2 = 0; k2 < I.H; k2++)
        for (auto& st : I.row_stages)
        {
            uint32_t Ns = st.first, Rs = st.second;
            uint64_t q = I.W/(Ns*Rs);
            for (uint32_t jm = 0; jm < Ns; jm++)
                for (uint32_t r = 1; r < Rs; r++)
                    w(r*((uint64_t)jm*I.H + k2)*q, &rt[2*((size_t)k2*I.W + Ns - 1 + jm*(Rs - 1) + r - 1)]);
        }
    I.rowtw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*rt.size(), rt.data(), &err); check(err, "rowtw32");
}

static void run_col_passes(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half)
{
    size_t np = I.col_plan.size();
    cl_mem in = src;
    for (size_t t = 0; t < np; t++)
    {
        cl_mem out = ((np - t) % 2 == 1) ? dst : tmp;
        auto& p = I.col_plan[t];
        cl_kernel k = inv ? p.ki : (half && p.kz != nullptr ? p.kz : p.kf);
        cl_uint total = D/p.R;
        arg(k, 0, in); arg(k, 1, out); arg(k, 2, I.tw); arg(k, 3, (cl_uint)I.W); arg(k, 4, (cl_uint)I.lgW);
        arg(k, 5, (cl_uint)(I.H/p.R)); arg(k, 6, (cl_uint)p.Ns); arg(k, 7, (cl_uint)(D/(p.Ns*p.R))); arg(k, 8, (cl_uint)(D/p.R));
        arg(k, 9, total); arg(k, 10, p.F ? p.F : I.tw);
        I.run_local(k, (total + 255)/256*256, 256);
        in = out;
    }
}

static void run_row(Engine::Impl& I, cl_mem in, cl_mem out, int inv)
{
    cl_kernel k = inv ? I.k_rowi : I.k_row;
    arg(k, 0, in); arg(k, 1, out); arg(k, 2, I.rowtw); arg(k, 3, I.rowtr ? I.rowtr : I.rowtw);
    I.run_local(k, (size_t)I.row_groups*I.row_threads, I.row_threads);
}

// half: src is a value (slot) with nothing in the upper D reals (forward only).
static void run_fft(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half)
{
    if (I.h32)
    {
        // forward: both transforms read the slot (double digits); inverse: each its half
        auto& d = I.halves.at(dst);
        auto& t = I.halves.at(tmp);
        cl_mem s32 = inv ? I.halves.at(src).first : src, s31 = inv ? I.halves.at(src).second : src;
        if (!I.hcol.empty())
        {
            // forward: fused columns, then the rows of each; inverse: the tails already did the rows
            h32_cols(I, s32, s31, dst, tmp, D, inv, !inv && half && I.halfzero);
            if (!inv)
            {
                run_row(*I.i32, d.first, d.first, 0);
                ntt_row(*I.i31, d.second, d.second, 0);
            }
            return;
        }
        run_fft(*I.i32, s32, d.first, t.first, D, inv, half);
        ntt_fft(*I.i31, s31, d.second, t.second, inv, half);
        return;
    }
    if (I.ntt)
    {
        ntt_fft(I, src, dst, tmp, inv, half);
        return;
    }
    half = half && I.halfzero;
    if (I.v2)
    {
        size_t np = I.col_plan.size();
        if (!inv)
        {
            // columns (ending in dst), then the rows in place
            if (np == 0)
                run_row(I, src, dst, 0);
            else
            {
                run_col_passes(I, src, dst, tmp, D, 0, half);
                run_row(I, dst, dst, 0);
            }
        }
        else if (I.k_tail)
            run_col_passes(I, src, dst, tmp, D, 1, false);   // rows were done by row_tail (fft_product)
        else
        {
            if (np == 0)
                run_row(I, src, dst, 1);
            else
            {
                cl_mem mid = np % 2 == 1 ? tmp : dst;
                run_row(I, src, mid, 1);
                run_col_passes(I, mid, dst, tmp, D, 1, false);
            }
        }
        return;
    }
    size_t np = I.lds_plan.size();
    size_t total = np + (I.radix3 ? 1 : 0);
    cl_mem in = src;
    for (size_t t = 0; t < np; t++)
    {
        cl_mem out = ((total - t) % 2 == 1) ? dst : tmp;
        auto& p = I.lds_plan[t];
        set_lds_args(I, p, in, out, D, inv);
        I.run_local(p.k, (size_t)(D/p.R/p.G)*p.T, p.T);
        in = out;
    }
    if (I.radix3)
    {
        arg(I.k_r3, 0, in); arg(I.k_r3, 1, dst); arg(I.k_r3, 2, I.tw); arg(I.k_r3, 3, (cl_uint)D); arg(I.k_r3, 4, (cl_int)inv);
        I.run(I.k_r3, (D/3 + 63) & ~(size_t)63);
    }
}

// ---------------------------------------------------------------------------------------------
// NTT engine host side. GF(p^2), p = 2^61 - 1, elements (x, y) = x + y*i, i^2 = -1.
namespace
{
    typedef unsigned __int128 u128;
    // current field: p = 2^GB - 1 (61 or 31), or P3 (GMERS false), set by build_ntt_plan
    int GB = 61;
    bool GMERS = true;
    uint64_t GP = ((uint64_t)1 << 61) - 1;
    uint64_t gred(u128 x)
    {
        if (!GMERS)
            return (uint64_t)(x % GP);
        uint64_t r = (uint64_t)(x & GP) + (uint64_t)(x >> GB);
        r = (r & GP) + (r >> GB);
        r = (r & GP) + (r >> GB);
        return r >= GP ? r - GP : r;
    }
    struct GE { uint64_t x, y; };
    bool operator==(GE a, GE b) { return a.x == b.x && a.y == b.y; }
    GE gm(GE a, GE b)
    {
        uint64_t xx = gred((u128)a.x*b.x), yy = gred((u128)a.y*b.y), xy = gred((u128)a.x*b.y), yx = gred((u128)a.y*b.x);
        return { xx >= yy ? xx - yy : xx + GP - yy, (xy + yx) % GP };
    }
    GE gpow(GE a, uint64_t e)
    {
        GE r = { 1, 0 };
        for (; e != 0; e >>= 1, a = gm(a, a))
            if (e & 1)
                r = gm(r, a);
        return r;
    }
    // generator of the norm-1 subgroup (order p + 1): z^(p-1) has norm 1; a generator iff no
    // (p+1)/q-th power is 1 for the prime factors q of p + 1
    GE norm1_generator()
    {
        uint64_t n = GP + 1, r = n;
        std::vector<uint64_t> qs;
        for (uint64_t q = 2; q*q <= r; q++)
            if (r % q == 0) { qs.push_back(q); while (r % q == 0) r /= q; }
        if (r > 1) qs.push_back(r);
        for (uint64_t a = 2; ; a++)
        {
            GE y = gpow({ a, 1 }, GP - 1);
            bool gen = true;
            for (uint64_t q : qs)
                if (gpow(y, n/q) == GE{ 1, 0 })
                    gen = false;
            if (gen)
                return y;
        }
    }
    std::string hex(uint64_t v) { char b[32]; snprintf(b, sizeof(b), GB == 61 ? "0x%llxUL" : "0x%llxU", (unsigned long long)v); return b; }
}

static cl_kernel build_ntt(Engine::Impl& I, const std::string& opts, const char* name, const std::string& prefix = "")
{
    cl_int err;
    const char* srcs[2] = { prefix.c_str(), ntt_source };
    cl_program prog = clCreateProgramWithSource(I.ctx, 2, srcs, nullptr, &err);
    check(err, "clCreateProgramWithSource(ntt)");
    err = clBuildProgram(prog, 1, &I.device, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len = 0;
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        throw std::runtime_error("OpenCL build failed (ntt " + opts + "):\n" + log);
    }
    cl_kernel k = clCreateKernel(prog, name, &err);
    check(err, name);
    I.lds_progs.push_back(prog);
    I.lds_kernels.push_back(k);
    return k;
}

// NTT plan: D = W*H (powers of two), rows of W points in local memory, column passes of radix <= 16.
static const uint64_t P3 = 57*((uint64_t)1 << 25) - 1;    // 1912602623, prime, p + 1 = 3*19*2^25
static void build_ntt_plan(Engine::Impl& I, uint32_t D, uint32_t P)
{
    uint32_t P2 = is_pow2(D) ? D : D/3;
    if (!is_pow2(D) && !I.f3)
        throw std::runtime_error("GPU NTT engine: this field needs a power-of-two length.");
    uint32_t W = P2 < P ? P2 : P;
    if (const char* e = getenv("PRST_GPU_ROW"))
    {
        uint32_t w = (uint32_t)atoi(e);
        if (w >= 16 && is_pow2(w) && w <= P && w < P2)
            W = w;
    }
    uint32_t H = D/W;
    if (H < 2 || H % 2 != 0)
        throw std::runtime_error("GPU NTT engine: transform too short.");
    I.W = W; I.H = H; I.lgW = ilog2(W);
    GB = I.f31 ? 31 : I.f3 ? 31 : 61;
    GMERS = !I.f3;
    GP = I.f3 ? P3 : ((uint64_t)1 << GB) - 1;
    std::string fopt = I.f31 ? " -DF31" : I.f3 ? " -DF3" : "";
    // roots of unity and constants
    GE y = norm1_generator();
    GE wD = gpow(y, (GP + 1)/D);
    std::vector<GE> tw(D);
    tw[0] = { 1, 0 };
    for (uint32_t m = 1; m < D; m++)
        tw[m] = gm(tw[m - 1], wD);
    GE w4 = tw[D/4];
    if (!(w4 == GE{ 0, 1 } || w4 == GE{ 0, GP - 1 }))
        throw std::runtime_error("GPU NTT engine: bad 4th root of unity.");
    auto cst = [&](const char* n, GE v) { return std::string(" -D") + n + "X=" + hex(v.x) + " -D" + n + "Y=" + hex(v.y); };
    GE w8 = tw[(uint64_t)D/8];
    const uint64_t A30 = (uint64_t)1 << ((GB - 1)/2);     // Mersenne fields: w8 = 2^((GB-1)/2)*(+-1 +-i)
    if (!I.f3 && !((w8.x == A30 || w8.x == GP - A30) && (w8.y == A30 || w8.y == GP - A30)))
        throw std::runtime_error("GPU NTT engine: unexpected 8th root of unity.");
    // P3: Montgomery form (x*2^32 mod p) for everything multiplied on the device
    uint64_t RM = I.f3 ? ((uint64_t)1 << 32) % GP : 1;
    auto toM = [&](GE v) { return I.f3 ? GE{ gred((u128)v.x*RM), gred((u128)v.y*RM) } : v; };
    auto cstM = [&](const char* n, GE v) { return cst(n, toM(v)); };
    std::string consts = std::string(" -DW4I=") + (w4.y == 1 ? "1" : "0") + " -DW8SX=" + (w8.x == A30 ? "1" : "-1") + " -DW8SY=" + (w8.y == A30 ? "1" : "-1") +
                         cstM("W8", w8) + cstM("W83", tw[3*((uint64_t)D/8)]) +
                         cstM("W161", tw[(uint64_t)D/16]) + cstM("W163", tw[3*((uint64_t)D/16)]) + cstM("W169", tw[9*((uint64_t)D/16)]) + fopt;
    for (uint32_t k = 1; k < 16; k++)
        consts += cstM(("W32_" + std::to_string(k)).c_str(), tw[(uint64_t)k*D/32]);
    if (I.f3)
    {
        uint64_t pinv = 1;                       // p^-1 mod 2^32 (Newton)
        for (int i = 0; i < 5; i++) pinv = (pinv*(2 - GP*pinv)) & 0xFFFFFFFFu;
        consts += " -DP3=" + hex(GP) + " -DPINV=" + hex(pinv);
        I.n3pinv = (uint32_t)pinv;
        if (D % 3 == 0)
        {
            GE w3 = toM(tw[D/3]);
            consts += " -DK3X=" + hex(w3.x) + " -DK3Y=" + hex(w3.y);
        }
        // carry: x = a*R/D (one Montgomery product with R^2/D), and 2^E mod p (as 2^E*R) for E < 64
        uint64_t dinv = gpow({ D, 0 }, 0).x;     // placeholder, computed below
        (void)dinv;
        u128 r2 = (u128)RM*RM % GP;
        uint64_t Dm = D % GP, inv = 1, bb = Dm, ee = GP - 2;
        for (; ee; ee >>= 1, bb = (uint64_t)((u128)bb*bb % GP)) if (ee & 1) inv = (uint64_t)((u128)inv*bb % GP);
        I.n3C = (uint32_t)((u128)r2*inv % GP);
        std::vector<uint32_t> pw(64);
        uint64_t v = RM;
        for (int e = 0; e < 64; e++, v = (v*2) % GP) pw[e] = (uint32_t)v;
        cl_int e2;
        I.n3pw = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*64, pw.data(), &e2); check(e2, "n3pw");
    }
    I.ndinv = GB - ilog2(D);                        // 1/D = 2^-k = 2^(GB-k): carry_fold rotates by this
    I.nconsts = consts;
    // column passes, bits spread evenly over passes of radix <= 16; a factor 3 goes to the last pass
    uint32_t r3 = is_pow2(H) ? 1 : 3;
    uint32_t hb = ilog2(H/r3);
    uint32_t np = (hb + (r3 == 3 ? 2 : 0) + 3)/4;
    std::vector<uint32_t> bits(np, 0), cap(np, 4);
    if (r3 == 3) cap[np - 1] = 2;
    for (uint32_t b = 0; b < hb; b++)
    {
        uint32_t best = 0;
        for (uint32_t t = 1; t < np; t++)
            if (cap[t] - bits[t] > cap[best] - bits[best])
                best = t;
        bits[best]++;
    }
    if (const char* e = getenv("PRST_GPU_COLS"))
    {
        std::vector<uint32_t> user;
        uint32_t sum = 0;
        for (const char* p = e; *p; )
        {
            uint32_t v = (uint32_t)strtoul(p, (char**)&p, 10);
            user.push_back(v);
            sum += v;
            if (*p == ',') p++;
            else if (*p) break;
        }
        bool ok = sum == hb && !user.empty();
        for (size_t i = 0; i < user.size(); i++)
            ok = ok && user[i] <= 5 && (user[i] >= 1 || (r3 == 3 && i + 1 == user.size()));
        if (ok)
        {
            bits = user;
            np = (uint32_t)bits.size();
        }
    }
    std::map<std::string, cl_kernel> cache;
    auto variant = [&](const std::string& opts)
    {
        auto it = cache.find(opts);
        return it != cache.end() ? it->second : (cache[opts] = build_ntt(I, opts + consts, "col_pass"));
    };
    uint32_t Ns = 1;
    for (uint32_t t = 0; t < np; t++)
    {
        uint32_t R2 = 1u << bits[t], R3 = t + 1 == np ? r3 : 1, R = R2*R3;
        std::string opts = "-DCOLPASS -DR2=" + std::to_string(R2) + " -DR3=" + std::to_string(R3);
        cl_kernel kf = variant(opts + " -DINV=0"), ki = variant(opts + " -DINV=1");
        std::string sn = I.islot ? " -DSLOTN=int2" : "";
        cl_kernel kc = t == 0 ? variant(opts + " -DINV=0 -DCONV" + sn) : nullptr;
        cl_kernel kcz = t == 0 && R % 2 == 0 ? variant(opts + " -DINV=0 -DCONV -DHALFZ" + sn) : nullptr;
        I.ntt_col.push_back({ kf, ki, kc, kcz, R, Ns });
        Ns *= R;
    }
    // rows: as v2 (RV values per thread, RG rows per work-group)
    uint32_t RV = 8;
    if (const char* e = getenv("PRST_GPU_ROWV"))
    {
        uint32_t v = (uint32_t)atoi(e);
        if (v == 4 || v == 8 || v == 16)
            RV = v;
    }
    if (RV > W)
        RV = W;
    uint32_t RT = W/RV;
    std::vector<std::pair<uint32_t, uint32_t>> stages;
    for (uint32_t ns = 1; ns < W; )
    {
        uint32_t rs = W/ns < RV ? W/ns : RV;
        stages.emplace_back(ns, rs);
        ns *= rs;
    }
    std::string fwd, inv, fwdt;
    size_t nst = stages.size();
    for (size_t i = 0; i < nst; i++)
    {
        std::string a = "(" + std::to_string(stages[i].first) + "u," + std::to_string(stages[i].second) + "u,";
        fwd += "FSTAGE" + a + (i == 0 ? "1," : "0,") + (i + 1 == nst ? "1)" : "0)");
        fwdt += "FSTAGE" + a + (i == 0 ? "1," : "0,") + "0)";
        inv = "ISTAGE" + a + (i + 1 == nst ? "1," : "0,") + (i == 0 ? "1)" : "0)") + inv;
    }
    uint32_t RG = 1;
    while (RG*RT < 64 && 2*RG*W <= P && H % (2*RG) == 0)
        RG *= 2;
    std::string opts = "-DROWPASS -DRW=" + std::to_string(W) + "u -DRT=" + std::to_string(RT) + " -DRV=" + std::to_string(RV) + "u -DRG=" + std::to_string(RG) + consts;
    std::string prefix = "#define STAGES_F " + fwd + "\n#define STAGES_I " + inv + "\n#define STAGES_FT " + fwdt + "\n";
    I.k_nrow = build_ntt(I, opts, "row_fwd", prefix);
    I.k_nrowi = build_ntt(I, opts, "row_inv", prefix);
    // fused tail (two rows per work-group); padded rows if they fit, else the swizzled layout.
    // PRST_GPU_NTAIL=0 disables.
    {
        const char* te = getenv("PRST_GPU_NTAIL");
        cl_ulong lmem = 0;
        size_t maxwg = 0;
        lmem = local_mem(I);
        clGetDeviceInfo(I.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
        cl_ulong es = I.f31 ? 8 : 16;
        cl_ulong need = 2*((cl_ulong)W + W/16 + 1)*es;
        bool swz = need > lmem;
        if (swz)
            need = 2*(cl_ulong)W*es;
        if (!(te != nullptr && te[0] == '0') && H % 2 == 0 && RV >= 2 && need <= lmem && 2*RT <= maxwg)
        {
            std::string topts = "-DROWPASS -DRW=" + std::to_string(W) + "u -DRT=" + std::to_string(RT) + " -DRV=" + std::to_string(RV) + "u -DRG=1 -DTAILK" + (swz ? " -DSWZ" : "") + consts;
            try { I.k_ntail = build_ntt(I, topts, "row_tail", prefix); }    // optional, as the FFT tail
            catch (const std::exception&) { I.k_ntail = nullptr; }
            I.ntail_threads = 2*RT;
            I.tail_opts = topts;
            I.tail_prefix = prefix;
        }
    }
    I.nrow_threads = RT*RG;
    I.nrow_groups = H/RG;
    I.k_nmul = build_ntt(I, "-DREALMUL" + consts, "realmul");
    // tables
    cl_int err;
    std::vector<GE> rt(D, GE{ 0, 0 });
    for (uint32_t k2 = 0; k2 < H; k2++)
        for (auto& st : stages)
        {
            uint32_t ns = st.first, rs = st.second;
            uint64_t q = W/(ns*rs);
            for (uint32_t jm = 0; jm < ns; jm++)
                for (uint32_t r = 1; r < rs; r++)
                    rt[(size_t)k2*W + ns - 1 + jm*(rs - 1) + r - 1] = tw[(r*((uint64_t)jm*H + k2)*q) % D];
        }
    size_t nu = (size_t)(H/2 + 1)*W;
    std::vector<GE> u2(nu);
    uint64_t inv4 = (GP + 1)/4;                  // 1/4 (p = 3 mod 4)
    for (size_t s = 0; s < nu; s++)
    {
        uint32_t k2 = (uint32_t)(s >> I.lgW), k1 = (uint32_t)(s & (W - 1));
        uint64_t f = k2 + (uint64_t)H*k1;
        if (2*f > D) f = D - f;
        // u = -(1 + t^2)/4 with t^2 = w_D^f
        GE t2 = tw[f % D];
        uint64_t ax = gred((u128)((t2.x + 1) % GP)*inv4), ay = gred((u128)t2.y*inv4);
        u2[s] = { ax ? GP - ax : 0, ay ? GP - ay : 0 };
    }
    // device tables: pairs of 64-bit (M61) or 32-bit (M31) residues
    auto upload = [&](const std::vector<GE>& v, const char* what)
    {
        if (!I.f31 && !I.f3)
        {
            cl_mem m = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(GE)*v.size(), (void*)v.data(), &err); check(err, what);
            return m;
        }
        std::vector<uint32_t> h(2*v.size());
        for (size_t i = 0; i < v.size(); i++) { GE m = toM(v[i]); h[2*i] = (uint32_t)m.x; h[2*i + 1] = (uint32_t)m.y; }
        cl_mem m = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*h.size(), h.data(), &err); check(err, what);
        return m;
    };
    I.ntw = upload(tw, "ntw");
    I.nrowtw = upload(rt, "nrowtw");
    I.nu2s = upload(u2, "nu2s");
}

static void ntt_cols(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, int inv, bool half)
{
    size_t np = I.ntt_col.size();
    uint32_t D = I.W*I.H;
    cl_mem in = src;
    for (size_t t = 0; t < np; t++)
    {
        cl_mem out = ((np - t) % 2 == 1) ? dst : tmp;
        auto& p = I.ntt_col[t];
        cl_kernel k = inv ? p.ki : t == 0 ? (half && p.kcz ? p.kcz : p.kc) : p.kf;
        cl_uint total = D/p.R;
        arg(k, 0, in); arg(k, 1, out); arg(k, 2, I.ntw); arg(k, 3, (cl_uint)I.W); arg(k, 4, (cl_uint)I.lgW);
        arg(k, 5, (cl_uint)(I.H/p.R)); arg(k, 6, (cl_uint)p.Ns); arg(k, 7, (cl_uint)(D/(p.Ns*p.R))); arg(k, 8, total);
        arg(k, 9, (cl_uint)(D/p.R));
        I.run_local(k, (total + 255)/256*256, 256);
        in = out;
    }
}

static void ntt_row(Engine::Impl& I, cl_mem in, cl_mem out, int inv)
{
    cl_kernel k = inv ? I.k_nrowi : I.k_nrow;
    arg(k, 0, in); arg(k, 1, out); arg(k, 2, I.nrowtw);
    I.run_local(k, (size_t)I.nrow_groups*I.nrow_threads, I.nrow_threads);
}

// Forward: src is a slot (double digits). Inverse: src holds a spectrum; dst gets the coefficients.
static void ntt_fft(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, int inv, bool half)
{
    if (!inv)
    {
        ntt_cols(I, src, dst, tmp, 0, half && I.halfzero);
        ntt_row(I, dst, dst, 0);
    }
    else if (I.k_ntail)
        ntt_cols(I, src, dst, tmp, 1, false);     // rows were done by row_tail (ntt_product)
    else
    {
        cl_mem mid = I.ntt_col.size() % 2 == 1 ? tmp : dst;
        ntt_row(I, src, mid, 1);
        ntt_cols(I, mid, dst, tmp, 1, false);
    }
}

static void ntt_product(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, cl_mem Bsp, int square, bool half)
{
    if (I.k_ntail)
    {
        // columns, then rows + product + inverse rows (the inverse that follows only does columns)
        ntt_cols(I, src, dst, tmp, 0, half && I.halfzero);
        cl_kernel k = I.k_ntail;
        arg(k, 0, dst); arg(k, 1, dst); arg(k, 2, I.nrowtw); arg(k, 3, square ? dst : Bsp); arg(k, 4, I.nu2s);
        arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)(I.W*I.H)); arg(k, 7, (cl_int)square);
        I.run_local(k, (size_t)(I.H/2)*I.ntail_threads, I.ntail_threads);
        return;
    }
    ntt_fft(I, src, dst, tmp, 0, half);
    cl_uint total = (I.H/2 + 1)*I.W;
    cl_kernel k = I.k_nmul;
    arg(k, 0, dst); arg(k, 1, square ? dst : Bsp); arg(k, 2, I.nu2s); arg(k, 3, (cl_uint)I.W); arg(k, 4, (cl_uint)I.lgW);
    arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)(I.W*I.H)); arg(k, 7, (cl_int)square); arg(k, 8, total);
    I.run_local(k, (total + 63)/64*64, 64);
}

// ---------------------------------------------------------------------------------------------
// Hybrid: fused column passes. One kernel runs the FP32 pass on the first `total` work items and the
// NTT pass on the next `total` (a multiple of 256: no wave runs both), so the two transforms share
// the GPU instead of running one after the other. Source: FP32 v2 + the NTT source with its colliding
// names renamed, both column kernels turned into functions of the work-item index.
static void fused_replace(std::string& t, const std::string& from, const std::string& to, size_t start = 0)
{
    size_t i = t.find(from, start);
    if (i == std::string::npos)
        throw std::runtime_error("fused_source: pattern not found: " + from);
    t.replace(i, from.size(), to);
}

// The FP32 v2 source and the NTT source with its colliding names renamed, the column and tail kernels
// turned into functions (of the work-item index / of the row pair and the local buffer).
static void fused_parts(std::string& f, std::string& n)
{
    static std::string sf, sn;
    if (sf.empty())
    {
        f = v2_source_f32();
        n = ntt_source;
        const char* clash[] = { "mul_mi", "dft2", "dft3", "dft4", "dft8", "dft16", "dft32", "DFT", "dftR", "col_pass", "tail_pair", "row_tail", "row_fwd", "row_inv" };
        for (const char* c : clash)
            n = std::regex_replace(n, std::regex(std::string("\\b") + c + "\\b"), std::string(c) + "_n");
        fused_replace(f, "__kernel void col_pass(", "static inline void col_pass_f(uint id, ");
        fused_replace(n, "__kernel void col_pass_n(", "static inline void col_pass_nn(uint id, ");
        for (std::string* t : { &f, &n })
        {
            size_t a = t->find("static inline void col_pass_");
            fused_replace(*t, "uint id = get_global_id(0);", "", a);
        }
        // Two column passes in one work-group (col_pass2_h): copy A of the pass writes its output to
        // the local buffer (row ob + k*Ns of the work-group's CW columns), copy B reads its input from
        // there. B never converts a slot nor has a zero half: those #ifdef blocks are cut.
        auto lds_copies = [](std::string& t, const std::string& fname, const std::string& T, const std::string& conv)
        {
            std::string head = "static inline void " + fname + "(uint id, ";
            size_t s = t.find(head);
            size_t e = s == std::string::npos ? s : t.find("\n}\n", s);
            if (e == std::string::npos)
                throw std::runtime_error("fused_source: column pass not found");
            e += 3;
            std::string fn = t.substr(s, e - s), a = fn, b = fn;
            fused_replace(a, head, "static inline void " + fname + "A(uint id, ");
            fused_replace(a, "__global " + T + "* out", "__local " + T + "* out");
            fused_replace(a, "out[n1 + W*(ob + k*Ns)]", "out[LIDX(n1, ob + k*Ns)]");
            fused_replace(b, head, "static inline void " + fname + "B(uint id, ");
            fused_replace(b, "__global const " + T + "* in", "__local const " + T + "* in");
            fused_replace(b, "= in[n1 + W*(jb + r*hq)]", "= in[LIDX(n1, jb + r*hq)]");
            fused_replace(b, "#ifdef HALFZ", "#if 0");
            fused_replace(b, "#ifdef " + conv + "\n", "#if 0\n");
            t.insert(e, "#ifdef COLPASS2\n#ifndef LIDX\n#define LIDX(n1, row) ((row)*CW + ((n1) & (CW - 1)))\n#endif\n" + a + "#undef RIN\n" + b + "#undef RIN\n#endif\n");
        };
        lds_copies(f, "col_pass_f", "C", "CONVD");
        lds_copies(n, "col_pass_nn", "G", "CONV");
        // row tails: the work-group's row pair g and the local buffer become parameters
        auto tail = [](std::string& t, const char* name, const char* fname, const char* type)
        {
            std::string sig = std::string("void ") + name + "(";
            size_t s = t.find(sig);
            size_t k = s == std::string::npos ? s : t.rfind("__kernel", s);
            if (k == std::string::npos)
                throw std::runtime_error("fused_source: row tail not found");
            t.replace(k, s + sig.size() - k, std::string("static inline void ") + fname + "(uint g, __local " + type + "* buf, ");
            fused_replace(t, std::string("__local ") + type + " buf[2*LROW];", "", k);
            fused_replace(t, "uint g = get_group_id(0);", "", k);
        };
        tail(f, "row_tail", "row_tail_f", "C");
        tail(n, "row_tail_n", "row_tail_nn", "G");
        sf = f;
        sn = n;
    }
    f = sf;
    n = sn;
}

// Both row tails in one source: each part sees its own stage macros (the macros of the row section
// that both sources define are dropped in between; PADR, LROW and BUF are the same in both).
static std::string fused_tail_source(const std::string& prefix_f, const std::string& prefix_n)
{
    std::string f, n;
    fused_parts(f, n);
    return prefix_f + f + "\n#undef STAGES_F\n#undef STAGES_I\n#undef STAGES_FT\n#undef FSTAGE\n#undef ISTAGE\n#undef BUF\n#undef ROW_KERNEL\n#undef PADR\n#undef LROW\n"
         + prefix_n + n + R"CLC(
#ifdef TAILH
// Work-groups 0..H/2-1: the FP32 tail of row pair g; H/2..H-1: the NTT tail of row pair g - H/2.
// The two element types have the same size (checked on the host), so they share the local buffer.
__kernel __attribute__((reqd_work_group_size(2*RT, 1, 1)))
void row_tail_h(__global const C* in_f, __global C* out_f, __global const C* rtw_f, __global const C* Bsp_f, __global const C* u2s_f,
                __global const C* rtr_f, uint D_f, __global const G* in_n, __global G* out_n, __global const G* rtw_n,
                __global const G* Bsp_n, __global const G* u2s_n, uint D_n, uint H, int square)
{
    __local C buf[2*LROW];
    uint g = get_group_id(0);
    if (g < H/2)
        row_tail_f(g, buf, in_f, out_f, rtw_f, Bsp_f, u2s_f, H, D_f, square, rtr_f);
    else
        row_tail_nn(g - H/2, (__local G*)buf, in_n, out_n, rtw_n, Bsp_n, u2s_n, H, D_n, square);
}
#endif
)CLC";
}

static const std::string& fused_source()
{
    static std::string src;
    if (!src.empty())
        return src;
    std::string f, n;
    fused_parts(f, n);
    src = f + "\n" + n + R"CLC(
#ifdef IBW
#define KIBWP , __global const float2* kw2lo, __global const float2* kw2hi, __global const float2* kwklo, __global const float2* kwkhi, \
              __global const uint2* kn2lo, __global const uint2* kn2hi, __global const uint2* knklo, __global const uint2* knkhi, \
              __global const uint* kfl, __global const uint* kfh, uint kL
#define IBWA_F , kw2lo, kw2hi, kwklo, kwkhi, kfl, kfh, kL
#define IBWA_N , kn2lo, kn2hi, knklo, knkhi, kfl, kfh, kL
#else
#define KIBWP
#define IBWA_F
#define IBWA_N
#endif
#ifdef COLPASS
__kernel void col_pass_h(__global const void* in_f, __global void* out_f, __global const C* tw_f,
                         __global const void* in_n, __global void* out_n, __global const G* tw_n,
                         uint W, uint lgW, uint hq, uint Ns, uint step, uint qR, uint total, uint half_items KIBWP)
{
    uint id = get_global_id(0);
    if (id < half_items)
        col_pass_f(id, (__global const C*)in_f, (__global C*)out_f, tw_f, W, lgW, hq, Ns, step, qR, total, tw_f IBWA_F);
    else
        col_pass_nn(id - half_items, (__global const G*)in_n, (__global G*)out_n, tw_n, W, lgW, hq, Ns, step, total, qR IBWA_N);
}
#endif
#ifdef COLPASS2
// Both column passes (H = R*R: first Ns = 1, then Ns = R) in one launch. A work-group owns CW
// columns and all hq = H/R butterflies of each (CW*hq = 256 work items), so the second pass needs
// only the first pass's results of its own columns: they stay in local memory (H*CW elements).
// Work-groups 0..half_groups-1: FP32, the rest: NTT (the element types have the same size).
// The local index LIDX(column, row) is defined with the pass copies.
__kernel __attribute__((reqd_work_group_size(CW*R, 1, 1)))
void col_pass2_h(__global const void* in_f, __global void* out_f, __global const C* tw_f,
                 __global const void* in_n, __global void* out_n, __global const G* tw_n,
                 uint W, uint lgW, uint hq, uint stepA, uint stepB, uint qR, uint total, uint half_groups KIBWP)
{
    __local C lbuf[CW*R*R];
    uint g = get_group_id(0), l = get_local_id(0);
    bool fp = g < half_groups;
    if (!fp)
        g -= half_groups;
    uint n1 = g*CW + (l & (CW - 1)), jb = l/CW;
    uint id = n1 + (jb << lgW);
    if (fp)
        col_pass_fA(id, (__global const C*)in_f, lbuf, tw_f, W, lgW, hq, 1, stepA, qR, total, tw_f IBWA_F);
    else
        col_pass_nnA(id, (__global const G*)in_n, (__local G*)lbuf, tw_n, W, lgW, hq, 1, stepA, total, qR IBWA_N);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (fp)
        col_pass_fB(id, lbuf, (__global C*)out_f, tw_f, W, lgW, hq, R, stepB, qR, total, tw_f IBWA_F);
    else
        col_pass_nnB(id, (__local const G*)lbuf, (__global G*)out_n, tw_n, W, lgW, hq, R, stepB, total, qR IBWA_N);
}
#endif
)CLC";
    return src;
}

static cl_kernel build_fused(Engine::Impl& I, const std::string& opts, const std::string& source = fused_source(), const char* name = "col_pass_h")
{
    cl_int err;
    const char* srcs[1] = { source.c_str() };
    cl_program prog = clCreateProgramWithSource(I.ctx, 1, srcs, nullptr, &err);
    check(err, "clCreateProgramWithSource(fused)");
    err = clBuildProgram(prog, 1, &I.device, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len = 0;
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, I.device, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        throw std::runtime_error("OpenCL build failed (fused " + opts + "):\n" + log);
    }
    cl_kernel k = clCreateKernel(prog, name, &err);
    check(err, name);
    I.lds_progs.push_back(prog);
    I.lds_kernels.push_back(k);
    return k;
}

// Fused plan when both inner transforms use the same passes and both have fused tails. Default for
// all lengths: P3 (3*2^k) measured 1.5-5% faster; M31 (2^k) was once 1.6% slower, but on the current
// kernels it halves the column-pass launches (8 -> 4 per squaring) and measured 13-15% faster on an
// RX 7900 XTX (gfx1100) at 131072-524288 reals, results identical. PRST_GPU_HFUSE=0/1 overrides.
static void build_hybrid_fused(Engine::Impl& I)
{
    const char* e = getenv("PRST_GPU_HFUSE");
    bool want = e != nullptr ? e[0] != '0' : true;
    if (!want)
        return;
    auto& a = I.i32->col_plan;
    auto& b = I.i31->ntt_col;
    if (a.empty() || a.size() != b.size() || !I.i32->k_tail || !I.i31->k_ntail)
        return;
    for (size_t t = 0; t < a.size(); t++)
        if (a[t].R != b[t].R || a[t].F != nullptr)
            return;
    std::map<std::string, cl_kernel> cache;
    auto variant = [&](const std::string& o)
    {
        auto it = cache.find(o);
        return it != cache.end() ? it->second : (cache[o] = build_fused(I, o + I.i31->nconsts));
    };
    uint32_t H = I.i32->H;
    for (size_t t = 0; t < a.size(); t++)
    {
        uint32_t R = a[t].R, R3 = R % 3 == 0 ? 3 : 1, R2 = R/R3;
        std::string o = "-DCOLPASS -DR2=" + std::to_string(R2) + " -DR3=" + std::to_string(R3);
        cl_kernel kf = variant(o + " -DINV=0"), ki = variant(o + " -DINV=1");
        std::string sl = I.islot ? " -DCONVD -DSLOTT=int2 -DCONV -DSLOTN=int2" : " -DCONVD -DSLOTT=double2 -DCONV";
        cl_kernel kc = t == 0 ? variant(o + " -DINV=0" + sl) : nullptr;
        if (t == 0 && I.ibdwt)
            I.kcib = variant(o + " -DINV=0" + sl + " -DIBW");
        cl_kernel kcz = t == 0 && R % 2 == 0 && H % 2 == 0 ? variant(o + " -DINV=0" + sl + " -DHALFZ") : nullptr;
        I.hcol.push_back({ kf, ki, kc, kcz, R, a[t].Ns });
    }
    // Two passes of the same radix R (H = R*R): both in one launch, the intermediate in local memory
    // (cw columns per work-group, cw*R work items, 8*R bytes of local memory each). RX 7900 XTX:
    // 1024x64 (R = 8): 0.047 -> 0.041 ms per squaring. 1024x256 (R = 16): no gain, the saved launch
    // (an extra pass each way costs 0.019 ms of 0.078) is lost to the lower occupancy: 32 columns
    // 0.073-0.077 ms, 16 columns 0.082-0.088, the separate passes 0.073-0.075. Default for R <= 8,
    // PRST_GPU_COL2=0/1 overrides.
    if (I.hcol.size() != 2)
        return;
    uint32_t R = I.hcol[0].R, hq = H/R;
    if (I.hcol[1].R != R || R*R != H || I.hcol[0].Ns != 1 || I.hcol[1].Ns != R || R % 3 == 0)
        return;
    const char* e2 = getenv("PRST_GPU_COL2");
    bool want2 = e2 != nullptr ? e2[0] == '1' : R <= 8;
    if (!want2)
        return;
    std::string sl = I.islot ? " -DCONVD -DSLOTT=int2 -DCONV -DSLOTN=int2" : " -DCONVD -DSLOTT=double2 -DCONV";
    auto build2 = [&](const std::string& opts) { return build_fused(I, opts + I.i31->nconsts, fused_source(), "col_pass2_h"); };
    try
    {
        // the widest block of columns that fits (wider: longer contiguous global accesses); the
        // device limit on work-group size is lower than what a kernel with a required size allows,
        // so ask the built kernel
        for (uint32_t cw = 64; cw*hq >= 64; cw /= 2)
        {
            if ((cl_ulong)cw*H*8 > local_mem(I) || I.W % cw != 0)
                continue;
            std::string o = "-DCOLPASS -DCOLPASS2 -DR2=" + std::to_string(R) + " -DR3=1 -DCW=" + std::to_string(cw);
            cl_kernel kc = build2(o + " -DINV=0" + sl);
            size_t wg = 0;
            if (clGetKernelWorkGroupInfo(kc, I.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr) != CL_SUCCESS || wg < cw*hq)
                continue;
            cl_kernel ki = build2(o + " -DINV=1");
            cl_kernel kz = I.hcol[0].kcz ? build2(o + " -DINV=0" + sl + " -DHALFZ") : nullptr;
            I.kcol2f = I.mmgw || I.ibdwt ? build2(o + " -DINV=0") : nullptr;
            I.kcol2ib = I.ibdwt ? build2(o + " -DINV=0" + sl + " -DIBW") : nullptr;
            I.kcol2c = kc; I.kcol2z = kz; I.kcol2i = ki;
            I.col2_cw = cw;
            break;
        }
    }
    catch (const std::exception& ex)
    {
        // optional: the two separate passes otherwise
        I.kcol2c = I.kcol2z = I.kcol2i = I.kcol2f = I.kcol2ib = nullptr;
        if (e2 != nullptr)
            fprintf(stderr, "%s\n", ex.what());
    }
}

// Both row tails (FP32 and NTT) in one launch. Only with the M31 field (powers of two): its elements
// are 8 bytes like the FP32 ones, so the local buffer and the padded or swizzled row layout are the
// same. Opt-in (PRST_GPU_HTAIL=1): on an RX 7900 XTX the two separate tails already overlap, the
// fused one measured the same within 2% (131072-524288 reals), unlike the fused column passes.
static void build_hybrid_tail(Engine::Impl& I)
{
    const char* e = getenv("PRST_GPU_HTAIL");
    if (!(e != nullptr && e[0] == '1'))
        return;
    Engine::Impl& a = *I.i32;
    Engine::Impl& b = *I.i31;
    if (!a.k_tail || !b.k_ntail || !b.f31 || a.tail_threads != b.ntail_threads || a.W != b.W || a.H != b.H)
        return;
    bool swz_a = a.tail_opts.find("-DSWZ") != std::string::npos, swz_b = b.tail_opts.find("-DSWZ") != std::string::npos;
    if (swz_a != swz_b)
        return;
    I.k_htail = build_fused(I, a.tail_opts + " " + b.tail_opts + " -DTAILH", fused_tail_source(a.tail_prefix, b.tail_prefix), "row_tail_h");
}

static void hybrid_tail(Engine::Impl& I, cl_mem d32, cl_mem d31, uint32_t D, cl_mem b32, cl_mem b31, int square)
{
    Engine::Impl& a = *I.i32;
    Engine::Impl& b = *I.i31;
    cl_kernel k = I.k_htail;
    arg(k, 0, d32); arg(k, 1, d32); arg(k, 2, a.rowtw); arg(k, 3, square ? d32 : b32); arg(k, 4, a.tw2s);
    arg(k, 5, a.rowtr ? a.rowtr : a.rowtw); arg(k, 6, (cl_uint)D);
    arg(k, 7, d31); arg(k, 8, d31); arg(k, 9, b.nrowtw); arg(k, 10, square ? d31 : b31); arg(k, 11, b.nu2s);
    arg(k, 12, (cl_uint)(b.W*b.H)); arg(k, 13, (cl_uint)a.H); arg(k, 14, (cl_int)square);
    I.run_local(k, (size_t)a.H*a.tail_threads, a.tail_threads);
}

// Fused column passes of both transforms: src32/src31 (the same slot for a forward transform),
// results in the halves of dst.
static void h32_cols(Engine::Impl& I, cl_mem src32, cl_mem src31, cl_mem dst, cl_mem tmp, uint32_t D, int inv, bool half)
{
    auto& d = I.halves.at(dst);
    auto& tm = I.halves.at(tmp);
    size_t np = I.hcol.size();
    cl_mem in32 = src32, in31 = src31;
    if (I.kcol2c && np == 2)
    {
        // both passes in one launch, straight into dst
        cl_kernel k = inv ? I.kcol2i : (half && I.kcol2z ? I.kcol2z : I.kcol2c);
        uint32_t R = I.hcol[0].R;
        cl_uint total = D/R, groups = I.W/I.col2_cw;
        arg(k, 0, in32); arg(k, 1, d.first); arg(k, 2, I.i32->tw); arg(k, 3, in31); arg(k, 4, d.second); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/R)); arg(k, 9, (cl_uint)(D/R));
        arg(k, 10, (cl_uint)(D/(R*R))); arg(k, 11, (cl_uint)(D/R)); arg(k, 12, total); arg(k, 13, groups);
        I.run_local(k, 2*(size_t)groups*I.col2_cw*R, I.col2_cw*R);
        return;
    }
    for (size_t t = 0; t < np; t++)
    {
        bool last = (np - t) % 2 == 1;
        cl_mem o32 = last ? d.first : tm.first, o31 = last ? d.second : tm.second;
        auto& p = I.hcol[t];
        cl_kernel k = inv ? p.ki : t == 0 ? (half && p.kcz ? p.kcz : p.kc) : p.kf;
        cl_uint total = D/p.R, hi = (total + 255)/256*256;
        arg(k, 0, in32); arg(k, 1, o32); arg(k, 2, I.i32->tw); arg(k, 3, in31); arg(k, 4, o31); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/p.R)); arg(k, 9, (cl_uint)p.Ns);
        arg(k, 10, (cl_uint)(D/(p.Ns*p.R))); arg(k, 11, (cl_uint)(D/p.R)); arg(k, 12, total); arg(k, 13, hi);
        I.run_local(k, 2*(size_t)hi, 256);
        in32 = o32;
        in31 = o31;
    }
}

static void v2_tail(Engine::Impl& I, cl_mem dst, uint32_t D, cl_mem Bsp, int square)
{
    cl_kernel k = I.k_tail;
    arg(k, 0, dst); arg(k, 1, dst); arg(k, 2, I.rowtw); arg(k, 3, square ? dst : Bsp); arg(k, 4, I.tw2s);
    arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)D); arg(k, 7, (cl_int)square); arg(k, 8, I.rowtr ? I.rowtr : I.rowtw);
    I.run_local(k, (size_t)(I.H/2)*I.tail_threads, I.tail_threads);
}

static void ntt_tail(Engine::Impl& I, cl_mem dst, cl_mem Bsp, int square)
{
    cl_kernel k = I.k_ntail;
    arg(k, 0, dst); arg(k, 1, dst); arg(k, 2, I.nrowtw); arg(k, 3, square ? dst : Bsp); arg(k, 4, I.nu2s);
    arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)(I.W*I.H)); arg(k, 7, (cl_int)square);
    I.run_local(k, (size_t)(I.H/2)*I.ntail_threads, I.ntail_threads);
}

// Forward FFT of src into dst followed by the pointwise product (square, or with spectrum Bsp).
// Uses the fused last pass when available, otherwise a separate realmul kernel.
static void fft_product(Engine::Impl& I, cl_mem src, cl_mem dst, cl_mem tmp, uint32_t D, cl_mem Bsp, int square, bool half = false)
{
    if (I.h32)
    {
        auto& d = I.halves.at(dst);
        auto& t = I.halves.at(tmp);
        cl_mem b32 = Bsp ? I.halves.at(Bsp).first : nullptr, b31 = Bsp ? I.halves.at(Bsp).second : nullptr;
        if (!I.hcol.empty())
        {
            h32_cols(I, src, src, dst, tmp, D, 0, half && I.halfzero);
            if (I.k_htail)
                hybrid_tail(I, d.first, d.second, D, b32, b31, square);
            else
            {
                v2_tail(*I.i32, d.first, D, b32, square);
                ntt_tail(*I.i31, d.second, b31, square);
            }
            return;
        }
        fft_product(*I.i32, src, d.first, t.first, D, b32, square, half);
        ntt_product(*I.i31, src, d.second, t.second, b31, square, half);
        return;
    }
    if (I.ntt)
    {
        ntt_product(I, src, dst, tmp, Bsp, square, half);
        return;
    }
    if (I.v2 && I.k_tail)
    {
        // columns, then rows + product + inverse rows in one kernel (in place in dst); the inverse
        // FFT that follows (run_fft with inv) only does the column passes
        run_col_passes(I, src, dst, tmp, D, 0, half);
        cl_kernel k = I.k_tail;
        arg(k, 0, dst); arg(k, 1, dst); arg(k, 2, I.rowtw); arg(k, 3, square ? dst : Bsp); arg(k, 4, I.tw2s);
        arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)D); arg(k, 7, (cl_int)square); arg(k, 8, I.rowtr ? I.rowtr : I.rowtw);
        I.run_local(k, (size_t)(I.H/2)*I.tail_threads, I.tail_threads);
        return;
    }
    if (I.v2)
    {
        run_fft(I, src, dst, tmp, D, 0, half);
        cl_uint total = (I.H/2 + 1)*I.W;
        cl_kernel k = I.k_realmul2;
        arg(k, 0, dst); arg(k, 1, square ? dst : Bsp); arg(k, 2, I.tw2s); arg(k, 3, (cl_uint)I.W); arg(k, 4, (cl_uint)I.lgW);
        arg(k, 5, (cl_uint)I.H); arg(k, 6, (cl_uint)D); arg(k, 7, (cl_int)square); arg(k, 8, total);
        I.run_local(k, (total + 63)/64*64, 64);
        return;
    }
    if (!I.lds_fused.k)
    {
        run_fft(I, src, dst, tmp, D, 0);
        arg(I.k_realmul, 0, dst); arg(I.k_realmul, 1, square ? dst : Bsp); arg(I.k_realmul, 2, I.tw2); arg(I.k_realmul, 3, (cl_uint)D); arg(I.k_realmul, 4, (cl_int)square);
        I.run(I.k_realmul, (D/2 + 64) & ~(size_t)63);
        return;
    }
    size_t np = I.lds_plan.size();
    cl_mem in = src;
    for (size_t t = 0; t < np; t++)
    {
        cl_mem out = ((np - t) % 2 == 1) ? dst : tmp;
        auto& p = t + 1 == np ? I.lds_fused : I.lds_plan[t];
        set_lds_args(I, p, in, out, D, 0);
        if (t + 1 == np)
        {
            arg(p.k, 12, square ? out : Bsp); arg(p.k, 13, I.tw2); arg(p.k, 14, (cl_int)square);
        }
        I.run_local(p.k, (size_t)(D/p.R/p.G)*p.T, p.T);
        in = out;
    }
}

// gran: digits per carry produced by the previous kernel (I.CH for per-chunk carries,
// I.CH*CW for the group carries of carry_d / fold_carry).
// carry_fold: ein (e-level group carries), etop (top digits of e), tcb and tc enter the first pass.
static void carry_pass(Engine::Impl& I, cl_mem buf, uint32_t len, uint32_t w, uint32_t gran, int dbl = 0, uint32_t off = 0, cl_long ek = 0, uint32_t tn = 0,
                       cl_mem ein = nullptr, cl_mem etop = nullptr, cl_long tcb = 0, cl_long tc = 0, int wrap = 0)
{
    (void)w;             // digit parameters come from I.wk / I.B
    cl_uint nch = (len + off + gran - 1)/gran;
    if (ein == nullptr) ein = I.car3;
    if (etop == nullptr) etop = I.car3;
    auto fix_args = [&](cl_kernel k, cl_mem cin, cl_mem cout, int last)
    {
        arg(k, 0, buf); arg(k, 1, cin); arg(k, 2, cout); arg(k, 3, I.wk);
        arg(k, 4, (cl_uint)gran); arg(k, 5, nch); arg(k, 6, I.errflag); arg(k, 7, (cl_int)last);
        arg(k, 8, I.B); arg(k, 9, I.invB); arg(k, 10, (cl_int)dbl); arg(k, 11, (cl_uint)len);
        arg(k, 12, (cl_uint)off); arg(k, 13, ein); arg(k, 14, last ? (cl_long)0 : ek); arg(k, 15, etop);
        arg(k, 16, (cl_uint)(last ? 0 : tn)); arg(k, 17, tcb); arg(k, 18, tc);
        arg(k, k == I.k_fix3 ? 21 : 19, (cl_int)wrap);
    };
    // carry_fix3's last work-group runs the second pass over all chunks: worth it for the group
    // carries of the fused carries (a few thousand chunks), not for chunk carries of whole products
    // (Montgomery, carry_pass with gran = CH: 98304-393216 chunks, 0.12-0.58 ms instead of a launch)
    if (I.k_fix3 && nch <= 8192)
    {
        // first pass car1 -> car2, second pass (last work-group) car2 -> car1, as below
        fix_args(I.k_fix3, I.car1, I.car2, 0);
        arg(I.k_fix3, 19, I.car1); arg(I.k_fix3, 20, I.ticket);
        // 256 work items: the last work-group's second pass over all chunks is then short (RX 7900 XTX:
        // 64 gave no gain at 2^19, 128-256 about 2-8%); less if the device does not allow it
        size_t wg3 = 256;
        auto it = I.max_local.find(I.k_fix3);
        while (it != I.max_local.end() && wg3 > 64 && wg3 > it->second)
            wg3 /= 2;
        I.run_local(I.k_fix3, (nch + wg3 - 1)/wg3*wg3, wg3);
        return;
    }
    for (int last = 0; last < 2; last++)
    {
        fix_args(I.k_fix, last ? I.car2 : I.car1, last ? I.car1 : I.car2, last);
        I.run(I.k_fix, (nch + 63)/64*64);
    }
    // leftovers of the second pass are in car1
    arg(I.k_seq, 0, buf); arg(I.k_seq, 1, I.car1); arg(I.k_seq, 2, I.errflag); arg(I.k_seq, 3, (cl_uint)len);
    arg(I.k_seq, 4, (cl_uint)gran); arg(I.k_seq, 5, I.wk); arg(I.k_seq, 6, I.B); arg(I.k_seq, 7, I.invB); arg(I.k_seq, 8, (cl_int)dbl);
    arg(I.k_seq, 9, (cl_uint)off); arg(I.k_seq, 10, (cl_int)wrap);
    I.run(I.k_seq, 1);
}

static void carry_from_fft(Engine::Impl& I, cl_mem dst, uint32_t D, uint32_t w, uint32_t mulc, int wrap = 0)
{
    (void)w;             // digit parameters come from I.wk / I.B
    uint32_t len = 2*D;
    cl_uint nch = len/I.CH;
    run_fft(I, I.fA, I.fB, I.fT, D, 1);
    if (I.h32)
    {
        // hybrid: FP32 coefficients in the first half of fB, residues in the second (as carry_fold_i);
        // the FP32 product already took 2^k/D, so 1/2^k remains (M31: 1/D, P3: 3/D)
        bool f3 = I.i31->f3;
        cl_kernel k = I.k_carry_h;
        arg(k, 0, I.halves.at(I.fB).first); arg(k, 1, I.halves.at(I.fB).second); arg(k, 2, dst); arg(k, 3, I.car1);
        arg(k, 4, (cl_double)(f3 ? 3.0/D : 1.0/D)); arg(k, 5, I.wk); arg(k, 6, I.maxerr); arg(k, 7, (cl_long)mulc);
        arg(k, 8, I.B); arg(k, 9, I.invB); arg(k, 10, nch); arg(k, 11, (cl_int)f3); arg(k, 12, (cl_uint)I.i31->ndinv);
        arg(k, 13, (cl_uint)P3); arg(k, 14, (cl_uint)I.i31->n3pinv); arg(k, 15, (cl_uint)I.i31->n3C);
        arg(k, 16, I.i31->n3pw ? I.i31->n3pw : I.fB);
        I.run_local(k, (nch + 63)/64*64, 64);
        carry_pass(I, dst, len, w, I.CH*64, 0, 0, 0, 0, nullptr, nullptr, 0, 0, wrap);
        return;
    }
    // round + carry: fB (as 2D doubles) -> dst (long digits), times mulc
    arg(I.k_carry_d, 0, I.fB); arg(I.k_carry_d, 1, dst); arg(I.k_carry_d, 2, I.car1); arg(I.k_carry_d, 3, (cl_double)(1.0/D));
    arg(I.k_carry_d, 4, I.wk); arg(I.k_carry_d, 5, (cl_uint)I.CH); arg(I.k_carry_d, 6, I.maxerr); arg(I.k_carry_d, 7, (cl_long)mulc);
    arg(I.k_carry_d, 8, I.B); arg(I.k_carry_d, 9, I.invB); arg(I.k_carry_d, 10, nch);
    I.run_local(I.k_carry_d, (nch + 63)/64*64, 64);
    carry_pass(I, dst, len, w, I.CH*64);
}

// Significant digits after the first (A0) and second (A1) fold of a product of len digits.
static void fold_lengths(Engine::Impl& I, uint32_t fq, uint32_t len, uint32_t& A0, uint32_t& A1)
{
    double lb = std::log2((double)I.B);
    uint32_t kd = (uint32_t)std::ceil(32/lb) + 3, cd = (uint32_t)std::ceil(21/lb) + 3;
    // rounded to 64 chunks so the carry kernels launch in full work-groups of 64
    uint32_t unit = 64*I.CH;
    auto active = [&](uint32_t x) { x = (x + unit - 1)/unit*unit; return x < len ? x : len; };
    A0 = active(std::max(fq + 1 + kd, len - fq + cd));
    A1 = active(std::max(fq + 1 + kd, A0 - fq + cd));
}

// Fold engines: decide whether the fused reduction (carry_fold) applies, and the digit ranges.
// A value has at most m = q + 1 + kd digits, so a product has 2m + 1, its H part 2m - q, and
// the first fold's result (e) ends a few digits above that: U. PRST_GPU_FUSED=0 disables.
static void setup_fused(Engine::Impl& I, uint32_t D, uint32_t fq, int64_t fbs, int64_t fbmul, uint32_t k, int32_t c)
{
    uint32_t len = 2*D, A0, A1;
    fold_lengths(I, fq, len, A0, A1);
    I.halfzero = A1 <= D;
    const char* env = getenv("PRST_GPU_FUSED");
    if (I.fp64 && env != nullptr && env[0] == '0')
        return;
    double lb = std::log2((double)I.B);
    double cb = std::fabs((double)c)*(double)fbmul;
    uint32_t kd = (uint32_t)std::ceil(32/lb) + 3;
    uint32_t m = fq + 1 + kd;
    uint32_t U = std::max(fq + (uint32_t)std::ceil(53/lb) + 1, 2*m + 1 - fq) + (uint32_t)std::ceil(52/lb) + 2;
    // group grid: q sits 128 digits below the end of its group, so the top digits q..U-1 stay inside
    // that group and every group boundary below q is far from it. The slot gets q + 128 digits.
    uint32_t off, fA1;
    fold_grid(64*I.CH, fq, off, fA1);
    // exact double arithmetic in the kernel, and no overflow of the combined carry (int64)
    bool ok = ((double)k + cb)*(double)I.B < 0x1p51 && (double)k + cb < 512.0*(double)I.B;
    if (!ok || U - fq > 64 || m > fA1 || fA1 > len || (double)fbs > 0x1p31)
        return;
    cl_int err;
    std::vector<cl_int> zeros(64, 0);
    I.etop = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*zeros.size(), zeros.data(), &err); check(err, "etop");
    I.fused = true;
    I.fU = U;
    I.fA1 = fA1;
    I.foff = off;
    I.halfzero = fA1 <= D && A1 <= D;
}

// MMGW: column passes of the negacyclic transform (a complex FFT of length D): forward from a complex
// buffer (the weighted input), or inverse; src and dst are (FP32, field) halves, neither in fT.
static void nega_cols(Engine::Impl& I, const std::pair<cl_mem, cl_mem>& src, const std::pair<cl_mem, cl_mem>& dst, int inv)
{
    auto& tm = I.halves.at(I.fT);
    size_t np = I.hcol.size();
    uint32_t D = I.W*I.H;
    if (I.kcol2c && np == 2 && (inv || I.kcol2f))
    {
        cl_kernel k = inv ? I.kcol2i : I.kcol2f;
        uint32_t R = I.hcol[0].R;
        cl_uint total = D/R, groups = I.W/I.col2_cw;
        arg(k, 0, src.first); arg(k, 1, dst.first); arg(k, 2, I.i32->tw); arg(k, 3, src.second); arg(k, 4, dst.second); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/R)); arg(k, 9, (cl_uint)(D/R));
        arg(k, 10, (cl_uint)(D/(R*R))); arg(k, 11, (cl_uint)(D/R)); arg(k, 12, total); arg(k, 13, groups);
        I.run_local(k, 2*(size_t)groups*I.col2_cw*R, I.col2_cw*R);
        return;
    }
    cl_mem in32 = src.first, in31 = src.second;
    for (size_t t = 0; t < np; t++)
    {
        bool last = (np - t) % 2 == 1;
        cl_mem o32 = last ? dst.first : tm.first, o31 = last ? dst.second : tm.second;
        auto& p = I.hcol[t];
        cl_kernel k = inv ? p.ki : p.kf;
        cl_uint total = D/p.R, hi = (total + 255)/256*256;
        arg(k, 0, in32); arg(k, 1, o32); arg(k, 2, I.i32->tw); arg(k, 3, in31); arg(k, 4, o31); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/p.R)); arg(k, 9, (cl_uint)p.Ns);
        arg(k, 10, (cl_uint)(D/(p.Ns*p.R))); arg(k, 11, (cl_uint)(D/p.R)); arg(k, 12, total); arg(k, 13, hi);
        I.run_local(k, 2*(size_t)hi, 256);
        in32 = o32;
        in31 = o31;
    }
}

static cl_uint field_p(Engine::Impl& I) { return I.i31->f3 ? (cl_uint)P3 : 0x7FFFFFFFu; }

// Negacyclic spectrum of a slot (2D digits) into dst: weight, column passes, rows.
static void nega_forward(Engine::Impl& I, cl_mem slot, const std::pair<cl_mem, cl_mem>& dst)
{
    auto& w = I.halves.at(I.fW);
    uint32_t D = I.W*I.H;
    cl_kernel k = I.k_qweight;
    arg(k, 0, slot); arg(k, 1, w.first); arg(k, 2, w.second); arg(k, 3, I.qflo); arg(k, 4, I.qfhi); arg(k, 5, I.qnlo); arg(k, 6, I.qnhi);
    arg(k, 7, (cl_uint)D); arg(k, 8, (cl_int)I.i31->f3); arg(k, 9, field_p(I)); arg(k, 10, (cl_uint)I.i31->n3pinv);
    I.run_local(k, (D + 63)/64*64, 64);
    nega_cols(I, w, dst, 0);
    run_row(*I.i32, dst.first, dst.first, 0);
    ntt_row(*I.i31, dst.second, dst.second, 0);
}

// MMGW buffers and the tables of zeta^j (zeta^D = i, order 4D): lo[t] = zeta^t (t < 1024),
// hi[s] = zeta^(1024*s). FP32: exp(i*pi/(2D)); field: an element of the norm-1 subgroup (its
// inverse is its conjugate), in the form the device multiplies by (P3: times 2^32).
static void setup_mmgw(Engine::Impl& I, uint32_t D)
{
    if (I.hcol.empty() || !I.islot || !I.k_qweight)
        throw std::runtime_error("GPU engine: the cyclic/negacyclic Montgomery engine needs the fused hybrid transform.");
    cl_int err;
    size_t csz = sizeof(double)*2*(size_t)D;
    I.fQ = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fQ");
    I.fW = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fW");
    I.mspec = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, 2*csz, nullptr, &err); check(err, "mspec");
    add_halves(I, I.fQ, csz);
    add_halves(I, I.fW, csz);
    add_halves(I, I.mspec, csz);
    add_halves2(I, I.mspec, csz);
    uint32_t nhi = D/1024 + 1;
    const long double PI = 3.141592653589793238462643383279502884L;
    std::vector<float> flo(2*1024), fhi(2*(size_t)nhi);
    for (uint32_t t = 0; t < 1024; t++)
    {
        long double a = PI*t/(2.0L*D);
        flo[2*t] = (float)cosl(a); flo[2*t + 1] = (float)sinl(a);
    }
    for (uint32_t t = 0; t < nhi; t++)
    {
        long double a = PI*(1024.0L*t)/(2.0L*D);
        fhi[2*t] = (float)cosl(a); fhi[2*t + 1] = (float)sinl(a);
    }
    bool f3 = I.i31->f3;
    GB = 31;
    GMERS = !f3;
    GP = f3 ? P3 : ((uint64_t)1 << 31) - 1;
    GE y = norm1_generator();
    GE z = gpow(y, (GP + 1)/(4*(uint64_t)D));
    GE zd = gpow(z, D);
    if (zd == GE{ 0, GP - 1 })
        z = GE{ z.x, z.y ? GP - z.y : 0 };          // conjugate: zeta^D = i
    else if (!(zd == GE{ 0, 1 }))
        throw std::runtime_error("GPU engine: bad negacyclic root.");
    uint64_t RM = f3 ? ((uint64_t)1 << 32) % GP : 1;
    auto toM = [&](GE v) { return f3 ? GE{ gred((u128)v.x*RM), gred((u128)v.y*RM) } : v; };
    std::vector<uint32_t> nlo(2*1024), nhv(2*(size_t)nhi);
    GE p = { 1, 0 };
    for (uint32_t t = 0; t < 1024; t++, p = gm(p, z))
    {
        GE m = toM(p);
        nlo[2*t] = (uint32_t)m.x; nlo[2*t + 1] = (uint32_t)m.y;
    }
    GE z1024 = gpow(z, 1024);
    p = { 1, 0 };
    for (uint32_t t = 0; t < nhi; t++, p = gm(p, z1024))
    {
        GE m = toM(p);
        nhv[2*t] = (uint32_t)m.x; nhv[2*t + 1] = (uint32_t)m.y;
    }
    I.qflo = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*flo.size(), flo.data(), &err); check(err, "qflo");
    I.qfhi = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*fhi.size(), fhi.data(), &err); check(err, "qfhi");
    I.qnlo = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*nlo.size(), nlo.data(), &err); check(err, "qnlo");
    I.qnhi = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*nhv.size(), nhv.data(), &err); check(err, "qnhi");
}

// x*y mod N (y: both spectra in slot spec, or x*x when spec is null) into dst, times mulc; x may be dst.
void Engine::mmgw_product(void* x_ptr, void* spec_ptr, void* dst_ptr, uint32_t mulc)
{
    Impl& I = *_impl;
    cl_mem x = (cl_mem)x_ptr, spec = (cl_mem)spec_ptr, dst = (cl_mem)dst_ptr;
    uint32_t D = _D;
    auto& hQ = I.halves.at(I.fQ);
    auto& hA = I.halves.at(I.fA);
    nega_forward(I, x, hQ);
    // mod R: t = x*y, u = t*(-N^-1)
    fft_product(I, x, I.fA, I.fT, D, spec, spec ? 0 : 1, false);
    carry_from_fft(I, I.fC, D, _w, 1, 1);
    fft_product(I, I.fC, I.fA, I.fT, D, I.fNinv, 0, false);
    carry_from_fft(I, I.P, D, _w, 1, 1);
    // mod Q: x*y + u*N
    nega_forward(I, I.P, hA);
    bool f3 = I.i31->f3;
    auto& hB2 = spec ? I.halves2.at(spec) : hQ;
    auto& hN = I.halves.at(I.fN);
    cl_kernel k = I.k_qmul;
    arg(k, 0, hA.first); arg(k, 1, hA.second); arg(k, 2, hQ.first); arg(k, 3, hQ.second); arg(k, 4, hB2.first); arg(k, 5, hB2.second);
    arg(k, 6, hN.first); arg(k, 7, hN.second); arg(k, 8, (cl_int)(spec ? 0 : 1)); arg(k, 9, (cl_float)(f3 ? 1.0/3 : 1.0));
    arg(k, 10, (cl_uint)D); arg(k, 11, (cl_int)f3); arg(k, 12, field_p(I)); arg(k, 13, (cl_uint)I.i31->n3pinv);
    I.run_local(k, (D + 63)/64*64, 64);
    // inverse: rows in place, then the column passes into fB
    run_row(*I.i32, hA.first, hA.first, 1);
    ntt_row(*I.i31, hA.second, hA.second, 1);
    auto& hB = I.halves.at(I.fB);
    nega_cols(I, hA, hB, 1);
    // z = -(x*y + u*N)/2 mod Q, times mulc, negacyclic carry
    cl_kernel kc = I.k_qcarry;
    arg(kc, 0, hB.first); arg(kc, 1, hB.second); arg(kc, 2, dst); arg(kc, 3, I.car1); arg(kc, 4, (cl_double)(f3 ? 3.0/D : 1.0/D));
    arg(kc, 5, I.wk); arg(kc, 6, I.maxerr); arg(kc, 7, (cl_long)mulc); arg(kc, 8, (cl_uint)D);
    arg(kc, 9, I.qflo); arg(kc, 10, I.qfhi); arg(kc, 11, I.qnlo); arg(kc, 12, I.qnhi); arg(kc, 13, (cl_int)f3);
    arg(kc, 14, (cl_uint)I.i31->ndinv); arg(kc, 15, field_p(I)); arg(kc, 16, (cl_uint)I.i31->n3pinv); arg(kc, 17, (cl_uint)I.i31->n3C);
    arg(kc, 18, I.i31->n3pw ? I.i31->n3pw : hB.first);
    I.run_local(kc, D/I.CH, 64);
    carry_pass(I, dst, 2*D, _w, I.CH*64, 0, 0, 0, 0, nullptr, nullptr, 0, 0, -1);
}

// ---------------------------------------------------------------------------------------------
// IBDWT for k*2^n - 1 (see ib_carry in the kernel source).
static uint64_t ib_mulm(uint64_t a, uint64_t b, uint64_t p) { return (uint64_t)((unsigned __int128)a*b % p); }
static uint64_t ib_powm(uint64_t a, uint64_t e, uint64_t p)
{
    uint64_t r = 1;
    a %= p;
    for (; e; e >>= 1, a = ib_mulm(a, a, p))
        if (e & 1)
            r = ib_mulm(r, a, p);
    return r;
}

// x in GF(p) with x^L = y, or 0: p = 3 mod 4 and gcd(L, p - 1) = 2. y must be a square; its square
// root in the subgroup of odd order (p - 1)/2 then has the (L/2)-th root s^((L/2)^-1).
static uint64_t ib_root(uint64_t y, uint64_t L, uint64_t p)
{
    uint64_t m2 = (p - 1)/2;
    y %= p;
    if (y == 0 || (L & 1) || p % 4 != 3)
        return 0;
    uint64_t a0 = (L/2) % m2, b0 = m2;
    while (b0) { uint64_t t = a0 % b0; a0 = b0; b0 = t; }
    if (a0 != 1 || ib_powm(y, m2, p) != 1)
        return 0;
    uint64_t s = ib_powm(y, (p + 1)/4, p);
    if (ib_powm(s, m2, p) != 1)
        s = p - s;
    int64_t a = (int64_t)((L/2) % m2), b = (int64_t)m2, x0 = 1, x1 = 0;
    while (b)
    {
        int64_t q = a/b, t = a - q*b;
        a = b; b = t;
        t = x0 - q*x1; x0 = x1; x1 = t;
    }
    uint64_t u = (uint64_t)((x0 % (int64_t)m2 + (int64_t)m2) % (int64_t)m2);
    uint64_t x = ib_powm(s, u, p);
    return ib_powm(x, L, p) == y ? x : 0;
}

// Chooses the length and builds the engine; false (and the engine untouched) if the IBDWT does not
// apply: PRST_GPU_IBDWT=0, no FP64, the hybrid disabled, k too large, or no L-th roots of 2 and 1/k
// in the field of the chosen length (the weights must be in GF(p) for the real transform).
// Digits: beta = n/L bits on average, plus about log2(k)/2 for the output's factor k and 1/2 for
// the products of length L instead of the fold's half-zero ones.
bool Engine::ib_setup(uint32_t k, uint32_t n, int device, int bits_reduce)
{
    const char* e = getenv("PRST_GPU_IBDWT");
    if ((e != nullptr && e[0] == '0') || k >= 4096 || !device_fp64(device) || !h32_requested(true) || ntt_requested(true))
        return false;
    // the shortest length the digits allow, d0; if the field of a length has no roots (M31 and k = 3:
    // 3 is not a square mod 2^31 - 1), a longer one below 2*d0 (the fold's length) still gains
    uint32_t D = 0, d0 = 0;
    uint64_t L = 0, p = 0, theta = 0, kappa = 0;
    for (uint32_t d = 16; d < (1u << 27); d = h32_next_len(d))
    {
        if (d < 4096)
            continue;
        double beta = (double)n/(2.0*d);
        double wreal = max_digit_bits_real(d) + h32_extra_bits() - bits_reduce;
        if (d0 == 0 && !(beta + 0.5 + 0.5*std::log2((double)k) <= wreal && beta <= 23.0))
            continue;
        if (d0 == 0)
            d0 = d;
        if (d >= 2*d0)
            break;
        L = 2*(uint64_t)d;
        p = is_pow2(d) ? 0x7FFFFFFFull : P3;
        theta = ib_root(2, L, p);
        kappa = ib_root(ib_powm(k, p - 2, p), L, p);
        if (theta != 0 && kappa != 0)
        {
            D = d;
            break;
        }
        if (e != nullptr)
            fprintf(stderr, "IBDWT: no %llu-th roots of 2 and 1/%u mod %llu.\n", (unsigned long long)L, k, (unsigned long long)p);
    }
    if (D == 0 || n/L < 4)
        return false;
    try
    {
        Impl& I = *_impl;
        _D = D;
        _ibdwt = I.ibdwt = true;
        I.fp64 = true;
        I.h32 = true;
        I.CH = carry_chunk(true, true);
        _ibq0 = (uint32_t)(n/L);
        _ibr = (uint32_t)(n % L);
        _w = _ibq0 + (_ibr ? 1 : 0);
        _nd = (uint32_t)L;
        _b = 2;
        _e = _w;
        I.wk = _w;
        I.B = (int64_t)1 << _w;
        I.invB = 1.0/I.B;
        init_device(device, bits_reduce);
        if (I.hcol.empty() || !I.islot || !I.k_ibw || I.i31->f3 == is_pow2(D) || I.W*I.H != D || I.CH < 4 || (L/I.CH) % 64 != 0)
            throw std::runtime_error("IBDWT: needs the fused hybrid transform and 32-bit slots.");
        cl_int err;
        size_t csz = sizeof(double)*2*(size_t)D;
        I.fW = clCreateBuffer(I.ctx, CL_MEM_READ_WRITE, csz, nullptr, &err); check(err, "fW");
        add_halves(I, I.fW, csz);
        // f tables
        uint32_t nhi = (uint32_t)(L/1024) + 1;
        uint64_t nm = n % L;
        std::vector<uint32_t> fl(1024), fh(nhi);
        for (uint32_t t = 0; t < 1024; t++)
            fl[t] = (uint32_t)((L - (t*nm) % L) % L);
        for (uint32_t s = 0; s < nhi; s++)
            fh[s] = (uint32_t)((L - (((1024*(uint64_t)s) % L)*nm) % L) % L);
        I.ibfl = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*fl.size(), fl.data(), &err); check(err, "ibfl");
        I.ibfh = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*fh.size(), fh.data(), &err); check(err, "ibfh");
        // weight tables: 0 w2lo, 1 w2hi, 2 wklo, 3 wkhi (FP32), 4..7 the same in the field
        long double lk = std::log2((long double)k);
        std::vector<float> w2lo(2048), wklo(2048), w2hi(2*(size_t)nhi), wkhi(2*(size_t)nhi);
        for (uint32_t t = 0; t < 1024; t++)
        {
            long double a = (long double)t/L;
            w2lo[2*t] = (float)exp2l(a); w2lo[2*t + 1] = (float)exp2l(-a);
            wklo[2*t] = (float)exp2l(-a*lk); wklo[2*t + 1] = (float)exp2l(a*lk);
        }
        for (uint32_t s = 0; s < nhi; s++)
        {
            long double a = 1024.0L*s/L;
            w2hi[2*s] = (float)exp2l(a); w2hi[2*s + 1] = (float)(k*exp2l(-a));
            wkhi[2*s] = (float)exp2l(-a*lk); wkhi[2*s + 1] = (float)exp2l(a*lk);
        }
        bool f3 = I.i31->f3;
        uint64_t RM = f3 ? ((uint64_t)1 << 32) % p : 1;
        auto toM = [&](uint64_t v) { return (uint32_t)ib_mulm(v, RM, p); };
        uint64_t thinv = ib_powm(theta, p - 2, p), kainv = ib_powm(kappa, p - 2, p);
        std::vector<uint32_t> n2lo(2048), nklo(2048), n2hi(2*(size_t)nhi), nkhi(2*(size_t)nhi);
        for (uint32_t t = 0; t < 1024; t++)
        {
            n2lo[2*t] = toM(ib_powm(theta, t, p)); n2lo[2*t + 1] = toM(ib_powm(thinv, t, p));
            nklo[2*t] = toM(ib_powm(kappa, t, p)); nklo[2*t + 1] = toM(ib_powm(kainv, t, p));
        }
        for (uint32_t s = 0; s < nhi; s++)
        {
            uint64_t ex = 1024*(uint64_t)s;
            n2hi[2*s] = toM(ib_powm(theta, ex, p)); n2hi[2*s + 1] = toM(ib_mulm(k, ib_powm(thinv, ex, p), p));
            nkhi[2*s] = toM(ib_powm(kappa, ex, p)); nkhi[2*s + 1] = toM(ib_powm(kainv, ex, p));
        }
        const void* data[8] = { w2lo.data(), w2hi.data(), wklo.data(), wkhi.data(), n2lo.data(), n2hi.data(), nklo.data(), nkhi.data() };
        size_t bytes[8] = { 4*w2lo.size(), 4*w2hi.size(), 4*wklo.size(), 4*wkhi.size(), 4*n2lo.size(), 4*n2hi.size(), 4*nklo.size(), 4*nkhi.size() };
        for (int t = 0; t < 8; t++)
        {
            I.ibt[t] = clCreateBuffer(I.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes[t], const_cast<void*>(data[t]), &err);
            check(err, "ibt");
        }
        finish();
        return true;
    }
    catch (const std::exception& ex)
    {
        if (e != nullptr)
            fprintf(stderr, "%s\n", ex.what());
        _impl.reset(new Impl());
        _ibdwt = false;
        _D = 0;
        _w = 0;
        return false;
    }
}

// Digit j of the IBDWT layout starts at bit e_j = ceil(j*n/L).
static inline uint64_t ib_pos(uint64_t j, uint64_t n, uint64_t L) { return (j*n + L - 1)/L; }

// Value (< 2*k*2^n) -> balanced IBDWT digits; the top digit has base k*2^width, its carry wraps.
void Engine::ib_to_digits(const uint32_t* data, int size, std::vector<double>& digits) const
{
    uint64_t L = 2*(uint64_t)_D;
    digits.assign(L, 0.0);
    uint64_t total_bits = (uint64_t)size*32;
    auto bits = [&](uint64_t pos, uint32_t width)
    {
        unsigned __int128 win = 0;
        uint64_t limb = pos/32;
        for (int t = 0; t < 4; t++)
            if (limb + t < (uint64_t)size)
                win |= (unsigned __int128)data[limb + t] << (32*t);
        win >>= pos % 32;
        return (int64_t)(uint64_t)(win & ((((unsigned __int128)1) << width) - 1));
    };
    int64_t carry = 0;
    for (uint64_t j = 0; j < L; j++)
    {
        uint64_t pos = ib_pos(j, _n, L);
        uint32_t b = (uint32_t)(ib_pos(j + 1, _n, L) - pos);
        bool top = j == L - 1;
        int64_t v = pos < total_bits ? bits(pos, top ? std::min<uint32_t>(b + 40, 90) : b) : 0;
        v += carry;
        carry = 0;
        if (!top)
        {
            if (v >= ((int64_t)1 << (b - 1))) { v -= (int64_t)1 << b; carry = 1; }
        }
        else
        {
            int64_t base = (int64_t)_k << b;
            while (v >= base/2) { v -= base; carry++; }
        }
        digits[j] = (double)v;
    }
    digits[0] += (double)carry;
}

// IBDWT digits -> sign and magnitude of sum d_j*2^e_j (not reduced).
static bool ib_magnitude(const std::vector<double>& digits, uint64_t n, uint64_t L, std::vector<uint32_t>& magnitude)
{
    size_t limbs = (size_t)(n/32) + 4;
    std::vector<uint32_t> pos(limbs, 0), neg(limbs, 0);
    for (uint64_t i = 0; i < L && i < digits.size(); i++)
    {
        int64_t d = (int64_t)digits[i];
        if (d == 0)
            continue;
        std::vector<uint32_t>& acc = d > 0 ? pos : neg;
        uint64_t bit = ib_pos(i, n, L);
        size_t j = bit/32;
        unsigned __int128 v = (unsigned __int128)(uint64_t)(d > 0 ? d : -d) << (bit % 32);
        uint64_t carry = 0;
        for (; j < limbs && (v != 0 || carry != 0); j++)
        {
            uint64_t s = (uint64_t)acc[j] + (uint32_t)v + carry;
            acc[j] = (uint32_t)s;
            carry = s >> 32;
            v >>= 32;
        }
    }
    int cmp = 0;
    for (size_t j = limbs; j-- > 0 && cmp == 0; )
        if (pos[j] != neg[j])
            cmp = pos[j] > neg[j] ? 1 : -1;
    bool negative = cmp < 0;
    std::vector<uint32_t>& big = negative ? neg : pos;
    std::vector<uint32_t>& small = negative ? pos : neg;
    int64_t borrow = 0;
    for (size_t j = 0; j < limbs; j++)
    {
        int64_t s = (int64_t)big[j] - small[j] - borrow;
        borrow = s < 0;
        big[j] = (uint32_t)(s + (borrow << 32));
    }
    while (!big.empty() && big.back() == 0)
        big.pop_back();
    magnitude.swap(big);
    return negative;
}

static void ib_weight(Engine::Impl& I, cl_mem slot)
{
    auto& w = I.halves.at(I.fW);
    uint32_t L = 2*I.W*I.H;
    cl_kernel k = I.k_ibw;
    arg(k, 0, slot); arg(k, 1, w.first); arg(k, 2, w.second);
    for (int t = 0; t < 8; t++)
        arg(k, 3 + t, I.ibt[t]);
    arg(k, 11, I.ibfl); arg(k, 12, I.ibfh); arg(k, 13, (cl_uint)L); arg(k, 14, (cl_int)I.i31->f3); arg(k, 15, field_p(I));
    arg(k, 16, (cl_uint)I.i31->n3pinv);
    I.run_local(k, (L + 63)/64*64, 64);
}

// Forward column passes of a slot's weighted digits into dst: the weights applied in the first pass
// (IBW kernels), else ib_weight into fW and the passes from there (also with PRST_GPU_IBW0).
static void ib_cols(Engine::Impl& I, cl_mem slot, const std::pair<cl_mem, cl_mem>& dst)
{
    size_t np = I.hcol.size();
    uint32_t D = I.W*I.H;
    static bool off = getenv("PRST_GPU_IBW0") != nullptr;
    auto ibargs = [&](cl_kernel k)
    {
        for (int t = 0; t < 8; t++)
            arg(k, 14 + t, I.ibt[t]);
        arg(k, 22, I.ibfl); arg(k, 23, I.ibfh); arg(k, 24, (cl_uint)(2*D));
    };
    if (I.kcol2ib && np == 2 && !off)
    {
        cl_kernel k = I.kcol2ib;
        uint32_t R = I.hcol[0].R;
        cl_uint total = D/R, groups = I.W/I.col2_cw;
        arg(k, 0, slot); arg(k, 1, dst.first); arg(k, 2, I.i32->tw); arg(k, 3, slot); arg(k, 4, dst.second); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/R)); arg(k, 9, (cl_uint)(D/R));
        arg(k, 10, (cl_uint)(D/(R*R))); arg(k, 11, (cl_uint)(D/R)); arg(k, 12, total); arg(k, 13, groups);
        ibargs(k);
        I.run_local(k, 2*(size_t)groups*I.col2_cw*R, I.col2_cw*R);
        return;
    }
    if (!I.kcib || off)
    {
        ib_weight(I, slot);
        nega_cols(I, I.halves.at(I.fW), dst, 0);
        return;
    }
    auto& tm = I.halves.at(I.fT);
    cl_mem in32 = slot, in31 = slot;
    for (size_t t = 0; t < np; t++)
    {
        bool last = (np - t) % 2 == 1;
        cl_mem o32 = last ? dst.first : tm.first, o31 = last ? dst.second : tm.second;
        auto& p = I.hcol[t];
        cl_kernel k = t == 0 ? I.kcib : p.kf;
        cl_uint total = D/p.R, hi = (total + 255)/256*256;
        arg(k, 0, in32); arg(k, 1, o32); arg(k, 2, I.i32->tw); arg(k, 3, in31); arg(k, 4, o31); arg(k, 5, I.i31->ntw);
        arg(k, 6, (cl_uint)I.W); arg(k, 7, (cl_uint)I.lgW); arg(k, 8, (cl_uint)(I.H/p.R)); arg(k, 9, (cl_uint)p.Ns);
        arg(k, 10, (cl_uint)(D/(p.Ns*p.R))); arg(k, 11, (cl_uint)(D/p.R)); arg(k, 12, total); arg(k, 13, hi);
        if (t == 0)
            ibargs(k);
        I.run_local(k, 2*(size_t)hi, 256);
        in32 = o32;
        in31 = o31;
    }
}

// Spectrum of a slot's weighted digits into dst (columns, rows: as transform()).
static void ib_forward(Engine::Impl& I, cl_mem slot, const std::pair<cl_mem, cl_mem>& dst)
{
    ib_cols(I, slot, dst);
    run_row(*I.i32, dst.first, dst.first, 0);
    ntt_row(*I.i31, dst.second, dst.second, 0);
}

// dst = k*x*y (y: spectrum in slot spec, or x*x when spec is null) times mulc; x may be dst.
void Engine::ib_product(void* x_ptr, void* spec_ptr, void* dst_ptr, uint32_t mulc)
{
    Impl& I = *_impl;
    cl_mem x = (cl_mem)x_ptr, spec = (cl_mem)spec_ptr, dst = (cl_mem)dst_ptr;
    uint32_t D = _D, L = 2*D;
    auto& hA = I.halves.at(I.fA);
    ib_cols(I, x, hA);
    int square = spec ? 0 : 1;
    cl_mem b32 = spec ? I.halves.at(spec).first : nullptr, b31 = spec ? I.halves.at(spec).second : nullptr;
    if (I.k_htail)
        hybrid_tail(I, hA.first, hA.second, D, b32, b31, square);
    else
    {
        v2_tail(*I.i32, hA.first, D, b32, square);
        ntt_tail(*I.i31, hA.second, b31, square);
    }
    run_fft(I, I.fA, I.fB, I.fT, D, 1);
    bool f3 = I.i31->f3;
    auto& hB = I.halves.at(I.fB);
    cl_kernel k = I.k_ibc;
    arg(k, 0, hB.first); arg(k, 1, hB.second); arg(k, 2, dst); arg(k, 3, I.car1); arg(k, 4, (cl_double)(f3 ? 3.0/D : 1.0/D));
    arg(k, 5, I.maxerr); arg(k, 6, (cl_long)mulc); arg(k, 7, (cl_uint)L); arg(k, 8, (cl_uint)_ibq0); arg(k, 9, (cl_uint)_ibr);
    arg(k, 10, (cl_long)_k);
    for (int t = 0; t < 8; t++)
        arg(k, 11 + t, I.ibt[t]);
    arg(k, 19, I.ibfl); arg(k, 20, I.ibfh); arg(k, 21, (cl_int)f3); arg(k, 22, (cl_uint)I.i31->ndinv); arg(k, 23, field_p(I));
    arg(k, 24, (cl_uint)I.i31->n3pinv); arg(k, 25, (cl_uint)I.i31->n3C); arg(k, 26, I.i31->n3pw ? I.i31->n3pw : hB.first);
    cl_uint nch = L/I.CH, groups = nch/64;
    I.run_local(k, nch, 64);
    cl_kernel k2 = I.k_ibc2;
    arg(k2, 0, dst); arg(k2, 1, I.car1); arg(k2, 2, I.errflag); arg(k2, 3, (cl_uint)L); arg(k2, 4, (cl_uint)_ibq0);
    arg(k2, 5, (cl_uint)_ibr); arg(k2, 6, I.ibfl); arg(k2, 7, I.ibfh); arg(k2, 8, (cl_uint)(64*I.CH));
    I.run_local(k2, (groups + 63)/64*64, 64);
}

// The product is in fA (frequency domain, after realmul): reduce it into slot dst.
void Engine::finish_product(void* dst_ptr, uint32_t mulc)
{
    Impl& I = *_impl;
    cl_mem dst = (cl_mem)dst_ptr;
    uint32_t len = 2*_D;
    cl_uint nch = len/I.CH;
    // hybrid: the carries of the first stage are about D*B/4*mulc, and must stay exact in double
    bool h32m = I.h32 && (double)_D*(double)I.B/4*mulc < 0x1p51;
    if ((I.ntt && mulc > 16) || (I.h32 && !h32m))
        throw std::runtime_error("GPU NTT engine: multiplier too large.");
    if (I.fused && (mulc <= 16 || h32m))
    {
        run_fft(I, I.fA, I.fB, I.fT, _D, 1);
        cl_kernel k = I.k_cfold;
        double cb = std::fabs((double)_c)*(double)_fbmul;
        // k*L - c*H: the kernel computes kd*dL - cb*dH, so the sign of c goes into cb
        double cbs = _c < 0 ? -cb : cb;
        uint32_t groups = (I.fA1 + I.foff)/(64*I.CH);
        if (I.h32 && icarry_requested(I.fp64))
        {
            // integer / single precision carry (no FP64): carry_fold_i
            cl_kernel ki = I.k_cfoldi;
            bool f3 = I.i31->f3;
            arg(ki, 0, I.halves[I.fB].first); arg(ki, 1, I.halves[I.fB].second); arg(ki, 2, dst); arg(ki, 3, I.car1); arg(ki, 4, I.car3);
            arg(ki, 5, I.etop); arg(ki, 6, I.maxerr); arg(ki, 7, I.errflag);
            arg(ki, 8, (cl_float)(f3 ? 3.0/_D : 1.0/_D)); arg(ki, 9, (cl_int)mulc); arg(ki, 10, (cl_int)_fq); arg(ki, 11, (cl_int)len);
            arg(ki, 12, (cl_long)_k); arg(ki, 13, (cl_long)cbs); arg(ki, 14, (cl_int)I.B); arg(ki, 15, (cl_float)(1.0/(double)I.B));
            arg(ki, 16, (cl_int)_fbs); arg(ki, 17, (cl_int)I.fU); arg(ki, 18, (cl_int)I.foff); arg(ki, 19, (cl_int)(f3 ? 3 : 2));
            arg(ki, 20, (cl_uint)I.i31->ndinv); arg(ki, 21, (cl_uint)P3); arg(ki, 22, (cl_uint)I.i31->n3pinv); arg(ki, 23, (cl_uint)I.i31->n3C);
            arg(ki, 24, I.i31->n3pw ? I.i31->n3pw : I.fB); arg(ki, 25, (cl_int)I.islot);
            I.run_local(ki, (size_t)groups*64, 64);
            carry_pass(I, dst, I.fA1, _w, I.CH*64, I.islot ? 0 : 1, I.foff, (cl_long)_k, I.fU - _fq, I.car3, I.etop, (cl_long)cbs, (cl_long)_c);
            return;
        }
        arg(k, 0, I.fB); arg(k, 1, dst); arg(k, 2, I.car1); arg(k, 3, I.car3); arg(k, 4, I.etop); arg(k, 5, I.maxerr); arg(k, 6, I.errflag);
        arg(k, 7, (cl_double)(1.0/_D)); arg(k, 8, (cl_double)mulc); arg(k, 9, (cl_int)_fq); arg(k, 10, (cl_int)len);
        arg(k, 11, (cl_double)_k); arg(k, 12, (cl_double)cbs); arg(k, 13, (cl_double)I.B); arg(k, 14, I.invB);
        arg(k, 15, (cl_double)_fbs); arg(k, 16, (cl_double)(1.0/(double)_fbs)); arg(k, 17, (cl_int)I.fU);
        arg(k, 18, (cl_long)_k); arg(k, 19, (cl_long)cbs); arg(k, 20, (cl_int)I.foff);
        if (I.h32)
        {
            // FP32 coefficients in the first half of fB, residues mod 2^31 - 1 in the second
            arg(k, 0, I.halves[I.fB].first);
            arg(k, 21, (cl_int)(I.i31->f3 ? 3 : 2)); arg(k, 22, (cl_ulong)I.i31->ndinv); arg(k, 25, I.halves[I.fB].second);
            // FP32 output scale: the FP32 product step already took 2^k/D, so 1/2^k remains (exact)
            if (I.i31->f3)
                arg(k, 7, (cl_double)(3.0/_D));
            arg(k, 26, (cl_uint)P3); arg(k, 27, (cl_uint)I.i31->n3pinv); arg(k, 28, (cl_uint)I.i31->n3C);
            arg(k, 29, I.i31->n3pw ? I.i31->n3pw : I.fB);
            arg(k, 30, (cl_int)I.islot);
        }
        else
        {
            arg(k, 21, (cl_int)I.ntt); arg(k, 22, (cl_ulong)I.ndinv); arg(k, 25, I.fB);
            arg(k, 26, (cl_uint)0); arg(k, 27, (cl_uint)0); arg(k, 28, (cl_uint)0); arg(k, 29, I.fB); arg(k, 30, (cl_int)0);
        }
        arg(k, 23, I.wk); arg(k, 24, I.B);
        I.run_local(k, (size_t)groups*64, 64);
        // first pass of carry_fix: e-level group carries (car3) and the top digits of e (etop)
        carry_pass(I, dst, I.fA1, _w, I.CH*64, I.islot ? 0 : 1, I.foff, (cl_long)_k, I.fU - _fq, I.car3, I.etop, (cl_long)cbs, (cl_long)_c);
        return;
    }
    carry_from_fft(I, I.P, _D, _w, mulc);
    if (!_mont)
    {
        // Only the significant digits are processed: fold 1 output z' = k*L - c*H has at most
        // max(q + digits(k*L), len - q + digits(c*H)) digits, fold 2 output correspondingly fewer.
        // A carry out of the active range is reported as an error, so a wrong bound cannot go unnoticed.
        uint32_t A0, A1;
        fold_lengths(I, _fq, len, A0, A1);
        // fold 1: P -> Q (int64); fold 2: Q -> dst slot (double), so no separate conversion pass.
        // Slot digits beyond A1 stay zero.
        cl_mem ins[2] = { I.P, I.Q }, outs[2] = { I.Q, dst };
        // double-precision fold path needs k*B well below 2^53
        int fold_fast = I.wk == 0 && std::log2((double)_k) + std::log2((double)I.B) < 51.0;
        uint32_t in_len[2] = { len, A0 }, out_len[2] = { A0, A1 };
        for (int f = 0; f < 2; f++)
        {
            arg(I.k_fold, 0, ins[f]); arg(I.k_fold, 1, outs[f]); arg(I.k_fold, 2, I.car1); arg(I.k_fold, 3, (cl_uint)_fq);
            arg(I.k_fold, 4, (cl_long)_fbs); arg(I.k_fold, 5, (cl_long)_fbmul); arg(I.k_fold, 6, (cl_long)_k); arg(I.k_fold, 7, (cl_long)_c);
            arg(I.k_fold, 8, (cl_uint)in_len[f]); arg(I.k_fold, 9, I.wk); arg(I.k_fold, 10, (cl_uint)I.CH); arg(I.k_fold, 11, I.B);
            arg(I.k_fold, 12, I.invB); arg(I.k_fold, 13, (cl_int)f);
            arg(I.k_fold, 14, (cl_int)fold_fast); arg(I.k_fold, 15, (cl_double)(1.0/(double)_fbs)); arg(I.k_fold, 16, (cl_uint)(out_len[f]/I.CH));
            arg(I.k_fold, 17, (cl_int)(I.B < (1 << 23)));   // |a| < B/2 and bs <= B fit mul24
            I.run_local(I.k_fold, (out_len[f]/I.CH + 63)/64*64, 64);
            carry_pass(I, outs[f], out_len[f], _w, I.CH*64, f);
        }
        return;
    }
    // Montgomery: m = (P mod R)*N' mod R
    cl_mem srcs[2] = { I.P, I.Q };
    cl_mem consts[2] = { I.fNinv, I.fN };
    for (int step = 0; step < 2; step++)
    {
        // fC is a slot (doubles, or 32-bit integers in the hybrid)
        cl_kernel kl = I.islot ? I.k_lowpart_i : I.k_lowpart;
        arg(kl, 0, srcs[step]); arg(kl, 1, I.fC); arg(kl, 2, (cl_uint)_q);
        I.run(kl, len);
        fft_product(I, I.fC, I.fA, I.fT, _D, consts[step], 0);
        carry_from_fft(I, I.Q, _D, _w, 1);     // step 0: Q = t*N', step 1: Q = m*N
    }
    // u = (P + m*N) / R
    arg(I.k_addl, 0, I.P); arg(I.k_addl, 1, I.Q); arg(I.k_addl, 2, I.car1); arg(I.k_addl, 3, I.wk); arg(I.k_addl, 4, (cl_uint)I.CH);
    arg(I.k_addl, 5, I.B); arg(I.k_addl, 6, I.invB);
    I.run(I.k_addl, nch);
    carry_pass(I, I.P, len, _w, I.CH);
    cl_kernel ks = I.islot ? I.k_shiftdown_i : I.k_shiftdown;
    arg(ks, 0, I.P); arg(ks, 1, dst); arg(ks, 2, (cl_uint)_q); arg(ks, 3, (cl_uint)len); arg(ks, 4, I.errflag);
    I.run(ks, len);
}

void Engine::square(int slot, int count, uint32_t mulc)
{
    Impl& I = *_impl;
    cl_mem x = I.slots[slot];
    for (int i = 0; i < count; i++)
    {
        if (I.mmgw || I.ibdwt)
        {
            if (I.ibdwt)
                ib_product(x, nullptr, x, mulc);
            else
                mmgw_product(x, nullptr, x, mulc);
            throttle(I);
            I.op_done();
            continue;
        }
        fft_product(I, x, I.fA, I.fT, _D, nullptr, 1, true);
        finish_product(x, mulc);
        throttle(I);
        I.op_done();
    }
}

void Engine::transform(int spec, int src)
{
    Impl& I = *_impl;
    if (spec == src)
        throw std::runtime_error("GPU engine: transform needs distinct slots.");
    if (I.ibdwt)
    {
        ib_forward(I, I.slots[src], I.halves.at(I.slots[spec]));
        return;
    }
    run_fft(I, I.slots[src], I.slots[spec], I.fT, _D, 0, true);
    if (I.mmgw)
        nega_forward(I, I.slots[src], I.halves2.at(I.slots[spec]));
}

void Engine::mul_transformed(int dst, int spec)
{
    Impl& I = *_impl;
    if (I.ibdwt)
        ib_product(I.slots[dst], I.slots[spec], I.slots[dst], 1);
    else if (I.mmgw)
        mmgw_product(I.slots[dst], I.slots[spec], I.slots[dst], 1);
    else
    {
        fft_product(I, I.slots[dst], I.fA, I.fT, _D, I.slots[spec], 0, true);
        finish_product(I.slots[dst], 1);
    }
    throttle(I);
    I.op_done();
}

void Engine::mul(int dst, int src)
{
    Impl& I = *_impl;
    if (I.ibdwt)
    {
        ib_forward(I, I.slots[src], I.halves.at(I.fB));
        ib_product(I.slots[dst], I.fB, I.slots[dst], 1);
    }
    else if (I.mmgw)
    {
        // both spectra of src into mspec, then as mul_transformed
        run_fft(I, I.slots[src], I.mspec, I.fT, _D, 0, true);
        nega_forward(I, I.slots[src], I.halves2.at(I.mspec));
        mmgw_product(I.slots[dst], I.mspec, I.slots[dst], 1);
    }
    else
    {
        run_fft(I, I.slots[src], I.fB, I.fT, _D, 0, true);
        fft_product(I, I.slots[dst], I.fA, I.fT, _D, I.fB, 0, true);
        finish_product(I.slots[dst], 1);
    }
    throttle(I);
    I.op_done();
}

double Engine::max_roundoff(bool reset)
{
    cl_uint v = 0;
    check(clEnqueueReadBuffer(_impl->queue, _impl->maxerr, CL_TRUE, 0, sizeof(v), &v, 0, nullptr, nullptr), "maxerr");
    if (reset)
    {
        cl_uint zero = 0;
        check(clEnqueueWriteBuffer(_impl->queue, _impl->maxerr, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "maxerr reset");
    }
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

bool Engine::carry_error(bool reset)
{
    cl_uint v = 0;
    check(clEnqueueReadBuffer(_impl->queue, _impl->errflag, CL_TRUE, 0, sizeof(v), &v, 0, nullptr, nullptr), "errflag");
    if (reset && v)
    {
        cl_uint zero = 0;
        check(clEnqueueWriteBuffer(_impl->queue, _impl->errflag, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "errflag reset");
    }
    return v != 0;
}

}
