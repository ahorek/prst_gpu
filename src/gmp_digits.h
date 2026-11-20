#pragma once

// Conversion between GMP integers and balanced base-B digits for the GPU engine
// (digit d_i in [-(B>>1), -(B>>1)+B-1], value = sum d_i*B^i).
// Divide and conquer over precomputed powers B^(2^i): O(M(n) log n), so converting a
// number of tens of millions of bits takes seconds instead of hours.

#include <gmp.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>

namespace gpu
{
    class DigitConverter
    {
    public:
        explicit DigitConverter(int64_t B) : _B(B) { }
        ~DigitConverter()
        {
            for (auto& p : _pw)
                mpz_clear(p.get());
        }
        DigitConverter(const DigitConverter&) = delete;
        DigitConverter& operator=(const DigitConverter&) = delete;

        // x >= 0. Produces balanced digits (may be one digit longer than the plain representation).
        void to_digits(const mpz_t x, std::vector<int64_t>& digits)
        {
            double lb = std::log2((double)_B);
            size_t count = (size_t)(mpz_sizeinbase(x, 2)/lb) + 2;
            size_t cnt = 1;
            while (cnt < count)
                cnt <<= 1;
            powers_upto(cnt);
            digits.assign(cnt + 1, 0);
            to_rec(x, digits.data(), cnt);
            int64_t lo = -(_B >> 1), hi = lo + _B - 1, carry = 0;
            for (size_t i = 0; i < digits.size(); i++)
            {
                int64_t v = digits[i] + carry;
                carry = 0;
                if (v > hi) { v -= _B; carry = 1; }
                digits[i] = v;
            }
            while (!digits.empty() && digits.back() == 0)
                digits.pop_back();
        }

        // Signed digits with |d| < 2^31 -> integer.
        void from_digits(const std::vector<int64_t>& digits, mpz_t res)
        {
            size_t cnt = 1;
            while (cnt < digits.size())
                cnt <<= 1;
            powers_upto(cnt);
            from_rec(digits, 0, cnt, res);
        }

    private:
        struct Free { void operator()(__mpz_struct* p) { delete p; } };
        int64_t _B;
        std::vector<std::unique_ptr<__mpz_struct, Free>> _pw;   // _pw[i] = B^(2^i)

        static int log2_exact(size_t x) { int k = 0; while (((size_t)1 << k) < x) k++; return k; }

        void powers_upto(size_t cnt)
        {
            // need B^(cnt/2)
            int need = log2_exact(cnt);
            while ((int)_pw.size() < need)
            {
                __mpz_struct* p = new __mpz_struct;
                mpz_init(p);
                if (_pw.empty())
                    mpz_set_ui(p, (unsigned long)_B);
                else
                    mpz_mul(p, _pw.back().get(), _pw.back().get());
                _pw.emplace_back(p);
            }
        }

        void to_rec(mpz_srcptr x, int64_t* out, size_t cnt)
        {
            if (mpz_sgn(x) == 0)
                return;
            if (cnt <= 16)
            {
                mpz_t t;
                mpz_init_set(t, x);
                for (size_t i = 0; i < cnt; i++)
                    out[i] = (int64_t)mpz_fdiv_q_ui(t, t, (unsigned long)_B);
                mpz_clear(t);
                return;
            }
            size_t half = cnt >> 1;
            mpz_srcptr p = _pw[log2_exact(half)].get();
            if (mpz_cmp(x, p) < 0)
            {
                to_rec(x, out, half);
                return;
            }
            mpz_t q, r;
            mpz_init(q);
            mpz_init(r);
            mpz_tdiv_qr(q, r, x, p);
            to_rec(r, out, half);
            to_rec(q, out + half, half);
            mpz_clear(q);
            mpz_clear(r);
        }

        void from_rec(const std::vector<int64_t>& d, size_t off, size_t cnt, mpz_ptr res)
        {
            if (off >= d.size())
            {
                mpz_set_ui(res, 0);
                return;
            }
            if (cnt <= 16)
            {
                mpz_set_ui(res, 0);
                size_t end = off + cnt < d.size() ? off + cnt : d.size();
                for (size_t i = end; i-- > off; )
                {
                    mpz_mul_ui(res, res, (unsigned long)_B);
                    int64_t v = d[i];
                    if (v >= 0)
                        mpz_add_ui(res, res, (unsigned long)v);
                    else
                        mpz_sub_ui(res, res, (unsigned long)(-v));
                }
                return;
            }
            size_t half = cnt >> 1;
            mpz_t hi;
            mpz_init(hi);
            from_rec(d, off + half, half, hi);
            from_rec(d, off, half, res);
            if (mpz_sgn(hi) != 0)
            {
                mpz_mul(hi, hi, _pw[log2_exact(half)].get());
                mpz_add(res, res, hi);
            }
            mpz_clear(hi);
        }
    };
}
