// Quick performance benchmark: time per squaring on the GPU (OpenCL engine) and on the CPU (gwnum)
// for common prime search forms, with an estimate of the full test time.
//
//   prst -benchmark [gpu | cpu] [time <seconds>] [ops <count>] ["<number>"]
//
// Without a number, a list of common forms is measured. -t, -gpu <n>, -fft+1, -cpu apply as usual.

#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include "gwnum.h"
#include "cpuid.h"
#include "arithmetic.h"
#include "exception.h"
#include "inputnum.h"
#include "task.h"
#include "prst.h"
#include "benchmark.h"
#ifdef OPENCL
#include "gpu_exp.h"
#endif

using namespace arithmetic;

// Representative numbers (sizes close to current search fronts).
static const char* benchmark_numbers[][2] = {
    { "412591*2^2799184+1",     "Proth, k*2^n+1" },
    { "27*2^7000000+1",         "Proth, small k, 7M bits" },
    { "3*2^18000000-1",         "Riesel/321, k*2^n-1, 18M bits" },
    { "6679881*2^6679881+1",    "Cullen, n*2^n+1" },
    { "999998^131072+1",        "Generalized Fermat b^2^17+1" },
    { "9998^1048576+1",         "Generalized Fermat b^2^20+1" },
    { "6436*5^4000000+1",       "Sierpinski base 5" },
    { "7851018*13^7851018+1",   "k*b^n+1, base 13" },
    { "5156374*47^5156374+1",   "k*b^n+1, base 47" },
    { "300000!+1",              "factorial (generic modulus)" },
    { "3000000#+1",             "primorial (generic modulus)" },
    { "836745!/(836729#)-1",    "compositorial n!/n#-1 (Montgomery)" },
};

static std::string human_time(double sec)
{
    char buf[64];
    if (sec < 120)
        snprintf(buf, sizeof(buf), "%.0f s", sec);
    else if (sec < 7200)
        snprintf(buf, sizeof(buf), "%.1f min", sec/60);
    else if (sec < 172800)
        snprintf(buf, sizeof(buf), "%.1f h", sec/3600);
    else
        snprintf(buf, sizeof(buf), "%.1f days", sec/86400);
    return buf;
}

static bool cpu_benchmark(InputNum& input, Options& options, double seconds, int min_ops, double& ms_per_op, int& ops, std::string& desc)
{
    GWState gwstate;
    options.configure(gwstate);
    gwstate.information_only = false;
    try
    {
        input.setup(gwstate);
        desc = gwstate.fft_description;
        double elapsed = 0;
        ops = 0;
        {
            GWArithmetic gw(gwstate);
            GWNum X(gw);
            X = Giant::rnd(gwstate.N->bitlen() - 1);
            for (int i = 0; i < 5; i++)
                gw.square(X, X, GWMUL_STARTNEXTFFT);
            auto t0 = std::chrono::steady_clock::now();
            while ((ops < min_ops || elapsed < seconds) && !Task::abort_flag())
            {
                for (int i = 0; i < 5; i++)
                    gw.square(X, X, GWMUL_STARTNEXTFFT);
                ops += 5;
                elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
        }
        gwstate.done();
        ms_per_op = ops > 0 ? 1000*elapsed/ops : 0;
        return ops > 0;
    }
    catch (const std::exception& e)
    {
        desc = e.what();
        gwstate.done();
        return false;
    }
}

int benchmark_main(Options& options, InputNum& input, bool run_gpu, bool run_cpu, double seconds, int min_ops)
{
#ifndef OPENCL
    if (run_gpu && !run_cpu)
    {
        printf("PRST was built without OpenCL support.\n");
        return PRST_EXIT_FAILURE;
    }
    run_gpu = false;
#endif
    int device = options.GpuDevice;
#ifdef OPENCL
    if (run_gpu)
    {
        auto devices = gpu::Engine::list_devices();
        int found = -1;
        // auto: the first GPU with FP64, else the first GPU (hybrid engine only)
        for (int pass = 0; pass < 2 && found < 0; pass++)
            for (auto& d : devices)
                if ((device >= 0 && d.index == device) || (device < 0 && d.is_gpu && (pass == 1 || d.fp64) && found < 0))
                    found = d.index;
        if (found < 0)
        {
            printf("GPU: no OpenCL GPU found%s, GPU part skipped (see -gpu list).\n", device >= 0 ? " with that index" : "");
            run_gpu = false;
        }
        else
        {
            printf("GPU %d: %s [%s]%s\n", devices[found].index, devices[found].name.data(), devices[found].platform.data(), devices[found].fp64 ? "" : " (no FP64: hybrid engine only)");
            device = found;
        }
    }
#endif
    if (run_cpu)
        printf("CPU: %d thread%s\n", options.thread_count, options.thread_count == 1 ? "" : "s");
    printf("Estimated test time = time per op * bits of N (one squaring per bit, plus a few %% for checks).\n\n");

    std::vector<std::pair<std::string, std::string>> list;
    if (!input.empty())
        list.emplace_back(input.input_text(), "");
    else
        for (auto& e : benchmark_numbers)
            list.emplace_back(e[0], e[1]);

    for (auto& item : list)
    {
        if (Task::abort_flag())
            break;
        InputNum num;
        if (!num.parse(item.first))
        {
            printf("%s: parse error.\n", item.first.data());
            continue;
        }
        double bits = num.bitlen();
        printf("%s  (%.2fM bits%s%s)\n", num.display_text().data(), bits/1e6, item.second.empty() ? "" : ", ", item.second.data());
#ifdef OPENCL
        if (run_gpu)
        {
            GPUBenchmarkResult r;
            std::string reason;
            if (gpu_benchmark(num, device, options.next_fft_count, seconds, min_ops, r, reason))
            {
                size_t fft = r.description.find("FFT ");   // drop the device name, printed above
                if (fft != std::string::npos)
                    r.description = r.description.substr(fft);
                printf("  GPU %9.3f ms/op  test ~%-10s %s, max round-off %.3f%s\n", r.ms_per_op, human_time(r.ms_per_op*bits/1000).data(),
                    r.description.data(), r.roundoff, r.carry_error ? ", CARRY ERROR" : "");
            }
            else
                printf("  GPU  not available: %s\n", reason.data());
        }
#endif
        if (run_cpu && !Task::abort_flag())
        {
            double ms = 0;
            int ops = 0;
            std::string desc;
            if (cpu_benchmark(num, options, seconds, min_ops < 10 ? min_ops : 10, ms, ops, desc))
                printf("  CPU %9.3f ms/op  test ~%-10s %s\n", ms, human_time(ms*bits/1000).data(), desc.data());
            else
                printf("  CPU  failed: %s\n", desc.data());
        }
    }
    return PRST_EXIT_NORMAL;
}
