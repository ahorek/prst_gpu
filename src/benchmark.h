#pragma once

#include "inputnum.h"

class Options;

// -benchmark: time per squaring on the GPU and/or CPU for common forms (or for input, if set).
int benchmark_main(Options& options, InputNum& input, bool run_gpu, bool run_cpu, double seconds, int min_ops);
