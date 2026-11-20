# prst-gpu

An experimental, vibe-coded fork of https://github.com/patnashev/prst with GPU support (OpenCL).

By default, prst runs on the CPU. To enable GPU acceleration, use the -gpu flag.

## Usage

Run with GPU acceleration:
```
./prst -log info "3*2^1660962+1" -gpu
```
Select a specific GPU (when multiple GPUs are available / without boinc):
```
./prst -log info "3*2^1660962+1" -gpu 0
```

## Benchmarking

You can benchmark performance to compare CPU vs GPU:
```
./prst -benchmark time 10 -gpu -t 16
```
Or benchmark GPU performance only:
```
./prst -benchmark gpu time 10 -gpu
```

Expected performance depends on the algorithm and the GPU/CPU combination. Modern GPUs should deliver performance comparable to, or better than, high-end CPUs. Example
```
CPS
fft 2x720K
Nvidia 5090    6291s
AMD 7900 xtx  18289s
5950x (16t)   25592s

GCW
fft 3584K
Nvidia 5090    7200s
AMD 7900 xtx  19737s
5950x (16t)   51251s
```

## GPU support
It should work on any modern GPU that supports OpenCL. However, performance may be really bad on GPUs older than the AMD RDNA or NVIDIA Turing architectures.
Some tests may require FP64 support

## Contributing
contributions are welcome!

![Developed By an AI Not By Human](badge.png)
