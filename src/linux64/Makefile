# Makefile for Linux

EXE       = prst
LIB_GWNUM = ../../framework/gwnum/linux64/gwnum.a

COMPOBJS_COMMON = benchmark.o md5.o arithmetic.o group.o giant.o lucas.o config.o inputnum.o integer.o logging.o file.o container.o task.o exp.o fermat.o order.o pocklington.o lucasmul.o morrison.o proof.o testing.o support.o batch.o abc_parser.o
COMPOBJS   = $(COMPOBJS_COMMON) prst.o

# OpenCL GPU support (make OPENCL=0 to build without it).
# MSYS2 UCRT64: pacman -S mingw-w64-ucrt-x86_64-opencl-icd mingw-w64-ucrt-x86_64-opencl-headers
#OPENCL ?= 1
#ifeq ($(OPENCL),1)
COMPOBJS_COMMON += ocl_engine.o gpu_exp.o
OPENCL_FLAGS = -DOPENCL
OPENCL_LIBS = -lOpenCL -L../../framework/OpenCL-Headers/lib
#endif

# Source directories

VPATH     = ..:../../framework:../../framework/arithmetic

# insert proper dummy file to get _GWDATA data segment aligned
# on a 32 byte boundary.  Use proper dummyNN.o to do the alignment

# DUMMY2 = ../../framework/gwnum/mingw64/gwnum.ld
DUMMY2 =
LIBS2  = -lm -ldl -static-libstdc++ -lgmp ${OPENCL_LIBS}

#CC  = gcc
#CXX = g++
CFLAGS   = -I.. -I../../framework/gwnum                                                -DGMP -DX86_64 -O2 -std=c99     -Wall -Wextra
CXXFLAGS = -I.. -I../../framework -I../../framework/arithmetic -I../../framework/OpenCL-Headers -DGMP -I../../framework/gwnum -DX86_64 -O2 -std=gnu++17 -Wall -Wextra
# the code is too noisy for all warnings
CXXFLAGS += -Wno-sign-compare -Wno-unused-parameter
CFLAGS   += $(OPENCL_FLAGS)
CXXFLAGS += $(OPENCL_FLAGS)

#########################################################################

all: $(EXE)

$(EXE): $(COMPOBJS)
	$(CXX) -o $@ $(COMPOBJS) $(LIB_GWNUM) $(DUMMY2) $(OPENCL_LIBS) $(LIBS2)

clean:
	rm -f $(EXE) $(COMPOBJS)
