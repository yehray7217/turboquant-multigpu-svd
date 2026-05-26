# Makefile for randomized_svd_multigpu_v3.cu

NVCC ?= nvcc
MPICXX ?= mpicxx

MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_INCLUDES := $(filter -I%,$(MPI_CFLAGS))

NVCC_CFLAGS = -O3 -std=c++17 -arch=sm_70 $(MPI_INCLUDES) -Xcompiler -pthread
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64
LIBS = -L$(CUDA_LIB_DIR) -lcublas -lcusolver -lcudart -ldl

TARGET = randomized_svd_multigpu_v3
SRC = randomized_svd_multigpu_v3.cu
TQ_SRC = ../turboquant/turboquant.cu
OBJS = randomized_svd_multigpu_v3.o turboquant.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) -pthread $(OBJS) -o $(TARGET) $(LIBS)

randomized_svd_multigpu_v3.o: $(SRC) ../turboquant/turboquant.hpp
	$(NVCC) $(NVCC_CFLAGS) -c $(SRC) -o $@

turboquant.o: $(TQ_SRC) ../turboquant/turboquant.hpp
	$(NVCC) $(NVCC_CFLAGS) -c $(TQ_SRC) -o $@

clean:
	rm -f $(TARGET) $(OBJS)
