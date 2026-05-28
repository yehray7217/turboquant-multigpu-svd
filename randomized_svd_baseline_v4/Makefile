# Makefile for randomized_svd_multigpu_v4.cu

NVCC ?= nvcc
MPICXX ?= mpicxx

MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_INCLUDES := $(filter -I%,$(MPI_CFLAGS))

NVCC_CFLAGS = -O3 -std=c++17 -arch=sm_70 $(MPI_INCLUDES) -Xcompiler -pthread
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64
LIBS = -L$(CUDA_LIB_DIR) -lcublas -lcusolver -lcudart -ldl

BUILD_DIR = .build
TARGET = $(BUILD_DIR)/randomized_svd_multigpu_v4
SRC = randomized_svd_multigpu_v4.cu
TQ_SRC = ../turboquant/turboquant.cu
OBJS = $(BUILD_DIR)/randomized_svd_multigpu_v4.o $(BUILD_DIR)/turboquant.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) -pthread $(OBJS) -o $(TARGET) $(LIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/randomized_svd_multigpu_v4.o: $(SRC) ../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(SRC) -o $@

$(BUILD_DIR)/turboquant.o: $(TQ_SRC) ../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(TQ_SRC) -o $@

clean:
	rm -f $(TARGET) $(OBJS)
