NVCC ?= nvcc

CUDA_ARCH ?= sm_70
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64

BUILD_DIR = .build
TARGET = $(BUILD_DIR)/tq_kernel_microbench
SRC = tq_kernel_microbench.cu
TQ_SRC = ../../turboquant/turboquant.cu
OBJS = $(BUILD_DIR)/tq_kernel_microbench.o $(BUILD_DIR)/turboquant.o

NVCC_CFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)
LIBS = -L$(CUDA_LIB_DIR) -lcublas -lcusolver -lcudart -ldl

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/tq_kernel_microbench.o: $(SRC) ../../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(SRC) -o $@

$(BUILD_DIR)/turboquant.o: $(TQ_SRC) ../../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(TQ_SRC) -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_CFLAGS) $(OBJS) -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)
