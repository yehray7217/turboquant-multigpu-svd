# Build the synthetic distributed GNN embedding exchange benchmark.

NVCC ?= nvcc
MPICXX ?= mpicxx

MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_INCLUDES := $(filter -I%,$(MPI_CFLAGS))

NVCC_CFLAGS = -O3 -std=c++17 -arch=sm_70 $(MPI_INCLUDES) -Xcompiler -pthread
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64
LIBS = -L$(CUDA_LIB_DIR) -lcublas -lcudart -ldl

BUILD_DIR = .build
TARGET = $(BUILD_DIR)/graph_embedding_comm_benchmark
MNIST_TARGET = $(BUILD_DIR)/mnist_qjl_feature_benchmark
SRC = graph_embedding_comm_benchmark.cu
MNIST_SRC = mnist_qjl_feature_benchmark.cu
TQ_SRC = ../turboquant/turboquant.cu
OBJS = $(BUILD_DIR)/graph_embedding_comm_benchmark.o $(BUILD_DIR)/turboquant.o
MNIST_OBJS = $(BUILD_DIR)/mnist_qjl_feature_benchmark.o $(BUILD_DIR)/turboquant.o

all: $(TARGET) $(MNIST_TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) -pthread $(OBJS) -o $(TARGET) $(LIBS)

$(MNIST_TARGET): $(MNIST_OBJS)
	$(NVCC) $(NVCC_CFLAGS) $(MNIST_OBJS) -o $(MNIST_TARGET) $(LIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/graph_embedding_comm_benchmark.o: $(SRC) ../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(SRC) -o $@

$(BUILD_DIR)/mnist_qjl_feature_benchmark.o: $(MNIST_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(MNIST_SRC) -o $@

$(BUILD_DIR)/turboquant.o: $(TQ_SRC) ../turboquant/turboquant.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFLAGS) -c $(TQ_SRC) -o $@

clean:
	rm -f $(TARGET) $(MNIST_TARGET) $(OBJS) $(MNIST_OBJS)
