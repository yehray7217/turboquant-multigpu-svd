# Makefile for randomized_svd_multigpu_v2.cu
NVCC = nvcc
CFLAGS = -O3 -std=c++17 -arch=sm_70
LIBS = -lcublas -lcusolver -lcudart

TARGET = randomized_svd_multigpu_v2
SRC = randomized_svd_multigpu_v2.cu
TQ_SRC = ../turboquant/turboquant.cu

all: $(TARGET)

$(TARGET): $(SRC) $(TQ_SRC) ../turboquant/turboquant.hpp
	$(NVCC) $(CFLAGS) $(SRC) $(TQ_SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
