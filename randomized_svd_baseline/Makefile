# Makefile for randomized_multigpu_svd.cu
NVCC = nvcc
CFLAGS = -O3 -std=c++17 -arch=sm_70
LIBS = -lcublas -lcusolver -lcudart

# TARGET = randomized_svd_multigpu
# SRC = randomized_svd_multigpu.cu
TARGET = randomized_svd_multigpu
SRC = randomized_svd_multigpu.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
