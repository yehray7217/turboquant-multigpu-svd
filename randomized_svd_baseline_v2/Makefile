# Makefile for randomized_svd_multigpu_v2.cu
NVCC = nvcc
CFLAGS = -O3 -std=c++17 -arch=sm_70
LIBS = -lcublas -lcusolver -lcudart

TARGET = randomized_svd_multigpu_v2
SRC = randomized_svd_multigpu_v2.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
