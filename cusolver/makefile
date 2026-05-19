# Makefile for cuSOLVER SVD Project
NVCC = nvcc
CFLAGS = -O3 -arch=sm_70
LIBS = -lcusolver -lcudart

TARGET = cusolver
SRC = cusolver.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)