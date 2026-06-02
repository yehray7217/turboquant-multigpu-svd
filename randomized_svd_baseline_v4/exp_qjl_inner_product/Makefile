NVCC ?= nvcc
BUILD_DIR := .build
TARGET := $(BUILD_DIR)/qjl_inner_product_probe
SRC := qjl_inner_product_probe.cu

NVCC_FLAGS ?= -O3 -std=c++17 -arch=sm_70
LDLIBS := -lcublas

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)
