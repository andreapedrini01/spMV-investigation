NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -Xcompiler -Wall
LIBS = -lcusparse -lm

BIN_DIR = bin
SRC_DIR = src
OUT_DIR = outputs

SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/mmio.c $(SRC_DIR)/spmv_cpu.c \
          $(SRC_DIR)/spmv_gpu.cu $(SRC_DIR)/utils.cu
TARGET = $(BIN_DIR)/spmv

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(BIN_DIR) $(OUT_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
