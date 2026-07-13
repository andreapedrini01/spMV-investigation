NVCC = nvcc
ARCH = sm_80
NVCC_FLAGS = -O3 -arch=$(ARCH) -Xcompiler -Wall
LIBS = -lmpi -lcusparse -lm

# Optional bonus features:
#   make NCCL=1   enable the NCCL AllGather transport (links -lnccl)
#   make NVML=1   print each rank's GPU UUID at startup (links -lnvidia-ml)
# After `module load OpenMPI CUDA/12.5.0` the wrappers put the MPI headers and
# libmpi on the default search path, so nvcc links MPI directly (as in Lab 12).
# Build on a COMPUTE node (edu01): the login node's module assembler crashes.
ifeq ($(NCCL),1)
  NVCC_FLAGS += -DUSE_NCCL
  LIBS += -lnccl
endif
ifeq ($(NVML),1)
  NVCC_FLAGS += -DUSE_NVML
  LIBS += -lnvidia-ml
endif

SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj
OUT_DIR = outputs

SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/mmio.cu $(SRC_DIR)/utils.cu \
          $(SRC_DIR)/spmv_local.cu $(SRC_DIR)/distributed.cu \
          $(SRC_DIR)/gen_matrix.cu $(SRC_DIR)/spmv_cpu_ref.cu
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
HEADERS = $(wildcard $(SRC_DIR)/*.h)
TARGET = $(BIN_DIR)/dspmv

all: $(TARGET)

# separate objects so `make -j` compiles the units in parallel (fast build
# under the 5-minute edu-short limit)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR) $(OUT_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

.PHONY: all clean
