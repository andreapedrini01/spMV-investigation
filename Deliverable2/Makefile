NVCC = nvcc
ARCH = sm_80
NVCC_FLAGS = -O3 -arch=$(ARCH) -Xcompiler -Wall
LIBS = -lmpi -lcusparse -lm

# Auto-detect the MPI install from the mpicc wrapper so nvcc finds mpi.h and
# libmpi with any OpenMPI module (including the CUDA-aware OpenMpi/*-CUDA-*).
# Falls back to the compiler's default search paths if mpicc is not on PATH.
MPICC := $(shell which mpicc 2>/dev/null)
ifneq ($(MPICC),)
  MPI_ROOT := $(patsubst %/bin/mpicc,%,$(MPICC))
  NVCC_FLAGS += -I$(MPI_ROOT)/include
  LIBS += -L$(MPI_ROOT)/lib
endif

# Optional bonus features:
#   make NCCL=1   enable the NCCL AllGather transport (links -lnccl)
#   make NVML=1   print each rank's GPU UUID at startup (links -lnvidia-ml)
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
