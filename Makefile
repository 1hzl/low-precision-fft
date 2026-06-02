CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3
NVCC       = "$(CUDA_PATH)/bin/nvcc.exe"
SM_ARCH    = 120
CUDA_FLAGS = -arch=sm_$(SM_ARCH) -O3 -lcufft -lcudart

SRC_DIR    = src/cuda
BUILD_DIR  = build
SOURCES    = $(wildcard $(SRC_DIR)/*.cu)
TARGETS    = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.exe,$(SOURCES))

# MSVC auto-detection (vswhere → MSVC bin path)
VSWHERE     = $(or $(wildcard /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer/vswhere.exe),$(wildcard "/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"))
VS_INSTALL := $(shell $(VSWHERE) -latest -property installationPath 2>/dev/null)
MSVC_BIN   := $(shell ls -d "$(VS_INSTALL)"/VC/Tools/MSVC/*/bin/Hostx64/x64 2>/dev/null | head -1)
ifeq ($(MSVC_BIN),)
  # fallback: glob under known BuildTools path
  MSVC_BIN := $(shell ls -d "/c/Program Files (x86)/Microsoft Visual Studio/"*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64 2>/dev/null | head -1)
endif
ifneq ($(MSVC_BIN),)
  export PATH := $(MSVC_BIN):$(PATH)
endif

.PHONY: all clean list

all: $(BUILD_DIR) $(TARGETS)
	@echo ">> All done: $(words $(TARGETS)) targets compiled."

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.exe: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*.exe $(BUILD_DIR)/*.pdb $(BUILD_DIR)/*.exp $(BUILD_DIR)/*.lib
	@echo ">> Clean done."

list:
	@echo "Sources:" && ls -1 $(SOURCES) && echo "---" && echo "Targets:" && ls -1 $(TARGETS) 2>/dev/null || echo "(not built yet)"
