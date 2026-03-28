# RAISR Makefile

CC     = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra -pthread
LIBS   = -ljpeg -lpng -lm -pthread

VULKAN_CFLAGS = $(shell pkg-config --cflags vulkan 2>/dev/null || echo "-I/usr/include")
VULKAN_LIBS   = $(shell pkg-config --libs   vulkan 2>/dev/null || echo "-lvulkan")
HAVE_VULKAN  := $(shell pkg-config --exists vulkan && echo 1 || echo 0)

SHADER_SRC  = raisr.glsl
SHADER_SPV  = raisr.spv
TARGET      = raisr

.PHONY: all clean

ifeq ($(HAVE_VULKAN),1)
all: $(SHADER_SPV) $(TARGET)

$(SHADER_SPV): $(SHADER_SRC)
	@echo "Compiling shader: $(SHADER_SRC) -> $(SHADER_SPV)"
	glslc -fshader-stage=compute $(SHADER_SRC) -o $(SHADER_SPV)

$(TARGET): raisr.c
	@echo "Building with Vulkan support..."
	$(CC) $(CFLAGS) -DHAVE_VULKAN=1 $(VULKAN_CFLAGS) -o $@ $< $(LIBS) $(VULKAN_LIBS)
	@echo "Build complete: $(TARGET)"
	@echo "Run './$(TARGET) --help' for usage."
else
all: $(TARGET)

$(TARGET): raisr.c
	@echo "Vulkan not found, building CPU-only..."
	$(CC) $(CFLAGS) -DHAVE_VULKAN=0 -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET)"
	@echo "Run './$(TARGET) --help' for usage."
endif

clean:
	rm -f $(TARGET) $(SHADER_SPV)