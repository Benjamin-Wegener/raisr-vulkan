CC = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra -pthread
LIBS = -ljpeg -lm
VULKAN_LIBS = $(shell pkg-config --libs vulkan) -lvulkan
VULKAN_CFLAGS = $(shell pkg-config --cflags vulkan)

# Check if Vulkan is available
HAVE_VULKAN := $(shell pkg-config --exists vulkan && echo YES || echo NO)

TARGET = raisr_vulkan
SRCS = raisr_vulkan.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
ifeq ($(HAVE_VULKAN),YES)
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -o $@ $^ $(LIBS) $(VULKAN_LIBS)
	@echo "Vulkan version built successfully"
else
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "Vulkan not available, built CPU-only version"
endif

%.o: %.c
ifeq ($(HAVE_VULKAN),YES)
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif

clean:
	rm -f $(OBJS) $(TARGET)
