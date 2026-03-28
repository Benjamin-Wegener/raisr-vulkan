/*
 * RAISR - Combined CPU/Vulkan Implementation
 * 
 * Automatically uses Vulkan GPU if available, falls back to CPU otherwise.
 * 
 * Usage:
 *   ./raisr input.jpg output.jpg              # Auto-detect (Vulkan if available)
 *   ./raisr input.jpg output.jpg --cpu        # Force CPU mode
 *   ./raisr input.jpg output.png --format png # PNG output
 *   ./raisr input.jpg out.jpg --jpeg 82       # JPEG with custom quality
 *
 * Default JPEG quality: 82%
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <jpeglib.h>
#include <png.h>

// Vulkan detection and includes
#ifdef __has_include
#  if __has_include(<vulkan/vulkan.h>)
#    define HAVE_VULKAN 1
#    include <vulkan/vulkan.h>
#  else
#    define HAVE_VULKAN 0
#  endif
#else
#  define HAVE_VULKAN 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// RAISR Parameters
#define R 2
#define PATCH_SIZE 11
#define GRADIENT_SIZE 9
#define QANGLE 24
#define QSTRENGTH 3
#define QCOHERENCE 3

// Calculate margins
#define MAX_BLOCK_SIZE ((PATCH_SIZE > GRADIENT_SIZE) ? PATCH_SIZE : GRADIENT_SIZE)
#define MARGIN (MAX_BLOCK_SIZE / 2)
#define PATCH_MARGIN (PATCH_SIZE / 2)
#define GRADIENT_MARGIN (GRADIENT_SIZE / 2)

// Global configuration
typedef struct {
    int use_cpu_only;          // Force CPU mode
    int use_png;               // PNG output format
    int jpeg_quality;          // JPEG quality (1-100)
    const char* input_file;    // Input image path
    const char* output_file;   // Output image path
} config_t;

// Thread data structure
typedef struct {
    unsigned char* y_channel;
    unsigned char* cb_channel;
    unsigned char* cr_channel;
    double* h;  // filter
    double* predictHR;
    int heightLR;
    int widthLR;
    int heightHR;
    int widthHR;
    int start_row;
    int end_row;
    int thread_id;
} thread_data_t;

// Vulkan globals (only if available)
#if HAVE_VULKAN
static VkInstance vk_instance;
static VkPhysicalDevice vk_physical_device;
static VkDevice vk_device;
static VkQueue vk_queue;
static uint32_t vk_queue_family_index;
static VkCommandPool vk_command_pool;
static VkPipelineLayout vk_pipeline_layout;
static VkPipeline vk_compute_pipeline;
static VkDescriptorSetLayout vk_descriptor_set_layout;
static VkDescriptorPool vk_descriptor_pool;
static VkShaderModule vk_compute_shader;
#endif

// Function prototypes
void print_usage(const char* program);
int parse_arguments(int argc, char* argv[], config_t* config);

int read_jpeg_file(const char* filename, unsigned char** y_channel, unsigned char** cb_channel, unsigned char** cr_channel, int* height, int* width);
int write_jpeg_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width, int quality);
int write_png_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width);
int write_image_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width, int use_png, int jpeg_quality);
void upscale_bilinear_channel(unsigned char* input, unsigned char* output, int heightLR, int widthLR, int heightHR, int widthHR);

void* process_rows_cpu(void* arg);
void hashkey(double* gradientblock, int Qangle, double* weighting, int* angle, int* strength, int* coherence);

#if HAVE_VULKAN
int init_vulkan(void);
void cleanup_vulkan(void);
void* process_rows_vulkan(void* arg);
#endif

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    config_t config = {
        .use_cpu_only = 0,
        .use_png = 0,
        .jpeg_quality = 82,  // Default 82% quality
        .input_file = NULL,
        .output_file = NULL
    };

    if (parse_arguments(argc, argv, &config) != 0) {
        return 1;
    }

    printf("RAISR - Rapid and Accurate Image Super Resolution\n");
    printf("=================================================\n");
    printf("Input:  %s\n", config.input_file);
    printf("Output: %s\n", config.output_file);
    printf("Format: %s\n", config.use_png ? "PNG (lossless)" : "JPEG");
    if (!config.use_png) {
        printf("JPEG Quality: %d%%\n", config.jpeg_quality);
    }
    printf("Mode:   %s\n", config.use_cpu_only ? "CPU (forced)" : "Auto (Vulkan if available)");
    printf("\n");

    // Initialize Vulkan if not forced to CPU
    int use_vulkan = 0;
#if HAVE_VULKAN
    if (!config.use_cpu_only) {
        use_vulkan = init_vulkan();
        if (use_vulkan) {
            printf("✓ Vulkan GPU acceleration enabled\n");
        } else {
            printf("ℹ Vulkan not available, using CPU\n");
        }
    } else {
        printf("ℹ CPU mode forced via --cpu flag\n");
    }
#else
    printf("ℹ Vulkan support not compiled in, using CPU\n");
#endif

    unsigned char* y_channel = NULL;
    unsigned char* cb_channel = NULL;
    unsigned char* cr_channel = NULL;
    int height, width;

    printf("\nReading input image...\n");
    if (!read_jpeg_file(config.input_file, &y_channel, &cb_channel, &cr_channel, &height, &width)) {
        fprintf(stderr, "Error: Failed to read input image '%s'\n", config.input_file);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }
    printf("Input size: %dx%d\n", width, height);

    int heightLR = height;
    int widthLR = width;
    int heightHR = heightLR * 2;
    int widthHR = widthLR * 2;

    printf("Output size: %dx%d (2x upscale)\n", widthHR, heightHR);

    unsigned char* upscaled_y = malloc(heightHR * widthHR);
    unsigned char* upscaled_cb = malloc(heightHR * widthHR);
    unsigned char* upscaled_cr = malloc(heightHR * widthHR);

    if (!upscaled_y || !upscaled_cb || !upscaled_cr) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(y_channel); free(cb_channel); free(cr_channel);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    printf("Upscaling with bilinear interpolation...\n");
    upscale_bilinear_channel(y_channel, upscaled_y, heightLR, widthLR, heightHR, widthHR);
    upscale_bilinear_channel(cb_channel, upscaled_cb, heightLR, widthLR, heightHR, widthHR);
    upscale_bilinear_channel(cr_channel, upscaled_cr, heightLR, widthLR, heightHR, widthHR);

    double* h = NULL;
    int filter_size = QANGLE * QSTRENGTH * QCOHERENCE * R * R * PATCH_SIZE * PATCH_SIZE;

    printf("Loading filter from filter.bin...\n");
    FILE* filter_file = fopen("filter.bin", "rb");
    if (!filter_file) {
        fprintf(stderr, "Error: Could not open filter.bin\n");
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    unsigned int file_filter_size;
    if (fread(&file_filter_size, sizeof(unsigned int), 1, filter_file) != 1) {
        fprintf(stderr, "Error: Failed to read filter size\n");
        fclose(filter_file);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    if (file_filter_size != (unsigned int)filter_size) {
        fprintf(stderr, "Error: Filter size mismatch. Expected %d, got %u\n", filter_size, file_filter_size);
        fclose(filter_file);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    h = malloc(filter_size * sizeof(double));
    if (!h) {
        fprintf(stderr, "Error: Memory allocation failed for filter\n");
        fclose(filter_file);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    size_t elements_read = fread(h, sizeof(double), filter_size, filter_file);
    fclose(filter_file);

    if (elements_read != (size_t)filter_size) {
        fprintf(stderr, "Error: Failed to read filter data\n");
        free(h);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    printf("Loaded %d filter coefficients\n", filter_size);

    int predictHR_height = heightHR - 2 * MARGIN;
    int predictHR_width = widthHR - 2 * MARGIN;
    double* predictHR = calloc(predictHR_height * predictHR_width, sizeof(double));
    if (!predictHR) {
        fprintf(stderr, "Error: Memory allocation failed for predictHR\n");
        free(h);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    printf("\nProcessing with %s...\n", use_vulkan ? "Vulkan GPU" : "CPU");
    clock_t start_time = clock();

    if (use_vulkan) {
#if HAVE_VULKAN
        thread_data_t thread_data;
        thread_data.y_channel = upscaled_y;
        thread_data.cb_channel = upscaled_cb;
        thread_data.cr_channel = upscaled_cr;
        thread_data.h = h;
        thread_data.predictHR = predictHR;
        thread_data.heightLR = heightLR;
        thread_data.widthLR = widthLR;
        thread_data.heightHR = heightHR;
        thread_data.widthHR = widthHR;
        thread_data.start_row = MARGIN;
        thread_data.end_row = heightHR - MARGIN;
        thread_data.thread_id = 0;

        process_rows_vulkan(&thread_data);
#endif
    } else {
        int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_threads < 1) num_threads = 4;
        pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
        thread_data_t* thread_data = malloc(num_threads * sizeof(thread_data_t));

        int rows_per_thread = (heightHR - 2 * MARGIN) / num_threads;

        for (int i = 0; i < num_threads; i++) {
            thread_data[i].y_channel = upscaled_y;
            thread_data[i].cb_channel = upscaled_cb;
            thread_data[i].cr_channel = upscaled_cr;
            thread_data[i].h = h;
            thread_data[i].predictHR = predictHR;
            thread_data[i].heightLR = heightLR;
            thread_data[i].widthLR = widthLR;
            thread_data[i].heightHR = heightHR;
            thread_data[i].widthHR = widthHR;
            thread_data[i].start_row = MARGIN + i * rows_per_thread;
            thread_data[i].end_row = (i == num_threads - 1) ? (heightHR - MARGIN) : (MARGIN + (i + 1) * rows_per_thread);
            thread_data[i].thread_id = i;

            pthread_create(&threads[i], NULL, process_rows_cpu, &thread_data[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        free(threads);
        free(thread_data);
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Processing complete in %.3f seconds\n", cpu_time_used);

    printf("Scaling Y channel...\n");
    for (int row = MARGIN; row < heightHR - MARGIN; row++) {
        for (int col = MARGIN; col < widthHR - MARGIN; col++) {
            double value = predictHR[(row - MARGIN) * predictHR_width + (col - MARGIN)] * 255.0;
            upscaled_y[row * widthHR + col] = (unsigned char)fmin(fmax(value, 0.0), 255.0);
        }
    }

    printf("Writing output image...\n");
    if (!write_image_file(config.output_file, upscaled_y, upscaled_cb, upscaled_cr, heightHR, widthHR, config.use_png, config.jpeg_quality)) {
        fprintf(stderr, "Error: Failed to write output image '%s'\n", config.output_file);
        free(h); free(predictHR);
        free(y_channel); free(cb_channel); free(cr_channel);
        free(upscaled_y); free(upscaled_cb); free(upscaled_cr);
#if HAVE_VULKAN
        if (use_vulkan) cleanup_vulkan();
#endif
        return 1;
    }

    printf("\n✓ Output written to %s\n", config.output_file);
    printf("Done!\n");

    free(h); free(predictHR);
    free(y_channel); free(cb_channel); free(cr_channel);
    free(upscaled_y); free(upscaled_cb); free(upscaled_cr);

#if HAVE_VULKAN
    if (use_vulkan) cleanup_vulkan();
#endif

    return 0;
}

// ============================================================================
// Command-line argument parsing
// ============================================================================

void print_usage(const char* program) {
    printf("Usage: %s [options] <input_image> <output_image>\n", program);
    printf("\nRAISR - Rapid and Accurate Image Super Resolution\n");
    printf("Automatically uses Vulkan GPU if available, falls back to CPU.\n");
    printf("\nOptions:\n");
    printf("  -f, --format <fmt>     Output format: jpeg or png (default: jpeg)\n");
    printf("  -j, --jpeg <quality>   JPEG quality 1-100 (default: 82)\n");
    printf("  -c, --cpu              Force CPU mode (disable Vulkan)\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s input.jpg output.jpg              # Auto mode, JPEG 82%%\n", program);
    printf("  %s input.jpg output.png -f png       # PNG format\n", program);
    printf("  %s input.jpg out.jpg -j 90           # JPEG 90%% quality\n", program);
    printf("  %s input.jpg out.jpg --cpu           # Force CPU mode\n", program);
    printf("  %s in.jpg out.jpg -f png -c          # PNG, CPU only\n", program);
}

int parse_arguments(int argc, char* argv[], config_t* config) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cpu") == 0) {
            config->use_cpu_only = 1;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--format") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -f/--format requires an argument\n");
                print_usage(argv[0]);
                return 1;
            }
            i++;
            if (strcasecmp(argv[i], "png") == 0) {
                config->use_png = 1;
            } else if (strcasecmp(argv[i], "jpeg") == 0 || strcasecmp(argv[i], "jpg") == 0) {
                config->use_png = 0;
            } else {
                fprintf(stderr, "Error: Unknown format '%s'. Use 'jpeg' or 'png'.\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--jpeg") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -j/--jpeg requires an argument\n");
                print_usage(argv[0]);
                return 1;
            }
            i++;
            char *end;
            long q = strtol(argv[i], &end, 10);
            if (*end != '\0' || q < 1 || q > 100) {
                fprintf(stderr, "Error: JPEG quality must be 1-100\n");
                return 1;
            }
            config->jpeg_quality = (int)q;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else if (config->input_file == NULL) {
            config->input_file = argv[i];
        } else if (config->output_file == NULL) {
            config->output_file = argv[i];
        } else {
            fprintf(stderr, "Error: Too many arguments\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config->input_file == NULL || config->output_file == NULL) {
        fprintf(stderr, "Error: Input and output filenames are required\n");
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}

// ============================================================================
// Image I/O functions
// ============================================================================

int read_jpeg_file(const char* filename, unsigned char** y_channel, unsigned char** cb_channel, unsigned char** cr_channel, int* height, int* width) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE* infile = fopen(filename, "rb");
    if (!infile) return 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_YCbCr;
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;

    *y_channel = malloc((*height) * (*width));
    *cb_channel = malloc((*height) * (*width));
    *cr_channel = malloc((*height) * (*width));

    if (!*y_channel || !*cb_channel || !*cr_channel) {
        fclose(infile);
        jpeg_destroy_decompress(&cinfo);
        if (*y_channel) free(*y_channel);
        if (*cb_channel) free(*cb_channel);
        if (*cr_channel) free(*cr_channel);
        return 0;
    }

    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, (*width) * 3, 1);

    for (int row = 0; row < *height; row++) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int col = 0; col < *width; col++) {
            (*y_channel)[row * (*width) + col] = buffer[0][col * 3];
            (*cb_channel)[row * (*width) + col] = buffer[0][col * 3 + 1];
            (*cr_channel)[row * (*width) + col] = buffer[0][col * 3 + 2];
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 1;
}

int write_jpeg_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE* outfile = fopen(filename, "wb");
    if (!outfile) return 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_YCbCr;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    unsigned char* row = malloc(width * 3);

    while (cinfo.next_scanline < cinfo.image_height) {
        for (int i = 0; i < width; i++) {
            row[i * 3] = y_channel[cinfo.next_scanline * width + i];
            row[i * 3 + 1] = cb_channel[cinfo.next_scanline * width + i];
            row[i * 3 + 2] = cr_channel[cinfo.next_scanline * width + i];
        }
        row_pointer[0] = row;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    free(row);
    return 1;
}

int write_png_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width) {
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) return 0;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(outfile);
        return 0;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(outfile);
        return 0;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(outfile);
        return 0;
    }

    png_init_io(png_ptr, outfile);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_bytep row = malloc(width * 3 * sizeof(png_byte));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            double Y_val = y_channel[idx];
            double Cb = cb_channel[idx] - 128.0;
            double Cr = cr_channel[idx] - 128.0;

            double R_val = Y_val + 1.402 * Cr;
            double G_val = Y_val - 0.344136 * Cb - 0.714136 * Cr;
            double B_val = Y_val + 1.772 * Cb;

            row[x * 3] = (png_byte)(R_val < 0 ? 0 : (R_val > 255 ? 255 : R_val));
            row[x * 3 + 1] = (png_byte)(G_val < 0 ? 0 : (G_val > 255 ? 255 : G_val));
            row[x * 3 + 2] = (png_byte)(B_val < 0 ? 0 : (B_val > 255 ? 255 : B_val));
        }
        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(outfile);
    free(row);

    return 1;
}

int write_image_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width, int use_png, int jpeg_quality) {
    if (use_png) {
        return write_png_file(filename, y_channel, cb_channel, cr_channel, height, width);
    } else {
        return write_jpeg_file(filename, y_channel, cb_channel, cr_channel, height, width, jpeg_quality);
    }
}

void upscale_bilinear_channel(unsigned char* input, unsigned char* output, int heightLR, int widthLR, int heightHR, int widthHR) {
    for (int y = 0; y < heightHR; y++) {
        for (int x = 0; x < widthHR; x++) {
            double lr_y = (double)y / 2.0;
            double lr_x = (double)x / 2.0;

            int y1 = (int)floor(lr_y);
            int x1 = (int)floor(lr_x);
            int y2 = y1 + 1;
            int x2 = x1 + 1;

            if (y2 >= heightLR) y2 = heightLR - 1;
            if (x2 >= widthLR) x2 = widthLR - 1;

            double Q11 = input[y1 * widthLR + x1];
            double Q21 = input[y1 * widthLR + x2];
            double Q12 = input[y2 * widthLR + x1];
            double Q22 = input[y2 * widthLR + x2];

            double wx = lr_x - x1;
            double wy = lr_y - y1;

            double result = Q11 * (1 - wx) * (1 - wy) +
                           Q21 * wx * (1 - wy) +
                           Q12 * (1 - wx) * wy +
                           Q22 * wx * wy;

            output[y * widthHR + x] = (unsigned char)result;
        }
    }
}

// ============================================================================
// CPU Processing
// ============================================================================

void* process_rows_cpu(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    double* weighting = malloc(GRADIENT_SIZE * GRADIENT_SIZE * sizeof(double));
    if (!weighting) return NULL;

    double sigma = 2.0;
    double sum = 0.0;
    for (int i = 0; i < GRADIENT_SIZE; i++) {
        for (int j = 0; j < GRADIENT_SIZE; j++) {
            int m = i - GRADIENT_MARGIN;
            int n = j - GRADIENT_MARGIN;
            double h = exp(-(m*m + n*n) / (2.0 * sigma * sigma));
            weighting[i * GRADIENT_SIZE + j] = h;
            sum += h;
        }
    }

    for (int i = 0; i < GRADIENT_SIZE * GRADIENT_SIZE; i++) {
        weighting[i] /= sum;
    }

    // Stack arrays to avoid ~28M heap calls for a 3118x4500 image
    double patch[PATCH_SIZE * PATCH_SIZE];
    double gradientblock[GRADIENT_SIZE * GRADIENT_SIZE];

    for (int row = data->start_row; row < data->end_row; row++) {
        for (int col = MARGIN; col < data->widthHR - MARGIN; col++) {
            int patch_idx = 0;
            for (int i = row - PATCH_MARGIN; i <= row + PATCH_MARGIN; i++) {
                for (int j = col - PATCH_MARGIN; j <= col + PATCH_MARGIN; j++) {
                    patch[patch_idx++] = data->y_channel[i * data->widthHR + j] / 255.0;
                }
            }

            int grad_idx = 0;
            for (int i = row - GRADIENT_MARGIN; i <= row + GRADIENT_MARGIN; i++) {
                for (int j = col - GRADIENT_MARGIN; j <= col + GRADIENT_MARGIN; j++) {
                    gradientblock[grad_idx++] = data->y_channel[i * data->widthHR + j] / 255.0;
                }
            }

            int angle, strength, coherence;
            hashkey(gradientblock, QANGLE, weighting, &angle, &strength, &coherence);

            int pixeltype = ((row - MARGIN) % R) * R + ((col - MARGIN) % R);

            int filter_idx = ((angle * QSTRENGTH + strength) * QCOHERENCE + coherence) * R * R * PATCH_SIZE * PATCH_SIZE +
                            pixeltype * PATCH_SIZE * PATCH_SIZE;

            double result = 0.0;
            for (int i = 0; i < PATCH_SIZE * PATCH_SIZE; i++) {
                result += patch[i] * data->h[filter_idx + i];
            }

            data->predictHR[(row - MARGIN) * (data->widthHR - 2 * MARGIN) + (col - MARGIN)] = result;
        }
    }

    free(weighting);
    return NULL;
}

void hashkey(double* gradientblock, int Qangle, double* weighting, int* angle, int* strength, int* coherence) {
    // Stack arrays - GRADIENT_SIZE is compile-time constant
    double gy[GRADIENT_SIZE * GRADIENT_SIZE];
    double gx[GRADIENT_SIZE * GRADIENT_SIZE];

    for (int i = 0; i < GRADIENT_SIZE; i++) {
        for (int j = 0; j < GRADIENT_SIZE; j++) {
            int idx = i * GRADIENT_SIZE + j;

            if (i == 0) {
                gy[idx] = gradientblock[(i + 1) * GRADIENT_SIZE + j] - gradientblock[idx];
            } else if (i == GRADIENT_SIZE - 1) {
                gy[idx] = gradientblock[idx] - gradientblock[(i - 1) * GRADIENT_SIZE + j];
            } else {
                gy[idx] = (gradientblock[(i + 1) * GRADIENT_SIZE + j] - gradientblock[(i - 1) * GRADIENT_SIZE + j]) / 2.0;
            }

            if (j == 0) {
                gx[idx] = gradientblock[i * GRADIENT_SIZE + (j + 1)] - gradientblock[idx];
            } else if (j == GRADIENT_SIZE - 1) {
                gx[idx] = gradientblock[i * GRADIENT_SIZE + j] - gradientblock[i * GRADIENT_SIZE + (j - 1)];
            } else {
                gx[idx] = (gradientblock[i * GRADIENT_SIZE + (j + 1)] - gradientblock[i * GRADIENT_SIZE + (j - 1)]) / 2.0;
            }
        }
    }

    double gxx = 0.0, gyy = 0.0, gxy = 0.0;
    for (int i = 0; i < GRADIENT_SIZE; i++) {
        for (int j = 0; j < GRADIENT_SIZE; j++) {
            int idx = i * GRADIENT_SIZE + j;
            double weight = weighting[idx];
            gxx += weight * gx[idx] * gx[idx];
            gyy += weight * gy[idx] * gy[idx];
            gxy += weight * gx[idx] * gy[idx];
        }
    }

    double trace = gxx + gyy;
    double discriminant = (gxx - gyy) * (gxx - gyy) + 4 * gxy * gxy;

    if (discriminant < 0) discriminant = 0;
    double sqrt_discriminant = sqrt(discriminant);

    double lambda1 = (trace + sqrt_discriminant) / 2.0;
    double lambda2 = (trace - sqrt_discriminant) / 2.0;

    double vx, vy;
    if (fabs(gxy) > 1e-10) {
        vx = lambda1 - gyy;
        vy = gxy;
    } else if (fabs(gxx - lambda1) > 1e-10) {
        vx = gxy;
        vy = lambda1 - gxx;
    } else {
        vx = 1.0;
        vy = 0.0;
    }

    double norm = sqrt(vx * vx + vy * vy);
    if (norm > 1e-10) {
        vx /= norm;
        vy /= norm;
    }

    double theta = atan2(vy, vx);
    if (theta < 0) theta += M_PI;

    double lamda = lambda1;

    double sqrt_lamda1 = sqrt(fabs(lambda1));
    double sqrt_lamda2 = sqrt(fabs(lambda2));
    double u = (sqrt_lamda1 + sqrt_lamda2 == 0) ? 0 : (sqrt_lamda1 - sqrt_lamda2) / (sqrt_lamda1 + sqrt_lamda2);

    *angle = (int)(theta / M_PI * Qangle);
    if (lamda < 0.0001) {
        *strength = 0;
    } else if (lamda > 0.001) {
        *strength = 2;
    } else {
        *strength = 1;
    }
    if (u < 0.25) {
        *coherence = 0;
    } else if (u > 0.5) {
        *coherence = 2;
    } else {
        *coherence = 1;
    }

    if (*angle > Qangle - 1) *angle = Qangle - 1;
    else if (*angle < 0) *angle = 0;
}

// ============================================================================
// Vulkan Implementation (only if HAVE_VULKAN)
// ============================================================================

#if HAVE_VULKAN

// Uniform buffer structure - must match shader exactly
typedef struct {
    uint32_t widthHR;
    uint32_t heightHR;
    uint32_t widthLR;
    uint32_t heightLR;
    uint32_t filter_size;
    uint32_t margin;
    uint32_t patch_size;
    uint32_t gradient_size;
    uint32_t qangle;
    uint32_t qstrength;
    uint32_t qcoherence;
    uint32_t r;
} UniformBufferObject;

static int vulkan_initialized = 0;

int init_vulkan(void) {
    VkResult result;

    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "RAISR";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    result = vkCreateInstance(&create_info, NULL, &vk_instance);
    if (result != VK_SUCCESS) return 0;

    uint32_t device_count = 0;
    result = vkEnumeratePhysicalDevices(vk_instance, &device_count, NULL);
    if (result != VK_SUCCESS || device_count == 0) {
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(vk_instance, &device_count, devices);
    vk_physical_device = devices[0];
    free(devices);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, NULL);
    
    VkQueueFamilyProperties* queue_families = malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, queue_families);

    vk_queue_family_index = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vk_queue_family_index = i;
            break;
        }
    }
    free(queue_families);

    if (vk_queue_family_index == UINT32_MAX) {
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = vk_queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    result = vkCreateDevice(vk_physical_device, &device_create_info, NULL, &vk_device);
    if (result != VK_SUCCESS) {
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    vkGetDeviceQueue(vk_device, vk_queue_family_index, 0, &vk_queue);

    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = vk_queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(vk_device, &pool_info, NULL, &vk_command_pool);
    if (result != VK_SUCCESS) {
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 4;
    layout_info.pBindings = bindings;

    result = vkCreateDescriptorSetLayout(vk_device, &layout_info, NULL, &vk_descriptor_set_layout);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkDescriptorPoolSize pool_sizes[2] = {0};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 3;

    VkDescriptorPoolCreateInfo pool_info_desc = {0};
    pool_info_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // FREE_DESCRIPTOR_SET_BIT required because process_rows_vulkan calls vkFreeDescriptorSets
    pool_info_desc.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info_desc.poolSizeCount = 2;
    pool_info_desc.pPoolSizes = pool_sizes;
    pool_info_desc.maxSets = 1;

    result = vkCreateDescriptorPool(vk_device, &pool_info_desc, NULL, &vk_descriptor_pool);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    FILE* file = fopen("raisr.spv", "rb");
    if (!file) {
        vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint32_t* code = malloc(file_size);
    fread(code, 1, file_size, file);
    fclose(file);

    VkShaderModuleCreateInfo shader_info = {0};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = file_size;
    shader_info.pCode = code;

    result = vkCreateShaderModule(vk_device, &shader_info, NULL, &vk_compute_shader);
    free(code);

    if (result != VK_SUCCESS) {
        vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = {0};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk_descriptor_set_layout;

    result = vkCreatePipelineLayout(vk_device, &pipeline_layout_info, NULL, &vk_pipeline_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(vk_device, vk_compute_shader, NULL);
        vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkComputePipelineCreateInfo pipeline_info = {0};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = vk_compute_shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = vk_pipeline_layout;

    result = vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &vk_compute_pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyShaderModule(vk_device, vk_compute_shader, NULL);
        vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    vulkan_initialized = 1;
    return 1;
}

void cleanup_vulkan(void) {
    if (!vulkan_initialized) return;
    vkDeviceWaitIdle(vk_device);
    vkDestroyPipeline(vk_device, vk_compute_pipeline, NULL);
    vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
    vkDestroyShaderModule(vk_device, vk_compute_shader, NULL);
    vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, NULL);
    vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
    vkDestroyDevice(vk_device, NULL);
    vkDestroyInstance(vk_instance, NULL);
    vulkan_initialized = 0;
}

VkBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory* out_memory) {
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(vk_device, &buffer_info, NULL, &buffer);
    if (result != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(vk_device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;

    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &mem_properties);

    int found = 0;
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((mem_requirements.memoryTypeBits & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            alloc_info.memoryTypeIndex = i;
            found = 1;
            break;
        }
    }

    if (!found) {
        vkDestroyBuffer(vk_device, buffer, NULL);
        return VK_NULL_HANDLE;
    }

    VkDeviceMemory memory;
    result = vkAllocateMemory(vk_device, &alloc_info, NULL, &memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(vk_device, buffer, NULL);
        return VK_NULL_HANDLE;
    }

    vkBindBufferMemory(vk_device, buffer, memory, 0);
    *out_memory = memory;
    return buffer;
}

void* process_rows_vulkan(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int widthHR = data->widthHR;
    int heightHR = data->heightHR;
    int filter_size = QANGLE * QSTRENGTH * QCOHERENCE * R * R * PATCH_SIZE * PATCH_SIZE;

    VkDeviceMemory input_memory = VK_NULL_HANDLE, output_memory = VK_NULL_HANDLE;
    VkDeviceMemory param_memory = VK_NULL_HANDLE, filter_memory = VK_NULL_HANDLE;
    VkBuffer input_buffer = VK_NULL_HANDLE, output_buffer = VK_NULL_HANDLE;
    VkBuffer param_buffer = VK_NULL_HANDLE, filter_buffer = VK_NULL_HANDLE;
    
    VkDeviceSize float_buffer_size = sizeof(float) * widthHR * heightHR;
    VkDeviceSize param_buffer_size = sizeof(UniformBufferObject);
    VkDeviceSize filter_buffer_size = sizeof(float) * filter_size;

    input_buffer = create_buffer(float_buffer_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &input_memory);

    output_buffer = create_buffer(float_buffer_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &output_memory);

    param_buffer = create_buffer(param_buffer_size,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &param_memory);

    filter_buffer = create_buffer(filter_buffer_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &filter_memory);

    if (!input_buffer || !output_buffer || !param_buffer || !filter_buffer) {
        goto cleanup;
    }

    float* input_data = malloc(float_buffer_size);
    for (int i = 0; i < widthHR * heightHR; i++) {
        input_data[i] = (float)data->y_channel[i] / 255.0f;
    }

    void* mapped;
    vkMapMemory(vk_device, input_memory, 0, float_buffer_size, 0, &mapped);
    memcpy(mapped, input_data, float_buffer_size);
    vkUnmapMemory(vk_device, input_memory);
    free(input_data);

    UniformBufferObject params = {
        .widthHR = (uint32_t)widthHR,
        .heightHR = (uint32_t)heightHR,
        .widthLR = (uint32_t)data->widthLR,
        .heightLR = (uint32_t)data->heightLR,
        .filter_size = filter_size,
        .margin = MARGIN,
        .patch_size = PATCH_SIZE,
        .gradient_size = GRADIENT_SIZE,
        .qangle = QANGLE,
        .qstrength = QSTRENGTH,
        .qcoherence = QCOHERENCE,
        .r = R
    };

    vkMapMemory(vk_device, param_memory, 0, param_buffer_size, 0, &mapped);
    memcpy(mapped, &params, param_buffer_size);
    vkUnmapMemory(vk_device, param_memory);

    float* filter_data = malloc(filter_buffer_size);
    for (int i = 0; i < filter_size; i++) {
        filter_data[i] = (float)data->h[i];
    }

    vkMapMemory(vk_device, filter_memory, 0, filter_buffer_size, 0, &mapped);
    memcpy(mapped, filter_data, filter_buffer_size);
    vkUnmapMemory(vk_device, filter_memory);
    free(filter_data);

    VkDescriptorSet descriptor_set;
    VkDescriptorSetAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk_descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vk_descriptor_set_layout;

    VkResult result = vkAllocateDescriptorSets(vk_device, &alloc_info, &descriptor_set);
    if (result != VK_SUCCESS) goto cleanup;

    VkWriteDescriptorSet descriptor_writes[4] = {0};
    VkDescriptorBufferInfo buffer_infos[4] = {0};

    buffer_infos[0].buffer = param_buffer;
    buffer_infos[0].offset = 0;
    buffer_infos[0].range = param_buffer_size;
    buffer_infos[1].buffer = output_buffer;
    buffer_infos[1].offset = 0;
    buffer_infos[1].range = float_buffer_size;
    buffer_infos[2].buffer = input_buffer;
    buffer_infos[2].offset = 0;
    buffer_infos[2].range = float_buffer_size;
    buffer_infos[3].buffer = filter_buffer;
    buffer_infos[3].offset = 0;
    buffer_infos[3].range = filter_buffer_size;

    for (int i = 0; i < 4; i++) {
        descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[i].dstSet = descriptor_set;
        descriptor_writes[i].dstBinding = i;
        descriptor_writes[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[i].descriptorCount = 1;
        descriptor_writes[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(vk_device, 4, descriptor_writes, 0, NULL);

    VkCommandBufferAllocateInfo cmd_alloc_info = {0};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = vk_command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(vk_device, &cmd_alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(command_buffer, &begin_info);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk_compute_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        vk_pipeline_layout, 0, 1, &descriptor_set, 0, NULL);

    uint32_t work_group_size_x = 16;
    uint32_t work_group_size_y = 16;
    uint32_t group_count_x = (widthHR + work_group_size_x - 1) / work_group_size_x;
    uint32_t group_count_y = (heightHR + work_group_size_y - 1) / work_group_size_y;

    vkCmdDispatch(command_buffer, group_count_x, group_count_y, 1);
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk_queue);

    vkMapMemory(vk_device, output_memory, 0, float_buffer_size, 0, &mapped);
    float* output_data = (float*)mapped;

    for (int row = data->start_row; row < data->end_row; row++) {
        for (int col = MARGIN; col < widthHR - MARGIN; col++) {
            float value = output_data[row * widthHR + col];
            data->predictHR[(row - MARGIN) * (widthHR - 2 * MARGIN) + (col - MARGIN)] = (double)value;
        }
    }
    vkUnmapMemory(vk_device, output_memory);

    vkFreeCommandBuffers(vk_device, vk_command_pool, 1, &command_buffer);
    vkFreeDescriptorSets(vk_device, vk_descriptor_pool, 1, &descriptor_set);

cleanup:
    if (input_buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk_device, input_buffer, NULL);
    if (input_memory != VK_NULL_HANDLE) vkFreeMemory(vk_device, input_memory, NULL);
    if (output_buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk_device, output_buffer, NULL);
    if (output_memory != VK_NULL_HANDLE) vkFreeMemory(vk_device, output_memory, NULL);
    if (param_buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk_device, param_buffer, NULL);
    if (param_memory != VK_NULL_HANDLE) vkFreeMemory(vk_device, param_memory, NULL);
    if (filter_buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk_device, filter_buffer, NULL);
    if (filter_memory != VK_NULL_HANDLE) vkFreeMemory(vk_device, filter_memory, NULL);

    return NULL;
}

#endif // HAVE_VULKAN
