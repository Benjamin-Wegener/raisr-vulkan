#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <jpeglib.h>

// Vulkan detection
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

// Parameters
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

// Function prototypes
int read_jpeg_file(const char* filename, unsigned char** y_channel, unsigned char** cb_channel, unsigned char** cr_channel, int* height, int* width);
int write_jpeg_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width);
void upscale_bilinear_channel(unsigned char* input, unsigned char* output, int heightLR, int widthLR, int heightHR, int widthHR);
void* process_rows(void* arg);
void* process_rows_vulkan(void* arg);
void calculate_gradient(unsigned char* block, int size, double** gy, double** gx);
void hashkey(double* gradientblock, int Qangle, double* weighting, int* angle, int* strength, int* coherence);
double dot_product(double* a, double* b, int size);
int init_vulkan();
void cleanup_vulkan();

// Vulkan variables
#if HAVE_VULKAN
static int vulkan_available = 0;
static VkInstance vk_instance;
static VkPhysicalDevice vk_physical_device;
static VkDevice vk_device;
static VkQueue vk_queue;
static VkCommandPool vk_command_pool;
#endif

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>
", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    // Try to initialize Vulkan
    #if HAVE_VULKAN
    vulkan_available = init_vulkan();
    if (vulkan_available) {
        printf("Vulkan acceleration enabled
");
    } else {
        printf("Vulkan not available, using CPU processing
");
    }
    #else
    printf("Vulkan support not compiled in, using CPU processing
");
    #endif

    // Read input image in YCbCr color space
    unsigned char* y_channel = NULL;
    unsigned char* cb_channel = NULL;
    unsigned char* cr_channel = NULL;
    int height, width;
    
    if (!read_jpeg_file(input_filename, &y_channel, &cb_channel, &cr_channel, &height, &width)) {
        fprintf(stderr, "Error reading input image
");
        return 1;
    }

    printf("Input image size: %dx%d
", width, height);

    // Upscale (bilinear interpolation)
    int heightLR = height;
    int widthLR = width;
    int heightHR = heightLR * 2;
    int widthHR = widthLR * 2;
    
    unsigned char* upscaled_y = malloc(heightHR * widthHR);
    unsigned char* upscaled_cb = malloc(heightHR * widthHR);
    unsigned char* upscaled_cr = malloc(heightHR * widthHR);
    
    if (!upscaled_y || !upscaled_cb || !upscaled_cr) {
        fprintf(stderr, "Memory allocation failed
");
        free(y_channel);
        free(cb_channel);
        free(cr_channel);
        if (upscaled_y) free(upscaled_y);
        if (upscaled_cb) free(upscaled_cb);
        if (upscaled_cr) free(upscaled_cr);
        return 1;
    }
    
    upscale_bilinear_channel(y_channel, upscaled_y, heightLR, widthLR, heightHR, widthHR);
    upscale_bilinear_channel(cb_channel, upscaled_cb, heightLR, widthLR, heightHR, widthHR);
    upscale_bilinear_channel(cr_channel, upscaled_cr, heightLR, widthLR, heightHR, widthHR);
    
    // Load filter from binary file
    double* h = NULL;
    int filter_size = QANGLE * QSTRENGTH * QCOHERENCE * R * R * PATCH_SIZE * PATCH_SIZE;
    
    FILE* filter_file = fopen("filter.bin", "rb");
    if (!filter_file) {
        fprintf(stderr, "Could not open filter file. Using default averaging filter.
");
        h = malloc(filter_size * sizeof(double));
        if (!h) {
            fprintf(stderr, "Memory allocation failed
");
            free(y_channel);
            free(cb_channel);
            free(cr_channel);
            free(upscaled_y);
            free(upscaled_cb);
            free(upscaled_cr);
            return 1;
        }
        
        // Initialize filter with some values (in a real implementation, load from trained data)
        for (int i = 0; i < filter_size; i++) {
            h[i] = 1.0 / (PATCH_SIZE * PATCH_SIZE);  // Simple averaging filter
        }
    } else {
        // Read filter size from file
        unsigned int file_filter_size;
        fread(&file_filter_size, sizeof(unsigned int), 1, filter_file);
        
        if (file_filter_size != filter_size) {
            fprintf(stderr, "Filter size mismatch. Expected %d, got %d
", filter_size, file_filter_size);
            fclose(filter_file);
            free(y_channel);
            free(cb_channel);
            free(cr_channel);
            free(upscaled_y);
            free(upscaled_cb);
            free(upscaled_cr);
            return 1;
        }
        
        h = malloc(filter_size * sizeof(double));
        if (!h) {
            fprintf(stderr, "Memory allocation failed
");
            fclose(filter_file);
            free(y_channel);
            free(cb_channel);
            free(cr_channel);
            free(upscaled_y);
            free(upscaled_cb);
            free(upscaled_cr);
            return 1;
        }
        
        // Read filter data
        size_t elements_read = fread(h, sizeof(double), filter_size, filter_file);
        fclose(filter_file);
        
        if (elements_read != filter_size) {
            fprintf(stderr, "Error reading filter data. Expected %d elements, read %zu
", filter_size, elements_read);
            free(h);
            free(y_channel);
            free(cb_channel);
            free(cr_channel);
            free(upscaled_y);
            free(upscaled_cb);
            free(upscaled_cr);
            return 1;
        }
        
        printf("Successfully loaded trained filter with %d elements
", filter_size);
    }
    
    // Calculate predictHR pixels (only for Y channel)
    int predictHR_height = heightHR - 2 * MARGIN;
    int predictHR_width = widthHR - 2 * MARGIN;
    double* predictHR = calloc(predictHR_height * predictHR_width, sizeof(double));
    if (!predictHR) {
        fprintf(stderr, "Memory allocation failed
");
        free(y_channel);
        free(cb_channel);
        free(cr_channel);
        free(upscaled_y);
        free(upscaled_cb);
        free(upscaled_cr);
        free(h);
        return 1;
    }
    
    // Timing
    clock_t start_time = clock();
    
    // Process image with multithreading (only Y channel gets RAISR processing)
    int num_threads = 4;  // Adjust based on your system
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
        
        // Use Vulkan processing if available, otherwise CPU
        #if HAVE_VULKAN
        if (vulkan_available) {
            pthread_create(&threads[i], NULL, process_rows_vulkan, &thread_data[i]);
        } else {
            pthread_create(&threads[i], NULL, process_rows, &thread_data[i]);
        }
        #else
        pthread_create(&threads[i], NULL, process_rows, &thread_data[i]);
        #endif
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("Processing time: %f seconds
", cpu_time_used);
    
    // Scale back to [0,255] for Y channel only
    for (int row = MARGIN; row < heightHR - MARGIN; row++) {
        for (int col = MARGIN; col < widthHR - MARGIN; col++) {
            double value = predictHR[(row - MARGIN) * predictHR_width + (col - MARGIN)] * 255.0;
            upscaled_y[row * widthHR + col] = (unsigned char)fmin(fmax(value, 0.0), 255.0);
        }
    }
    
    // Write output image
    if (!write_jpeg_file(output_filename, upscaled_y, upscaled_cb, upscaled_cr, heightHR, widthHR)) {
        fprintf(stderr, "Error writing output image
");
        free(y_channel);
        free(cb_channel);
        free(cr_channel);
        free(upscaled_y);
        free(upscaled_cb);
        free(upscaled_cr);
        free(h);
        free(predictHR);
        free(threads);
        free(thread_data);
        return 1;
    }
    
    printf("Output image written to %s
", output_filename);
    
    // Cleanup Vulkan
    #if HAVE_VULKAN
    if (vulkan_available) {
        cleanup_vulkan();
    }
    #endif
    
    // Cleanup
    free(y_channel);
    free(cb_channel);
    free(cr_channel);
    free(upscaled_y);
    free(upscaled_cb);
    free(upscaled_cr);
    free(h);
    free(predictHR);
    free(threads);
    free(thread_data);
    
    return 0;
}

// Read JPEG file and convert to YCbCr
int read_jpeg_file(const char* filename, unsigned char** y_channel, unsigned char** cb_channel, unsigned char** cr_channel, int* height, int* width) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    FILE* infile = fopen(filename, "rb");
    if (!infile) {
        return 0;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    
    // Set output color space to YCbCr
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

// Write JPEG file from YCbCr channels
int write_jpeg_file(const char* filename, unsigned char* y_channel, unsigned char* cb_channel, unsigned char* cr_channel, int height, int width) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) {
        return 0;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_YCbCr;
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    cinfo.comp_info[0].h_samp_factor = 1; // Y
    cinfo.comp_info[0].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1; // Cb
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1; // Cr
    cinfo.comp_info[2].v_samp_factor = 1;
    
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

// Bilinear interpolation upscaling
void upscale_bilinear_channel(unsigned char* input, unsigned char* output, int heightLR, int widthLR, int heightHR, int widthHR) {
    for (int y = 0; y < heightHR; y++) {
        for (int x = 0; x < widthHR; x++) {
            // Map HR coordinates to LR coordinates
            double lr_y = (double)y / 2.0;
            double lr_x = (double)x / 2.0;
            
            // Get integer and fractional parts
            int y1 = (int)floor(lr_y);
            int x1 = (int)floor(lr_x);
            int y2 = y1 + 1;
            int x2 = x1 + 1;
            
            // Handle boundary conditions
            if (y2 >= heightLR) y2 = heightLR - 1;
            if (x2 >= widthLR) x2 = widthLR - 1;
            
            // Get the four neighboring pixels
            double Q11 = input[y1 * widthLR + x1];
            double Q21 = input[y1 * widthLR + x2];
            double Q12 = input[y2 * widthLR + x1];
            double Q22 = input[y2 * widthLR + x2];
            
            // Calculate weights
            double wx = lr_x - x1;
            double wy = lr_y - y1;
            
            // Bilinear interpolation
            double result = Q11 * (1 - wx) * (1 - wy) +
                           Q21 * wx * (1 - wy) +
                           Q12 * (1 - wx) * wy +
                           Q22 * wx * wy;
            
            output[y * widthHR + x] = (unsigned char)result;
        }
    }
}

// Process rows function for multithreading (CPU version)
void* process_rows(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    // Preprocessing normalized Gaussian matrix W for hashkey calculation
    double* weighting = malloc(GRADIENT_SIZE * GRADIENT_SIZE * sizeof(double));
    if (!weighting) {
        return NULL;
    }
    
    // Simplified Gaussian calculation (sigma = 2)
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
    
    // Normalize
    for (int i = 0; i < GRADIENT_SIZE * GRADIENT_SIZE; i++) {
        weighting[i] /= sum;
    }
    
    // Process rows
    for (int row = data->start_row; row < data->end_row; row++) {
        for (int col = MARGIN; col < data->widthHR - MARGIN; col++) {
            // Get patch from Y channel (upscaled)
            double* patch = malloc(PATCH_SIZE * PATCH_SIZE * sizeof(double));
            if (!patch) {
                free(weighting);
                return NULL;
            }
            
            int patch_idx = 0;
            for (int i = row - PATCH_MARGIN; i <= row + PATCH_MARGIN; i++) {
                for (int j = col - PATCH_MARGIN; j <= col + PATCH_MARGIN; j++) {
                    patch[patch_idx++] = data->y_channel[i * data->widthHR + j] / 255.0;
                }
            }
            
            // Get gradient block from Y channel (upscaled)
            double* gradientblock = malloc(GRADIENT_SIZE * GRADIENT_SIZE * sizeof(double));
            if (!gradientblock) {
                free(patch);
                free(weighting);
                return NULL;
            }
            
            int grad_idx = 0;
            for (int i = row - GRADIENT_MARGIN; i <= row + GRADIENT_MARGIN; i++) {
                for (int j = col - GRADIENT_MARGIN; j <= col + GRADIENT_MARGIN; j++) {
                    gradientblock[grad_idx++] = data->y_channel[i * data->widthHR + j] / 255.0;
                }
            }
            
            // Calculate hashkey
            int angle, strength, coherence;
            hashkey(gradientblock, QANGLE, weighting, &angle, &strength, &coherence);
            
            // Get pixel type
            int pixeltype = ((row - MARGIN) % R) * R + ((col - MARGIN) % R);
            
            // Calculate filter index
            int filter_idx = ((angle * QSTRENGTH + strength) * QCOHERENCE + coherence) * R * R * PATCH_SIZE * PATCH_SIZE +
                            pixeltype * PATCH_SIZE * PATCH_SIZE;
            
            // Apply filter
            double result = 0.0;
            for (int i = 0; i < PATCH_SIZE * PATCH_SIZE; i++) {
                result += patch[i] * data->h[filter_idx + i];
            }
            
            // Store result
            data->predictHR[(row - MARGIN) * (data->widthHR - 2 * MARGIN) + (col - MARGIN)] = result;
            
            free(patch);
            free(gradientblock);
        }
    }
    
    free(weighting);
    return NULL;
}

// Process rows function for multithreading (Vulkan version)
void* process_rows_vulkan(void* arg) {
    // For now, fall back to CPU processing
    // In a full implementation, this would offload work to the GPU
    return process_rows(arg);
}

// Simplified hashkey calculation
void hashkey(double* gradientblock, int Qangle, double* weighting, int* angle, int* strength, int* coherence) {
    // Calculate gradient
    double* gy = malloc(GRADIENT_SIZE * GRADIENT_SIZE * sizeof(double));
    double* gx = malloc(GRADIENT_SIZE * GRADIENT_SIZE * sizeof(double));
    
    for (int i = 0; i < GRADIENT_SIZE; i++) {
        for (int j = 0; j < GRADIENT_SIZE; j++) {
            int idx = i * GRADIENT_SIZE + j;
            
            // Calculate gradient using central difference
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
    
    // Simplified SVD calculation
    // In a real implementation, you would use a proper SVD algorithm
    // For now, we'll use a simplified approach to estimate the dominant direction
    
    // Calculate structure tensor components
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
    
    // Calculate eigenvalues and eigenvectors
    // For a 2x2 matrix [[gxx, gxy], [gxy, gyy]]
    // The eigenvalues are: lambda = (gxx + gyy ± sqrt((gxx - gyy)^2 + 4*gxy^2)) / 2
    double trace = gxx + gyy;
    double discriminant = (gxx - gyy) * (gxx - gyy) + 4 * gxy * gxy;
    
    if (discriminant < 0) discriminant = 0;  // Avoid numerical issues
    double sqrt_discriminant = sqrt(discriminant);
    
    double lambda1 = (trace + sqrt_discriminant) / 2.0;
    double lambda2 = (trace - sqrt_discriminant) / 2.0;
    
    // Eigenvector corresponding to lambda1
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
    
    // Normalize eigenvector
    double norm = sqrt(vx * vx + vy * vy);
    if (norm > 1e-10) {
        vx /= norm;
        vy /= norm;
    }
    
    // Calculate theta (angle)
    double theta = atan2(vy, vx);
    if (theta < 0) theta += M_PI;
    
    // Calculate strength (lamda)
    double lamda = lambda1;
    
    // Calculate coherence (u)
    double sqrt_lamda1 = sqrt(fabs(lambda1));
    double sqrt_lamda2 = sqrt(fabs(lambda2));
    double u = (sqrt_lamda1 + sqrt_lamda2 == 0) ? 0 : (sqrt_lamda1 - sqrt_lamda2) / (sqrt_lamda1 + sqrt_lamda2);
    
    // Quantize
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
    
    // Bound the output to the desired ranges
    if (*angle > Qangle - 1) *angle = Qangle - 1;
    else if (*angle < 0) *angle = 0;
    
    free(gy);
    free(gx);
}

// Calculate dot product
double dot_product(double* a, double* b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Initialize Vulkan
int init_vulkan() {
    #if HAVE_VULKAN
    // This is a simplified initialization
    // A full implementation would be much more complex
    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "RAISR Vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "RAISR";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&create_info, NULL, &vk_instance);
    if (result != VK_SUCCESS) {
        return 0;
    }

    // Get physical device
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk_instance, &device_count, NULL);
    if (device_count == 0) {
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(vk_instance, &device_count, devices);
    vk_physical_device = devices[0];  // Use first device
    free(devices);

    // Create logical device
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, NULL);
    if (queue_family_count == 0) {
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    VkQueueFamilyProperties* queue_families = malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, queue_families);

    // Find a queue family that supports compute
    uint32_t queue_family_index = 0;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_index = i;
            break;
        }
    }
    free(queue_families);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index;
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

    // Get queue
    vkGetDeviceQueue(vk_device, queue_family_index, 0, &vk_queue);

    // Create command pool
    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(vk_device, &pool_info, NULL, &vk_command_pool);
    if (result != VK_SUCCESS) {
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        return 0;
    }

    return 1;
    #else
    return 0;
    #endif
}

// Cleanup Vulkan
void cleanup_vulkan() {
    #if HAVE_VULKAN
    if (vulkan_available) {
        vkDestroyCommandPool(vk_device, vk_command_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        vulkan_available = 0;
    }
    #endif
}