#version 450

layout(binding = 0, set = 0) uniform UniformBufferObject {
    uint widthHR;
    uint heightHR;
    uint widthLR;
    uint heightLR;
    uint filter_size;
    uint margin;
    uint patch_size;
    uint gradient_size;
    uint qangle;
    uint qstrength;
    uint qcoherence;
    uint r;
} ubo;

layout(binding = 1, set = 0) writeonly buffer OutputBuffer {
    float output_data[];
} outBuffer;

layout(binding = 2, set = 0) readonly buffer InputBuffer {
    float input_data[];
} inBuffer;

layout(binding = 3, set = 0) readonly buffer FilterBuffer {
    float h[];
} filter_data;

shared float shared_weighting[81];

const float M_PI        = 3.14159265358979323846;
const uint  PATCH_SIZE  = 11;
const uint  GRAD_SIZE   = 9;
const uint  PATCH_MAR   = 5;
const uint  GRAD_MAR    = 4;
const uint  QANGLE      = 24;
const uint  QSTRENGTH   = 3;
const uint  QCOHERENCE  = 3;
const uint  R           = 2;

void hashkey(float gb[81], out int angle, out int strength, out int coherence) {
    float gy[81];
    float gx[81];

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            int idx = i * 9 + j;
            if (i == 0)
                gy[idx] = gb[(i+1)*9 + j] - gb[idx];
            else if (i == 8)
                gy[idx] = gb[idx] - gb[(i-1)*9 + j];
            else
                gy[idx] = (gb[(i+1)*9 + j] - gb[(i-1)*9 + j]) / 2.0;

            if (j == 0)
                gx[idx] = gb[i*9 + (j+1)] - gb[idx];
            else if (j == 8)
                gx[idx] = gb[idx] - gb[i*9 + (j-1)];
            else
                gx[idx] = (gb[i*9 + (j+1)] - gb[i*9 + (j-1)]) / 2.0;
        }
    }

    float gxx = 0.0, gyy = 0.0, gxy = 0.0;
    for (int i = 0; i < 81; i++) {
        float w = shared_weighting[i];
        gxx += w * gx[i] * gx[i];
        gyy += w * gy[i] * gy[i];
        gxy += w * gx[i] * gy[i];
    }

    float trace = gxx + gyy;
    float disc  = (gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy;
    if (disc < 0.0) disc = 0.0;
    float sq    = sqrt(disc);
    float lam1  = (trace + sq) / 2.0;
    float lam2  = (trace - sq) / 2.0;

    const float eps = 1e-10;
    float vx, vy;
    if (abs(gxy) > eps) {
        vx = lam1 - gyy; vy = gxy;
    } else if (abs(gxx - lam1) > eps) {
        vx = gxy; vy = lam1 - gxx;
    } else {
        vx = 1.0; vy = 0.0;
    }
    float norm = sqrt(vx*vx + vy*vy);
    if (norm > eps) { vx /= norm; vy /= norm; }

    float theta = atan(vy, vx);
    if (theta < 0.0) theta += M_PI;

    float sl1 = sqrt(abs(lam1));
    float sl2 = sqrt(abs(lam2));
    float u   = (sl1 + sl2 < eps) ? 0.0 : (sl1 - sl2) / (sl1 + sl2);

    angle = int(theta / M_PI * float(QANGLE));
    if (angle > int(QANGLE) - 1) angle = int(QANGLE) - 1;
    if (angle < 0) angle = 0;

    strength  = (lam1 < 0.0001) ? 0 : (lam1 > 0.001) ? 2 : 1;
    coherence = (u < 0.25) ? 0 : (u > 0.5) ? 2 : 1;
}

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uvec2 gPos = gl_GlobalInvocationID.xy;
    uvec2 lPos = gl_LocalInvocationID.xy;

    // One thread initializes and normalizes the shared Gaussian weighting
    if (lPos.x == 0 && lPos.y == 0) {
        float sum   = 0.0;
        float sigma = 2.0;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                int   m = i - 4, n = j - 4;
                float v = exp(-(float(m*m + n*n)) / (2.0 * sigma * sigma));
                shared_weighting[i * 9 + j] = v;
                sum += v;
            }
        }
        for (int k = 0; k < 81; k++)
            shared_weighting[k] /= sum;
    }
    barrier();

    if (gPos.x < ubo.margin || gPos.x >= ubo.widthHR  - ubo.margin ||
        gPos.y < ubo.margin || gPos.y >= ubo.heightHR - ubo.margin)
        return;

    // Load patch (11x11)
    float local_patch[121];
    for (int dy = 0; dy < 11; dy++) {
        for (int dx = 0; dx < 11; dx++) {
            int y = int(gPos.y) - int(PATCH_MAR) + dy;
            int x = int(gPos.x) - int(PATCH_MAR) + dx;
            local_patch[dy * 11 + dx] = (y >= 0 && y < int(ubo.heightHR) &&
                                          x >= 0 && x < int(ubo.widthHR))
                ? inBuffer.input_data[y * int(ubo.widthHR) + x]
                : 0.0;
        }
    }

    // Load gradient block (9x9)
    float local_grad[81];
    for (int dy = 0; dy < 9; dy++) {
        for (int dx = 0; dx < 9; dx++) {
            int y = int(gPos.y) - int(GRAD_MAR) + dy;
            int x = int(gPos.x) - int(GRAD_MAR) + dx;
            local_grad[dy * 9 + dx] = (y >= 0 && y < int(ubo.heightHR) &&
                                        x >= 0 && x < int(ubo.widthHR))
                ? inBuffer.input_data[y * int(ubo.widthHR) + x]
                : 0.0;
        }
    }

    int angle, strength, coherence;
    hashkey(local_grad, angle, strength, coherence);

    int pixeltype  = (int(gPos.y - ubo.margin) % int(R)) * int(R)
                   + (int(gPos.x - ubo.margin) % int(R));
    int filter_idx = ((angle * int(QSTRENGTH) + strength) * int(QCOHERENCE) + coherence)
                   * int(R * R * PATCH_SIZE * PATCH_SIZE)
                   + pixeltype * int(PATCH_SIZE * PATCH_SIZE);

    float result = 0.0;
    for (int i = 0; i < 121; i++)
        result += local_patch[i] * filter_data.h[uint(filter_idx + i)];

    outBuffer.output_data[int(gPos.y) * int(ubo.widthHR) + int(gPos.x)] = result;
}