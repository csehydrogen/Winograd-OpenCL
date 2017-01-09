#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "timer.h"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;

cl_program create_and_build_program(cl_context context, cl_device_id device, const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("Failed to open %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    size_t source_size = ftell(file);
    rewind(file);
    char *source_code = (char*)malloc(source_size + 1);
    fread(source_code, sizeof(char), source_size, file);
    source_code[source_size] = '\0';
    fclose(file);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
    CHECK_ERROR(err);
    free(source_code);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        char *log = (char*)malloc(log_size + 1);
        CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
        log[log_size] = 0;
        printf("Compile error:\n%s\n", log);
        free(log);
    }
    CHECK_ERROR(err);

    return program;
}

int _ceil(int x, int y) {
    return (x + y - 1) / y * y;
}

int _ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

void fillData(float *d, int n) {
    for (int i = 0; i < n; ++i) {
        d[i] = rand() % 2;
    }
}

void printData(float *d, int N, int C, int H, int W, const char *name) {
    printf("%s.shape = (%d, %d, %d, %d)\n", name, N, C, H, W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            printf("(%d, %d, :, :) =\n", n, c);
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    printf("%f ", d[((n * C + c) * H + h) * W + w]);
                }
                printf("\n");
            }
        }
    }
}

void convolution_cpu(float *inputs, float *outputs, float *filters, float *bias, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int p = 0; p < P; ++p) {
                for (int q = 0; q < Q; ++q) {
                    float x = bias[k];
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int h = p + r - pad, w = q + s - pad;
                                if (0 <= h && h < H && 0 <= w && w < W) {
                                    x += inputs[((n * C + c) * H + h) * W + w] * filters[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    outputs[((n * K + k) * P + p) * Q + q] = x;
                }
            }
        }
    }
}

void convolution_wino(float *inputs, float *outputs, float *filters, float *bias, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad, cl_context context, cl_command_queue queue, cl_program program) {
    cl_kernel kernel0 = clCreateKernel(program, "NCHW2CHWN", &err);
    CHECK_ERROR(err);
    cl_kernel kernel1 = clCreateKernel(program, "winograd_2x2_3x3_32x32", &err);
    CHECK_ERROR(err);
    cl_kernel kernel2 = clCreateKernel(program, "CHWN2NCHW", &err);
    CHECK_ERROR(err);

    cl_mem inputs_dev = clCreateBuffer(context, 0, sizeof(float) * (N * C * H * W), NULL, &err);
    CHECK_ERROR(err);
    cl_mem inputs_CHWN_dev = clCreateBuffer(context, 0, sizeof(float) * (N * C * H * W), NULL, &err);
    CHECK_ERROR(err);
    cl_mem outputs_dev = clCreateBuffer(context, 0, sizeof(float) * (N * K * P * Q), NULL, &err);
    CHECK_ERROR(err);
    cl_mem outputs_CHWN_dev = clCreateBuffer(context, 0, sizeof(float) * (N * K * P * Q), NULL, &err);
    CHECK_ERROR(err);
    cl_mem filters_dev = clCreateBuffer(context, 0, sizeof(float) * (K * C * R * S), NULL, &err);
    CHECK_ERROR(err);
    cl_mem filters_CHWN_dev = clCreateBuffer(context, 0, sizeof(float) * (K * C * R * S), NULL, &err);
    CHECK_ERROR(err);
    cl_mem bias_dev = clCreateBuffer(context, 0, sizeof(float) * (K), NULL, &err);
    CHECK_ERROR(err);

    timer_start(0);
    CHECK_ERROR(clEnqueueWriteBuffer(queue, inputs_dev, CL_TRUE, 0, sizeof(float) * (N * C * H * W), inputs, 0, NULL, NULL));
    CHECK_ERROR(clEnqueueWriteBuffer(queue, filters_dev, CL_TRUE, 0, sizeof(float) * (K * C * R * S), filters, 0, NULL, NULL));
    CHECK_ERROR(clEnqueueWriteBuffer(queue, bias_dev, CL_TRUE, 0, sizeof(float) * (K), bias, 0, NULL, NULL));
    timer_end(0, "wino WriteBuffer");

    {
        timer_start(0);
        CHECK_ERROR(clSetKernelArg(kernel0, 0, sizeof(cl_mem), &inputs_dev));
        CHECK_ERROR(clSetKernelArg(kernel0, 1, sizeof(cl_mem), &inputs_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel0, 2, sizeof(int), &N));
        CHECK_ERROR(clSetKernelArg(kernel0, 3, sizeof(int), &C));
        CHECK_ERROR(clSetKernelArg(kernel0, 4, sizeof(int), &H));
        CHECK_ERROR(clSetKernelArg(kernel0, 5, sizeof(int), &W));
        size_t gws[1] = {_ceil(N * C * H * W, 256)};
        size_t lws[1] = {256};
        CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel0, 1, NULL, gws, lws, 0, NULL, NULL));
        clFinish(queue);
        timer_end(0, "wino input transform");
    }

    {
        timer_start(0);
        CHECK_ERROR(clSetKernelArg(kernel0, 0, sizeof(cl_mem), &filters_dev));
        CHECK_ERROR(clSetKernelArg(kernel0, 1, sizeof(cl_mem), &filters_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel0, 2, sizeof(int), &K));
        CHECK_ERROR(clSetKernelArg(kernel0, 3, sizeof(int), &C));
        CHECK_ERROR(clSetKernelArg(kernel0, 4, sizeof(int), &R));
        CHECK_ERROR(clSetKernelArg(kernel0, 5, sizeof(int), &S));
        size_t gws[1] = {_ceil(K * C * R * S, 256)};
        size_t lws[1] = {256};
        CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel0, 1, NULL, gws, lws, 0, NULL, NULL));
        clFinish(queue);
        timer_end(0, "wino filter transform");
    }

    {
        timer_start(0);
        int TP = _ceil_div(P, 2), TQ = _ceil_div(Q, 2);
        int BN = 1, BK = 1;
        int TPmask, TPwidth, TPshift, TQmask, TQwidth, TQshift, Nmask, Nwidth;
        if (N < 2)
            TPmask = 0x18, TPwidth = 2, TPshift = 3, TQmask = 0x07, TQwidth = 3, TQshift = 0, Nmask = 0x00, Nwidth = 0;
        else if (N < 4)
            TPmask = 0x18, TPwidth = 2, TPshift = 3, TQmask = 0x06, TQwidth = 2, TQshift = 1, Nmask = 0x01, Nwidth = 1;
        else if (N < 8)
            TPmask = 0x10, TPwidth = 1, TPshift = 4, TQmask = 0x0c, TQwidth = 2, TQshift = 2, Nmask = 0x03, Nwidth = 2;
        else if (N < 16)
            TPmask = 0x10, TPwidth = 1, TPshift = 4, TQmask = 0x08, TQwidth = 1, TQshift = 3, Nmask = 0x07, Nwidth = 3;
        else if (N < 32)
            TPmask = 0x00, TPwidth = 0, TPshift = 0, TQmask = 0x10, TQwidth = 1, TQshift = 4, Nmask = 0x0f, Nwidth = 4;
        else
            TPmask = 0x00, TPwidth = 0, TPshift = 0, TQmask = 0x00, TQwidth = 0, TQshift = 0, Nmask = 0x1f, Nwidth = 5;
        int TK = _ceil_div(K, 32), TN = _ceil_div(N, 1 << Nwidth);

        CHECK_ERROR(clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputs_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel1, 1, sizeof(cl_mem), &outputs_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel1, 2, sizeof(cl_mem), &filters_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel1, 3, sizeof(cl_mem), &bias_dev));
        CHECK_ERROR(clSetKernelArg(kernel1, 4, sizeof(int), &N));
        CHECK_ERROR(clSetKernelArg(kernel1, 5, sizeof(int), &C));
        CHECK_ERROR(clSetKernelArg(kernel1, 6, sizeof(int), &H));
        CHECK_ERROR(clSetKernelArg(kernel1, 7, sizeof(int), &W));
        CHECK_ERROR(clSetKernelArg(kernel1, 8, sizeof(int), &K));
        CHECK_ERROR(clSetKernelArg(kernel1, 9, sizeof(int), &P));
        CHECK_ERROR(clSetKernelArg(kernel1, 10, sizeof(int), &Q));
        CHECK_ERROR(clSetKernelArg(kernel1, 11, sizeof(int), &pad));
        CHECK_ERROR(clSetKernelArg(kernel1, 12, sizeof(int), &TP));
        CHECK_ERROR(clSetKernelArg(kernel1, 13, sizeof(int), &TQ));
        CHECK_ERROR(clSetKernelArg(kernel1, 14, sizeof(int), &BN));
        CHECK_ERROR(clSetKernelArg(kernel1, 15, sizeof(int), &BK));
        CHECK_ERROR(clSetKernelArg(kernel1, 16, sizeof(int), &TPmask));
        CHECK_ERROR(clSetKernelArg(kernel1, 17, sizeof(int), &TPwidth));
        CHECK_ERROR(clSetKernelArg(kernel1, 18, sizeof(int), &TPshift));
        CHECK_ERROR(clSetKernelArg(kernel1, 19, sizeof(int), &TQmask));
        CHECK_ERROR(clSetKernelArg(kernel1, 20, sizeof(int), &TQwidth));
        CHECK_ERROR(clSetKernelArg(kernel1, 21, sizeof(int), &TQshift));
        CHECK_ERROR(clSetKernelArg(kernel1, 22, sizeof(int), &Nmask));
        CHECK_ERROR(clSetKernelArg(kernel1, 23, sizeof(int), &Nwidth));
        size_t gws[3] = {TP * TQ * BN * BK * 256, TK / BK, TN / BN};
        size_t lws[3] = {256, 1, 1};
        CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel1, 3, NULL, gws, lws, 0, NULL, NULL));
        clFinish(queue);
        timer_end(0, "wino");
    }

    {
        timer_start(0);
        CHECK_ERROR(clSetKernelArg(kernel2, 0, sizeof(cl_mem), &outputs_CHWN_dev));
        CHECK_ERROR(clSetKernelArg(kernel2, 1, sizeof(cl_mem), &outputs_dev));
        CHECK_ERROR(clSetKernelArg(kernel2, 2, sizeof(int), &N));
        CHECK_ERROR(clSetKernelArg(kernel2, 3, sizeof(int), &K));
        CHECK_ERROR(clSetKernelArg(kernel2, 4, sizeof(int), &P));
        CHECK_ERROR(clSetKernelArg(kernel2, 5, sizeof(int), &Q));
        size_t gws[1] = {_ceil(N * K * P * Q, 256)};
        size_t lws[1] = {256};
        CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, gws, lws, 0, NULL, NULL));
        clFinish(queue);
        timer_end(0, "wino output transform");
    }

    timer_start(0);
    CHECK_ERROR(clEnqueueReadBuffer(queue, outputs_dev, CL_TRUE, 0, sizeof(float) * (N * K * P * Q), outputs, 0, NULL, NULL));
    timer_end(0, "wino ReadBuffer");

    clReleaseKernel(kernel0);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
}

int main() {
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    cl_program program = create_and_build_program(context, device, "kernel.cl");

    int N = 1, C = 1, H = 4, W = 4, K = 1, P = 2, Q = 2, R = 3, S = 3, pad = 0;
    float *inputs = (float*)malloc(sizeof(float) * (N * C * H * W));
    float *filters = (float*)malloc(sizeof(float) * (K * C * R * S));
    float *bias = (float*)malloc(sizeof(float) * (K));
    fillData(inputs, N * C * H * W);
    fillData(filters, K * C * R * S);
    fillData(bias, K);
    printData(inputs, N, C, H, W, "inputs");
    printData(filters, K, C, R, S, "filters");
    printData(bias, 1, 1, 1, K, "bias");

    float *outputs_cpu = (float*)malloc(sizeof(float) * (N * K * P * Q));
    float *outputs_wino = (float*)malloc(sizeof(float) * (N * K * P * Q));
    convolution_cpu(inputs, outputs_cpu, filters, bias, N, C, H, W, K, P, Q, R, S, pad);
    printData(outputs_cpu, N, K, P, Q, "outputs_cpu");
    convolution_wino(inputs, outputs_wino, filters, bias, N, C, H, W, K, P, Q, R, S, pad, context, queue, program);
    printData(outputs_wino, N, K, P, Q, "outputs_wino");

    free(inputs);
    free(filters);
    free(bias);
    free(outputs_cpu);
    free(outputs_wino);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
