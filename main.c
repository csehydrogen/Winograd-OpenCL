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
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);
        char *log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);
        log[log_size] = 0;
        printf("Compile error:\n%s\n", log);
        free(log);
    }
    CHECK_ERROR(err);

    return program;
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

    cl_kernel kernel;
    kernel = clCreateKernel(program, "my_kernel", &err);
    CHECK_ERROR(err);

    timer_start(0);

    int n = 256;
    cl_mem mem_a = clCreateBuffer(context, 0, sizeof(float) * n, NULL, &err);
    CHECK_ERROR(err);
    cl_mem mem_b = clCreateBuffer(context, 0, sizeof(float) * n, NULL, &err);
    CHECK_ERROR(err);
    cl_mem mem_c = clCreateBuffer(context, 0, sizeof(float) * n, NULL, &err);
    CHECK_ERROR(err);
    float *a = (float*)malloc(sizeof(float) * n);
    float *b = (float*)malloc(sizeof(float) * n);
    float *c = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) {
        a[i] = b[i] = i;
    }
    err = clEnqueueWriteBuffer(queue, mem_a, CL_FALSE, 0, sizeof(float) * n, a, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, mem_b, CL_FALSE, 0, sizeof(float) * n, b, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_c);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    CHECK_ERROR(err);
    const size_t gws[1] = {n}, lws[1] = {256};
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue, mem_c, CL_FALSE, 0, sizeof(float) * n, c, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clFinish(queue);
    CHECK_ERROR(err);
    for (int i = 0; i < n; ++i) {
        printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", i, a[i], i, b[i], i, c[i]);
    }
    free(a);
    free(b);
    free(c);

    timer_end(0, "Compute");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
