#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <error-code.hpp>
#include <get-kernel-source.hpp>

inline constexpr std::size_t vector_length = 1024;

int main() {
    cl_int err;

    cl_uint platform_cnt;
    err = clGetPlatformIDs(0, nullptr, &platform_cnt);
    check_error(err);
    if(!platform_cnt) {
        std::cerr << "Found 0 platforms.\n";
        return 1;
    }

    // using only first platform
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    check_error(err);

    cl_uint device_ids_count;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_ids_count);
    check_error(err);
    if(!device_ids_count) {
        std::cerr << "\tNo devices found on this platform.\n";
        return 1;
    }
    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, nullptr);
    check_error(err);

    std::size_t device_name_length;
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_name_length);
    check_error(err);
    std::string device_name(device_name_length, '\0');
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_length, device_name.data(), nullptr);
    check_error(err);
    std::cout << "Using device " << device_name << '\n';

    auto *a = new float[vector_length];
    auto *b = new float[vector_length];
    auto *c = new float[vector_length];

    for(std::size_t i = 0; i < vector_length; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    check_error(err);

    cl_queue_properties properties[] { 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
    check_error(err);

    char const *kernel_source = get_kernel_source("../kernel/opencl-vector-addition.cl");
    auto program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    check_error(err);
    delete[] kernel_source;

    std::cout << "Building program...";
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        check_error(err);
        std::string log(log_size, '\0');
        err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << log << std::endl;
        return 1;
    }
    check_error(err);
    std::cout << "Done!\n";

    auto kernel = clCreateKernel(program, "vector_add", &err);
    check_error(err);

    auto *da = clCreateBuffer(context, CL_MEM_READ_ONLY, vector_length * sizeof(float), nullptr, &err);
    check_error(err);
    auto *db = clCreateBuffer(context, CL_MEM_READ_ONLY, vector_length * sizeof(float), nullptr, &err);
    check_error(err);
    auto *dc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_length * sizeof(float), nullptr, &err);
    check_error(err);

    err = clEnqueueWriteBuffer(command_queue, da, CL_TRUE, 0, vector_length * sizeof(float), a, 0, nullptr, nullptr);
    check_error(err);
    err = clEnqueueWriteBuffer(command_queue, db, CL_TRUE, 0, vector_length * sizeof(float), a, 0, nullptr, nullptr);
    check_error(err);

    err = clSetKernelArg(kernel, 0, sizeof(da), &da);
    check_error(err);
    err = clSetKernelArg(kernel, 1, sizeof(db), &db);
    check_error(err);
    err = clSetKernelArg(kernel, 2, sizeof(dc), &dc);
    check_error(err);
    err = clSetKernelArg(kernel, 3, sizeof(unsigned long), &vector_length);

    std::size_t const local_work_size = 64;
    std::size_t const global_work_size = (vector_length + local_work_size - 1) / local_work_size * local_work_size;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    check_error(err);

    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, dc, CL_TRUE, 0, vector_length * sizeof(float), c, 0, nullptr, nullptr);

    char s[3] { '\0', ' ', '\0' };
    for(std::size_t i = 0; i < vector_length; i++) {
        std::cout << s << c[i];
        if(i % 20 == 19) s[0] = '\n', s[1] = '\0';
        else s[0] = ',', s[1] = ' ';
    }

    clReleaseMemObject(da);
    clReleaseMemObject(db);
    clReleaseMemObject(dc);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    delete[] a;
    delete[] b;
    delete[] c;
}