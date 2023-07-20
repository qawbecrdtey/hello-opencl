#include <iostream>
#include <exception>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "error-code.hpp"

int main() {
    cl_int err;

    // Get all possible platform ids
    cl_uint platform_count;
    err = clGetPlatformIDs(0, nullptr, &platform_count);
    check_error(err);
    if(!platform_count) {
        std::cerr << "Found 0 platforms.\n";
        return 1;
    }
    else {
        std::cout << "Found " << platform_count << " platform(s).\n";
    }
    std::vector<cl_platform_id> platform_ids(platform_count);
    err = clGetPlatformIDs(platform_count, platform_ids.data(), nullptr);
    check_error(err);

    std::uint64_t platform_idx = 0;
    for (cl_platform_id platform_id: platform_ids) {
        std::cout << "PLATFORM #" << platform_idx << '\n';

        // CL_PLATFORM_NAME
        std::size_t platform_name_length;
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);
        check_error(err);
        std::string platform_name(platform_name_length, '\0');
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name_length, platform_name.data(), nullptr);
        check_error(err);
        std::cout << "\tCL_PLATFORM_NAME: " << platform_name << '\n';

        // CL_PLATFORM_VENDOR
        std::size_t platform_vendor_length;
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, nullptr, &platform_vendor_length);
        check_error(err);
        std::string platform_vendor(platform_vendor_length, '\0');
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, platform_vendor_length, platform_vendor.data(), nullptr);
        check_error(err);
        std::cout << "\tCL_PLATFORM_VENDOR: " << platform_vendor << '\n';

        // CL_PLATFORM_VERSION
        std::size_t platform_version_length;
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, nullptr, &platform_version_length);
        check_error(err);
        std::string platform_version(platform_version_length, '\0');
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, platform_version_length, platform_version.data(), nullptr);
        check_error(err);
        std::cout << "\tCL_PLATFORM_VERSION: " << platform_version << '\n';

        // CL_DEVICE_TYPE_ALL
        cl_uint device_ids_count;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_ids_count);
        check_error(err);
        if(!device_ids_count) {
            std::cerr << "\tNo devices found on this platform.\n";
            continue;
        }
        std::vector<cl_device_id> device_ids(device_ids_count);
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_ids_count, device_ids.data(), nullptr);
        check_error(err);
        std::cout << "\tCL DEVICES (total " << device_ids_count << " device(s)):\n";

        std::size_t device_ids_index = 0;
        for(cl_device_id device_id : device_ids) {
            std::cout << "\tDEVICE #" << device_ids_index << '\n';

            // CL_DEVICE_NAME
            std::size_t device_name_length;
            err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_name_length);
            check_error(err);
            std::string device_name(device_name_length, '\0');
            err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_length, device_name.data(), nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_NAME: " << device_name << '\n';

            // CL_DEVICE_OPENCL_C_VERSION
            std::size_t device_opencl_c_version_length;
            err = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &device_opencl_c_version_length);
            check_error(err);
            std::string device_opencl_c_version(device_opencl_c_version_length, '\0');
            err = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, device_opencl_c_version_length, device_opencl_c_version.data(), nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_OPENCL_C_VERSION: " << device_opencl_c_version << '\n';

            // CL_DEVICE_MAX_COMPUTE_UNITS
            cl_uint device_max_compute_units;
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_max_compute_units), &device_max_compute_units, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_MAX_COMPUTE_UNITS: " << device_max_compute_units << '\n';

            // CL_DEVICE_LOCAL_MEM_SIZE
            cl_ulong device_local_mem_size;
            err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem_size), &device_local_mem_size, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_LOCAL_MEM_SIZE: " << device_local_mem_size << '\n';

            // CL_DEVICE_GLOBAL_MEM_SIZE
            cl_ulong device_global_mem_size;
            err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem_size), &device_global_mem_size, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_GLOBAL_MEM_SIZE: " << device_global_mem_size << '\n';

            // CL_DEVICE_MAX_MEM_ALLOC_SIZE
            cl_ulong device_max_mem_alloc_size;
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_max_mem_alloc_size), &device_max_mem_alloc_size, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " << device_max_mem_alloc_size << '\n';

            // CL_DEVICE_MAX_WORK_GROUP_SIZE
            std::size_t device_max_work_group_size;
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_max_work_group_size), &device_max_work_group_size, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " << device_max_work_group_size << '\n';

            // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
            cl_uint device_max_work_item_dimensions;
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(device_max_work_item_dimensions), &device_max_work_item_dimensions, nullptr);
            check_error(err);
            std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << device_max_work_item_dimensions << " (";

            // CL_DEVICE_MAX_WORK_ITEM_SIZES
            std::vector<std::size_t> device_max_work_item_sizes(device_max_work_item_dimensions);
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, device_max_work_item_dimensions * sizeof(std::size_t), device_max_work_item_sizes.data(), nullptr);
            check_error(err);
            char s[3] {'\0', ' ', '\0'};
            for(auto work_item_size : device_max_work_item_sizes) {
                std::cout << s << work_item_size;
                s[0] = ',';
            }
            std::cout << ")\n";
        }
        std::cout << std::flush;
        platform_idx++;
    }
}
