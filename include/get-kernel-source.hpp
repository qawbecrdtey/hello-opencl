#ifndef HELLO_OPENCL_GET_KERNEL_SOURCE_HPP
#define HELLO_OPENCL_GET_KERNEL_SOURCE_HPP

#include <CL/cl.h>

#include <cstdlib>

inline char *get_kernel_source(char const *path) {
    auto *f = fopen(path, "r");
    fseek(f, 0, SEEK_END);
    auto fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *kernel_source = new char[fsize + 1];
    fread(kernel_source, fsize, 1, f);
    fclose(f);
    kernel_source[fsize] = 0;

    return kernel_source;
}

#endif //HELLO_OPENCL_GET_KERNEL_SOURCE_HPP
