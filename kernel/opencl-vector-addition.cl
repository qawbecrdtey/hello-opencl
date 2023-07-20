__kernel void vector_add(__global float *a, __global float *b, __global float *c, unsigned long const len) {
    size_t id = get_global_id(0);
    if(id < len) {
        c[id] = a[id] + b[id];
    }
}