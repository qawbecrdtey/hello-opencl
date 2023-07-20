__kernel void matrix_multiplication_naive(__global float *A, __global float *B, __global float *C, int const I, int const K, int const J) {
    int row = get_global_id(0); // I
    int col = get_global_id(1); // J

    float sum = 0.0f;
    for(size_t k = 0; k < K; k++) {
        sum += A[row * I + k] * B[k * K + col];
    }
    C[row * I + col] = sum;
}