__kernel void my_kernel(__global float *A, __global float *B, __global float *C, int N) {
  int i = get_global_id(0);
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
