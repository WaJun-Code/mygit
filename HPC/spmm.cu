#include <cstdlib>                    // std::rand(), RAND_MAX
#include <cuda_runtime_api.h>         // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h> // cusparseSpMM (>= v11.0) or cusparseScsrmm
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main() {

  int M=2;                              // number of A-rows
  int K=3;                              // number of A-columns
  int nnz=3;                            // number of non-zeros in A
  int *csr_indptr_buffer;
  int *csr_indices_buffer; // buffer for indices (column-ids) array in CSR format
  csr_indptr_buffer = (int *)malloc(sizeof(int) * (M+1));
  csr_indices_buffer = (int *)malloc(sizeof(int) * nnz);
  csr_indptr_buffer[0]=0;csr_indptr_buffer[1]=1;csr_indptr_buffer[2]=3;
  csr_indices_buffer[0]=1;csr_indices_buffer[1]=0;csr_indices_buffer[2]=2;


  // Create GPU arrays
  int N = 4; // number of B-columns
  
  float *B_h = NULL, *C_h = NULL, *csr_values_h = NULL, *C_ref = NULL;
  float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL;

  B_h = (float *)malloc(sizeof(float) * K * N);
  C_h = (float *)malloc(sizeof(float) * M * N);
  csr_values_h = (float *)malloc(sizeof(float) * nnz);
  csr_values_h[0]=1;csr_values_h[1]=1;csr_values_h[2]=1;
  for(int i=0;i<K * N;i++){B_h[i]=i+1;}


  cudaMalloc((void **)&B_d, sizeof(float) * K * N);
  cudaMalloc((void **)&C_d, sizeof(float) * M * N);
  cudaMalloc((void **)&csr_values_d, sizeof(float) * nnz);
  cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1));
  cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz);

  cudaMemcpy(B_d, B_h, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemset(C_d, 0x0, sizeof(float) * M * N);
  cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz,cudaMemcpyHostToDevice);
  cudaMemcpy(csr_indptr_d, csr_indptr_buffer,
                        sizeof(int) * (M + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_indices_d, csr_indices_buffer,
                        sizeof(int) * nnz, cudaMemcpyHostToDevice);

  //
  // Run Cusparse-SpMM and check result
  //

  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  cusparseCreate(&handle);

  // creating sparse csr matrix
  cusparseCreateCsr(
      &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      );

  // creating dense matrices
  cusparseCreateDnMat(&dnMatInputDescr, K, N, N, B_d, CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW);

  // allocate workspace buffer
  size_t workspace_size;
  cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size);

  void *workspace = NULL;
  cudaMalloc(&workspace, workspace_size);

  // run SpMM
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                              &alpha, csrDescr, dnMatInputDescr, &beta,
                              dnMatOutputDescr, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, workspace);

  cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      printf("%f\t",C_h[i*N+j]);
    }
  }

  //
  // Benchmark Cusparse-SpMM performance
  //

  int warmup_iter = 10;
  int repeat_iter = 100;
  for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                 &alpha, csrDescr, dnMatInputDescr, &beta, dnMatOutputDescr,
                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
  }

  float MFlop_count = (float)nnz / 1e6 * N * 2;

  /// free memory

  if (B_h)
    free(B_h);
  if (C_h)
    free(C_h);
  if (C_ref)
    free(C_ref);
  if (csr_values_h)
    free(csr_values_h);
  if (B_d)
    cudaFree(B_d);
  if (C_d)
    cudaFree(C_d);
  if (csr_values_d)
    cudaFree(csr_values_d);
  if (csr_indptr_d)
    cudaFree(csr_indptr_d);
  if (csr_indices_d)
    cudaFree(csr_indices_d);
  if (workspace)
    cudaFree(workspace);

  // destroy matrix/vector descriptors
  cusparseDestroyDnMat(dnMatInputDescr);
  cusparseDestroyDnMat(dnMatOutputDescr);
  cusparseDestroySpMat(csrDescr);
  cusparseDestroy(handle);
  return EXIT_SUCCESS;
}
