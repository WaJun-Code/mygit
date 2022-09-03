#include <cuda_runtime.h> 
#include <cublas_v2.h>
#include <iostream> 
#include <stdio.h>
#include <ctime>
#include <vector>
using namespace std;

#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])
__global__void naive_MatrixMul(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, const int M, const int K, const int N, float alpha, float beta){
    // grid<n/blockDim.x,m/blockDim.y>, block<32,Dimy>
    float accm=0;

    int Arow = blockIdx.y*blockDim.y+threadIdx.y;
    int Bcol = blockIdx.x*blockDim.x+threadIdx.x;

    for(int i=0;i<K;i++){
        accm += A[OFFSET(Arow,i,K)]*B[OFFSET(i,Bcol,N)];
    }
    C[OFFSET(Arow,Bcol,N)] = accm;
}
__global__void Coalesed_MatrixMul(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, const int M, const int K, const int N, float alpha, float beta){
    // grid<n/blockDim.x,m/blockDim.y>, block<32,Dimy>
    extern __shared__ float sh[];
    int sm_offset = threadIdx.y<<5;
    int thread_idx = sm_offset + threadIdx.x;

    int cid = blockIdx.y<<5 + threadIdx.x;
    int rid = blockDim.y*blockIdx.x + threadIdx.y;

    if(rid<M){
        int lb = 0;
        int hb = K;
        int ptr = lb+threadIdx.x;
        float accm = 0;
        for(int jj=lb;jj<hb;jj+=32){
            if(ptr<hb){
                sh[thread_idx] = A[OFFSET(rid,ptr,K)];
            }
            __syncwarp();
            ptr += 32;
            for(int kk=0;kk<32&&jj+kk<hb;kk++){
                accm += sh[sm_offset+kk]*B[OFFSET(jj+kk,cid,N)];
            }
            __syncwarp();
        }
        C[OFFSET(Arow,Bcol,N)] = accm;
    }
}
template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__void SMEM_buffer(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, const int M, const int K, const int N, float alpha, float beta){
    //grid<n/bn,m/bm>，block<bn/tn,bm/tm>
    __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    float accm[THREAD_SIZE_M][THREAD_SIZE_N]={0};
    float reg_a[2][THREAD_SIZE_M],reg_b[2][THREAD_SIZE_N];
    int Arow0 = blockIdx.y*BLOCK_SIZE_M, Bcol0 = blockIdx.x*BLOCK_SIZE_N;
    int tArow0 = threadIdx.y*THREAD_SIZE_M, tBcol0 = threadIdx.x*THREAD_SIZE_N;
    //先写第0次迭代的结果
    for(int i=0;i<BLOCK_SIZE_M;i++){
        for(int j=0;j<BLOCK_SIZE_K;j+=4){
            FETCH_FLOAT4(As[0][i][j]) = FETCH_FLOAT4(A[ OFFSET(Arow0+i,j,K) ]);
        }
    }
    for(int i=0;i<BLOCK_SIZE_K;i++){
        for(int j=0;j<BLOCK_SIZE_N;j+=4){
            FETCH_FLOAT4(Bs[0][i][j]) = FETCH_FLOAT4(B[ OFFSET(i,Bcol0+j,N) ]);
        }
    }
    __syncthreads();
    int write_stage_idx = 1;
    for(ki=BLOCK_SIZE_K;ki<K+BLOCK_SIZE_K;ki+=BLOCK_SIZE_K){
        if(ki<K){
            for(int i=0;i<BLOCK_SIZE_M;i++){
                for(int j=0;j<BLOCK_SIZE_K;j+=4){
                    FETCH_FLOAT4(As[write_stage_idx][i][j]) = FETCH_FLOAT4(A[ OFFSET(Arow0+i,ki+j,K) ]);
                }
            }
            for(int i=0;i<BLOCK_SIZE_K;i++){
                for(int j=0;j<BLOCK_SIZE_N;j+=4){
                    FETCH_FLOAT4(Bs[write_stage_idx][i][j]) = FETCH_FLOAT4(B[ OFFSET(ki+i,Bcol0+j,N) ]);
                }
            }
        }
        //先预存第0次
        for(int j=0;j<THREAD_SIZE_M;j++){
            reg_a[0][j] = As[write_stage_idx^1][tArow0+j][0];
        }
        for(int j=0;j<THREAD_SIZE_N;j+=4){
            FETCH_FLOAT4(reg_b[0][j]) = FETCH_FLOAT4(Bs[write_stage_idx^1][0][tBcol0+j]);
        }
        //进行小迭代
        for(int i=1;i<BLOCK_SIZE_K+1;i++){
            if(i<BLOCK_SIZE_K){
                for(int j=0;j<THREAD_SIZE_M;j++){
                    reg_a[i%2][j] = As[write_stage_idx^1][tArow0+j][i];
                }
                for(int j=0;j<THREAD_SIZE_N;j+=4){
                    FETCH_FLOAT4(reg_b[i%2][j]) = FETCH_FLOAT4(Bs[write_stage_idx^1][i][tBcol0+j]);
                }
            }
            //两层循坏计算结果
            for(int j=0;j<THREAD_SIZE_M;j++){
                for(int k=0;k<THREAD_SIZE_N;k++){
                    accm[j][k] += reg_a[(i+1)%2][j]*reg_b[(i+1)%2][k];
                }
            }
        }
        __syncthreads();
        write_stage_idx ^= 1;
    }
    for(int j=0;j<THREAD_SIZE_M;j++){
        for(int k=0;k<THREAD_SIZE_N;k+=4){
            FETCH_FLOAT4(C[ OFFSET(Arow0+tArow0+j,Bcol0+tBcol0+k,N) ]) = FETCH_FLOAT4(accm[j][k]);
        }
    }
}
template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__void MatrixMul(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, const int M, const int K, const int N, float alpha, float beta){
    //grid<n/bn,m/bm>，block<bn/tn,bm/tm>
    float accm[THREAD_SIZE_M][THREAD_SIZE_N]={0};
    float reg_a[THREAD_SIZE_M],reg_b[THREAD_SIZE_N];
    int Arow0 = blockIdx.y*BLOCK_SIZE_M, Bcol0 = blockIdx.x*BLOCK_SIZE_N;
    int tArow0 = threadIdx.y*THREAD_SIZE_M, tBcol0 = threadIdx.x*THREAD_SIZE_N;

    for(ki=0;ki<K0;ki++){
        for(int j=0;j<THREAD_SIZE_M;j++){
            reg_a[j] = A[OFFSET(Arow0+tArow0+j,ki,K)];
        }
        for(int j=0;j<THREAD_SIZE_N;j+=4){
            FETCH_FLOAT4(reg_b[j]) = FETCH_FLOAT4(B[OFFSET(ki,Brow0+tBrow0+j,N)]);
        }
        //进行小迭代
        //两层循坏计算结果
        for(int j=0;j<THREAD_SIZE_M;j++){
            for(int k=0;k<THREAD_SIZE_N;k++){
                accm[j][k] += reg_a[j]*reg_b[k];
            }
        }
    }
    for(int j=0;j<THREAD_SIZE_M;j++){
        for(int k=0;k<THREAD_SIZE_N;k+=4){
            FETCH_FLOAT4(C[ OFFSET(Arow0+tArow0+j,Bcol0+tBcol0+k,N) ]) = FETCH_FLOAT4(accm[j][k]);
        }
    }
}
int main(void) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("device %d:%s \n",dev,deviceProp.name);
    cout << "SM的数量：" << devProp.multiProcessorCount << endl;
    cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
    cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << endl;
    cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << endl;
    cudaSetDevice(dev);

    int m=10240,k=7680,n=5120;
    float a=1.0,b=0.0,*h_A;
    const int BLOCK_SIZE_M = 128, BLOCK_SIZE_K = 8, BLOCK_SIZE_N = 64, THREAD_SIZE_M = 8, THREAD_SIZE_N = 8;
    h_A = (float*)malloc(sizeof(float)*m*k);
    float *h_B;float *h_C;float *tmp_C;
    h_B = (float*)malloc(sizeof(float)*k*n);
    h_C = (float*)malloc(sizeof(float)*m*n);
    tmp_C = (float*)malloc(sizeof(float)*m*n);
    float *d_A;float *d_B;float *d_C;
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            h_A[i*k+j] = (float)(rand()%100)/100;
        }
    }
    for(int i=0;i<k;i++){
        for(int j=0;j<n;j++){
            h_B[i*n+j] = (float)(rand()%100)/100;
        }
    }
    clock_t iStart;
    cudaMemcpy(d_A,h_A,m*k*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,k*n*sizeof(float),cudaMemcpyHostToDevice);

    cout<<"开始cuBLAS计算:"<<" ";
    iStart = clock();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, n, m, k, &a, d_B, n , d_A, k, &b ,d_C, n);
    cudaDeviceSynchronize();
    cout<<"elapsed:"<<(clock()-iStart)/1000.0<<"ms"<<endl;
    cudaMemcpy(tmp_C,d_C,m*n*sizeof(float),cudaMemcpyDeviceToHost);

    int tile_k,n_block;
    n_block = (n+BLOCK_SIZE_N-1)/BLOCK_SIZE_N;
    tile_k = (m+BLOCK_SIZE_M-1)/BLOCK_SIZE_M;
    cout<<"开始gemm计算:"<<tile_k<<", "<<n_block<<" ";
    iStart = clock();
    SMEM_buffer<BLOCK_SIZE_M , BLOCK_SIZE_K , BLOCK_SIZE_N , THREAD_SIZE_M , THREAD_SIZE_N > <<<dim3(n_block,tile_k,1), dim3(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M, 1)>>>(d_A,d_B,d_C,m,k,n,a,b);
    cudaDeviceSynchronize();
    cout<<"elapsed:"<<(clock()-iStart)/1000.0<<"ms"<<endl;
    cudaMemcpy(h_C,d_C,m*n*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(abs(h_C[i*n+j]-tmp_C[i*n+j])/abs(tmp_C[i][j])>0.01){cout<<h_C[i*n+j]<<"\t"<<tmp_C[i*n+j]<<"|";}
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
