#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <iostream> 
#include <stdio.h>
#include<chrono>
using namespace std;
// 定义测试矩阵的维度 
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;
__global__ void Muld(float* Ad, float* Bd, float* Cd, int width)
{
    // 2D Thread ID
    int tx = threadIdx.x;   //block中是（列,行）的下标
    int ty = threadIdx.y;
    // Cd[ty][tx]=Sum(k)Ad[ty][k]*Bd[k][tx]

   /* float cvalue = 0;
    for (int k = 0; k < width; ++k)
    {
        float ae = Ad[ty * width + k];
        float be = Bd[tx + k * width];
        cvalue += ae * be;
    }
    Cd[ty * width + tx] = cvalue;*/
    
    int ii, jj, j = width / 32;
    for (int i = 0; i < width*width / (32*32); i++) {
        float cvalue = 0;
        ii = i % j; jj = i / j;
        for (int k = 0; k < width; k++)
        {

            float ae = Ad[(ty + ii*32) * width + k];
            float be = Bd[(tx + jj * 32) + k * width];
            cvalue += ae * be;
        }
        // Write the matrix to device memory;
        // each thread writes one element
        Cd[(ty + ii * 32) * width + (tx + jj * 32)] = cvalue;
    }
}
Matrix Mul(const Matrix A, const Matrix B, Matrix C)
{
    int size = A.width * A.width * sizeof(float);
    // Load A and B to the device，Allocate C on the device
    float* Ad, * Bd, * Cd;
    cudaMalloc((void**)&Ad, size); //matrix stored in linear order
    cudaMemcpy(Ad, A.elements, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Bd, size);
    cudaMemcpy(Bd, B.elements, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Cd, size);
    // Launch the device computation threads!
    int Width = A.width;
    dim3 dimGrid(1);
    dim3 dimBlock(32, 32);

    auto time_start = chrono::system_clock::now();
    Muld << <dimGrid, dimBlock >> > (Ad, Bd, Cd, Width);
    auto time_end = chrono::system_clock::now();
    chrono::duration<double>endstart = time_end - time_start;

    cout << "程序运行所耗时间："<<endstart.count() <<endl;
    // Read C from the device
    cudaMemcpy(C.elements, Cd, size, cudaMemcpyDeviceToHost);
    // Free device matrices
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return C;
}
Matrix AllocateMatrix(int xdim, int ydim, float value) {
    Matrix M;
    M.width = xdim; M.height = ydim;
    int nBytes = xdim * ydim * sizeof(float);
    // 申请host内存
    M.elements = (float*)malloc(nBytes);
    // 初始化数据
    for (int i = 0; i < xdim * ydim; ++i)
    {
        M.elements[i] = value;
    }
    return M;
}
void FreeMatrix(Matrix M) {
    Matrix* p;
    p = &M;
    free(p->elements);
}
int main(void) { //host side

// Allocate and initialize the matrices
    int WIDTH = 64;
    Matrix A, B, C;
    A = AllocateMatrix(WIDTH, WIDTH, 1);
    B = AllocateMatrix(WIDTH, WIDTH, 1);
    C = AllocateMatrix(WIDTH, WIDTH, 0);

    // C = A * B on the device
    C = Mul(A, B, C);
    // Free matrices
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        cout << C.elements[i] << " ";
        if ((i+1) % WIDTH == 0) { cout << endl; }
    }
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C);
    system("pause");
    return 0;
}