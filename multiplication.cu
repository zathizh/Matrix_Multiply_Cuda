#include <stdio.h>
#include <stdlib.h>

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
        int width;
        int height;
        int* elements;
} Matrix;

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

__global__ void MatMulKernel(const Matrix, const Matrix, const Matrix);
void MatMul(int, int, int);

int main(int argc, char* argv[]){
        if (argc != 4){
                printf("usage: multiplication <X> <Y> <Z>");
                exit(1);
        }

        int m, n, p;
	m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);

        MatMul(m, n, p);
}

void MatMul(int m, int n, int p){
	Matrix A, B, C, d_A, d_B, d_C;
	A.height = d_A.height = m;
        A.width = d_A.width = n;

        B.height = d_B.height  = n;
        B.width = d_B.width = p;

        C.height = d_C.height = m;
        C.width = d_C.width = p;

	A.elements = (int*)malloc(A.width * A.height * sizeof(int));
	B.elements = (int*)malloc(B.width * B.height * sizeof(int));
	C.elements = (int*)malloc(C.width * C.height * sizeof(int));
//	int len = C.width * C.height * sizeof(float);

	// filling A, B Matrixes
	int i, j;
	for(i = 0; i < A.height; i++){
                for(j = 0; j < A.width; j++){
                        A.elements[i*A.width + j] = (int)(rand() % 100);
			printf("%d ", A.elements[i*A.width + j]);
		}
		printf("\n");
	}
	printf("\n");


 	for(i = 0; i < B.height; i++){
                for(j = 0; j < B.width; j++){
                        B.elements[i*B.width + j] = (int)(rand() % 100);
                        printf("%d ", B.elements[i*B.width + j]);
                }
                printf("\n");
        }
	printf("\n");

	size_t size = A.width * A.height * sizeof(int);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A : %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
        printf("Copy A to device: %s\n",cudaGetErrorString(err));

	size = B.width * B.height * sizeof(int);
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B : %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
        printf("Copy B to device: %s\n",cudaGetErrorString(err));
		
	size = C.width * C.height * sizeof(int);
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C : %s\n", cudaGetErrorString(err));

	// Invoke kernel
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);//Dimensions of blocks here it is 16-by-16
        dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);//Dimensions of grid in terms of block
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        err = cudaThreadSynchronize();
        printf("Run kernel: %s\n", cudaGetErrorString(err));

	// Read C from device memory
        err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
        printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// printing matrix C
        for(i = 0; i < C.height; i++){
                for(j = 0; j < C.width; j++){
                        printf("%d ", C.elements[i*C.width + j]);
                }
                printf("\n");
        }
	printf("\n");

	// Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
//	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

// Each thread computes one element of C
// by accumulating results into Cvalue

	float Cvalue = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > A.height || col > B.width)
		return;

	for (int e = 0; e < A.width; ++e){
		Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
	}

	C.elements[row * C.width + col] = Cvalue;
}

