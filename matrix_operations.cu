
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <chrono>

struct Matrix{
	int dim = 0;
	int matrix_elements = 0;
	cuDoubleComplex* elements = NULL;
};

void take_input_data(int *matrix_dim, bool *print_matrix);
void initialize_matrix(Matrix M);
void zero_matrix(Matrix M);

/*CPU matrix operations*/
void add_matrixes(Matrix A, Matrix B, Matrix C);
void sub_matrixes(Matrix A, Matrix B, Matrix C);
void transpose_matrix(Matrix A);
void multiply_matrix(Matrix A, double factor);
void matrix_multiplication(Matrix A, Matrix B, Matrix C);
double compare_results(Matrix A, Matrix B);

void print_matrix(Matrix A);

/*GPU matrix operations*/
void GPU_matrix_operations(Matrix &A, Matrix &B, Matrix &C);



__global__ void add_matrixesGPU(Matrix A, Matrix B, Matrix C) {
	if (blockIdx.x * blockDim.x + threadIdx.x < A.matrix_elements) {
		C.elements[blockIdx.x * blockDim.x + threadIdx.x] =
			cuCadd(A.elements[blockIdx.x * blockDim.x + threadIdx.x], B.elements[blockIdx.x * blockDim.x + threadIdx.x]);
	}
}

__global__ void sub_matrixesGPU(Matrix A, Matrix B, Matrix C) {
	if (blockIdx.x * blockDim.x + threadIdx.x < A.matrix_elements) {
		C.elements[blockIdx.x * blockDim.x + threadIdx.x] =
			cuCsub(A.elements[blockIdx.x * blockDim.x + threadIdx.x], B.elements[blockIdx.x * blockDim.x + threadIdx.x]);
	}
}

__global__ void scalar_mul_matrixGPU(Matrix A, double factor) {
	if (blockIdx.x * blockDim.x + threadIdx.x < A.matrix_elements) {
		A.elements[blockIdx.x * blockDim.x + threadIdx.x].x = factor * A.elements[blockIdx.x * blockDim.x + threadIdx.x].x;
		A.elements[blockIdx.x * blockDim.x + threadIdx.x].y = factor * A.elements[blockIdx.x * blockDim.x + threadIdx.x].y;
	}
}

__global__ void transpose_matrixGPU(Matrix A) {

	if (blockIdx.x * blockDim.x + threadIdx.x < A.matrix_elements) {
		int row = (blockIdx.x * blockDim.x + threadIdx.x) / A.dim;
		int col = (blockIdx.x * blockDim.x + threadIdx.x) % A.dim;

		if (col < row) {
			cuDoubleComplex temp = A.elements[row*A.dim + col];
			A.elements[row*A.dim + col] = A.elements[col*A.dim + row];
			A.elements[col*A.dim + row] = temp;
		}
	}
}

__global__ void matrix_multiplicationGPU(Matrix A, Matrix B, Matrix C) {
	if (blockIdx.x * blockDim.x + threadIdx.x < A.matrix_elements) {
		int row = (blockIdx.x * blockDim.x + threadIdx.x) / A.dim;
		int col = (blockIdx.x * blockDim.x + threadIdx.x) % A.dim;

		cuDoubleComplex temp = { 0, 0 };
		for (int z = 0; z < A.dim; z++) {
			temp = cuCadd(temp, cuCmul(A.elements[row*A.dim + z], B.elements[z*A.dim + col]));
		}
		C.elements[row*A.dim + col] = temp;
	}
}

void print_stats(double time_cpu, double time_gpu) {
	if (time_cpu != 0 && time_gpu != 0) {
		double speed = time_cpu / time_gpu;
		if (speed < 1) {
			std::cout << "CPU byl szybszy " << 1 / speed << " razy\n\n";
		}
		else {
			std::cout << "GPU byl szybszy " << speed << " razy\n\n";
		}
	}
	else {
		std::cout << "Jedna z wartosci zmierzonego czasu w danej jednostce wynosi 0" << std::endl;
	}

}

int main()
{
	int matrix_dim = 0;
	bool print_answer = false;
	take_input_data(&matrix_dim, &print_answer);
	int matrix_elements = pow(matrix_dim, 2);

	Matrix A;
	A.dim = matrix_dim; A.matrix_elements = matrix_elements; A.elements = new cuDoubleComplex[matrix_elements];

	Matrix B;
	B.dim = matrix_dim; B.matrix_elements = matrix_elements; B.elements = new cuDoubleComplex[matrix_elements];

	Matrix C;
	C.dim = matrix_dim; C.matrix_elements = matrix_elements; C.elements = new cuDoubleComplex[matrix_elements];

	srand((unsigned)time(NULL));
	initialize_matrix(A);
	initialize_matrix(B);
	zero_matrix(C);

	if (print_answer) {
		std::cout << "Macierz A\n";
		print_matrix(A);
		std::cout << "Macierz B\n";
		print_matrix(B);
	}

	std::chrono::steady_clock::time_point beginGPU = std::chrono::steady_clock::now();
	GPU_matrix_operations(A, B, C);
	std::chrono::steady_clock::time_point endGPU = std::chrono::steady_clock::now();
	
	double durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - beginGPU).count();
	std::cout << "Czas GPU: = " << durationGPU <<" millisekund"<< std::endl;

	if (print_answer) {
		std::cout << "\nWynik GPU: \n";
		print_matrix(C);
	}


	Matrix C_CPU;
	C_CPU.dim = matrix_dim; C_CPU.matrix_elements = matrix_elements; C_CPU.elements = new cuDoubleComplex[matrix_elements];
	zero_matrix(C_CPU);

	std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();

	matrix_multiplication(A, B, C_CPU);	 // X = AB
	add_matrixes(A, C_CPU, C_CPU); // X = AB + A
	multiply_matrix(B, 8);	// B = 8 * B
	sub_matrixes(C_CPU, B, C_CPU); // X = AB + A - 8B
	multiply_matrix(A, 5);
	transpose_matrix(A);
	add_matrixes(C_CPU, A, C_CPU); //X = AB + A - 8B + 5*A_T
	if (print_answer) {
		std::cout << "\nWynik CPU: \n";
		print_matrix(C_CPU);
	}

	std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
	double durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - beginCPU).count();
	std::cout << "Czas CPU: = " << durationCPU << " millisekund"<< std::endl;

	double error = compare_results(C, C_CPU);
	std::cout << "\nblad rezultatu = " << error << std::endl;
	print_stats(durationCPU, durationGPU);

	delete[] A.elements;
	delete[] B.elements;
	delete[] C.elements;
	delete[] C_CPU.elements;
	 
    return 0;
}

/* takes input data for dimenstions of matrix from consol terminal*/
void take_input_data(int *matrix_dim, bool *print_data) {
	using namespace std;

	int dim = 0;
	bool print = 0;
	cout << "Podaj rozmiar macierzy kwadratowej: ";
	cin >> dim;
	cout << "Czy chcesz wydrukowac macierze do konsoli? (1-Tak, 0-Nie) ";
	cin >> print;

	*matrix_dim = dim;
	*print_data = print;
}
/* initializes all matrix elements to random values */
void initialize_matrix(Matrix M) {

	for (int i = 0; i < M.matrix_elements; i++) {
		M.elements[i].x = (double)rand()/(RAND_MAX);
		M.elements[i].y = (double)rand()/ (RAND_MAX);
	}
}

/* initializes all matrix elements to zero */
void zero_matrix(Matrix M) {
	for (int i = 0; i < M.matrix_elements; i++) {
		M.elements[i].x = 0.0;
		M.elements[i].y = 0.0;
	}
}

void add_matrixes(Matrix A, Matrix B, Matrix C) {
	for (int i = 0; i < A.matrix_elements; i++) {
		C.elements[i] = cuCadd(A.elements[i], B.elements[i]);
	}
}

void sub_matrixes(Matrix A, Matrix B, Matrix C) {
	for (int i = 0; i < A.matrix_elements; i++) {
		C.elements[i] = cuCsub(A.elements[i], B.elements[i]);
	}
}

/*transposes matrix and saves result in the same matrix*/
void transpose_matrix(Matrix A) {

	for (int i = 0; i < A.dim; i++) {
		for (int j = 0; j < i; j++) {
			cuDoubleComplex temp = A.elements[i*A.dim + j];
			A.elements[i*A.dim + j] = A.elements[j*A.dim + i];
			A.elements[j*A.dim + i] = temp;
		}
	}
}

/*multiply matrix by factor*/
void multiply_matrix(Matrix A, double factor) {
	for (int i = 0; i < A.matrix_elements; i++) {
		A.elements[i].x = A.elements[i].x * factor;
		A.elements[i].y = A.elements[i].y * factor;
	}
}

/*find index based on thread index and find value for this index*/
void matrix_multiplication(Matrix A, Matrix B, Matrix C) {
	for (int i = 0; i < A.dim; i++) {
		for (int j = 0; j < A.dim; j++) {
			cuDoubleComplex temp = { 0, 0 };
			for (int z = 0; z < A.dim; z++) {
				temp = cuCadd(temp, cuCmul(A.elements[i*A.dim + z], B.elements[z*A.dim + j]));
			}
			C.elements[i*A.dim + j] = temp;
		}
		
	}
}

/*print matrix in two dimensions*/
void print_matrix(Matrix A) {
	
	std::cout << "\n";
	for (int i = 0; i < A.dim; i++) {
		for (int j = 0; j < A.dim; j++) {
			std::cout <<"\t" << A.elements[i*A.dim + j].x << " + i(" << A.elements[i*A.dim + j].y << ")";
		}
		std::cout << std::endl;
	}
}

/*GPU operations*/
void GPU_matrix_operations(Matrix &A, Matrix &B, Matrix &C)
{
	Matrix d_A;
	d_A.dim = A.dim; d_A.matrix_elements = A.matrix_elements;
	d_A.elements = new cuDoubleComplex[A.matrix_elements];

	size_t size = A.matrix_elements * sizeof(cuDoubleComplex);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.dim = A.dim; d_B.matrix_elements = A.matrix_elements;
	d_B.elements = new cuDoubleComplex[A.matrix_elements];

	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.dim = A.dim; d_C.matrix_elements = A.matrix_elements;
	d_C.elements = new cuDoubleComplex[A.matrix_elements];

	cudaMalloc(&d_C.elements, size);
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
	
	unsigned long total_threads = A.matrix_elements;
	const int threads_per_block = 256;
	int blocks_number = (total_threads + threads_per_block - 1) / threads_per_block;

	matrix_multiplicationGPU <<<blocks_number, threads_per_block>>>(d_A, d_B, d_C);	 // X = AB
	add_matrixesGPU <<< blocks_number, threads_per_block >> > (d_C, d_A, d_C); // X = AB + A
	scalar_mul_matrixGPU << < blocks_number, threads_per_block >> > (d_B, 8);	// B = 8 * B
	sub_matrixesGPU << < blocks_number, threads_per_block >> > (d_C, d_B, d_C); // X = AB + A - 8B
	scalar_mul_matrixGPU << < blocks_number, threads_per_block >> > (d_A, 5);
	transpose_matrixGPU << < blocks_number, threads_per_block >> > (d_A);
	add_matrixesGPU << < blocks_number, threads_per_block >> > (d_C, d_A, d_C); //X = AB + A - 8B + 5*A_T
	
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);


	cudaFree(d_A.elements);
	cudaFree(d_B.elements);	
	cudaFree(d_C.elements);
}

double compare_results(Matrix A, Matrix B) {
	double max_value = 0;
	for (int i = 0; i < A.matrix_elements; i++) {
		if (B.elements[i].x > max_value) {
			max_value = B.elements[i].x;
		}
	}
	double max_error = 0;
	for (int i = 0; i < A.matrix_elements; i++) {
		double error = A.elements[i].x - B.elements[i].x;
		double rel_error = error / max_value;
		if (abs(rel_error) > abs(max_error)) {
			max_error = rel_error;
		}
	}
	return max_error;
}