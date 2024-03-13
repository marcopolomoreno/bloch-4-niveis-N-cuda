#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "propDevice.h"  //Biblioteca externa

#define Pi 3.141592653589793

#define blocks 20
#define threads 32
#define gpu 1
#define arq 1		//número de arquivo para salvar

#define kMax 100000
#define pontos 40
#define passoFreq 0.5

__constant__ double A = (2 * Pi) * 6e6;
__constant__ double B = (2 * Pi) * 1e6;    //em rad/s

#define gama22 (2*Pi)*6.06e6
#define gama44 (2*Pi)*6.06e6

__constant__ double h = 5e-12;    //*10000/kMax;

__constant__ double gama12 = 0.5 * gama22;
__constant__ double gama13 = 0;
__constant__ double gama14 = 0.5 * gama44;
__constant__ double gama23 = 0.5 * gama22;
__constant__ double gama24 = 0.5 * (gama22 + gama44);
__constant__ double gama34 = 0.5 * gama44;

__constant__ int nucleos = blocks * threads;

#define CUDA_ERROR_CHECK
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
#endif
	return;
}

__device__ double f(double a11, double a22, double a33, double a44, double a12, double b12, double a13, double b13, double a14, double b14,
	double a23, double b23, double a24, double b24, double a34, double b34, 
	double delta21, double delta32, double delta41, double delta43, double delta31, double delta42, int j)  //sistema de 4 níveis
{
	/*a11*/ if (j == 1)  return 2 * A * b12 + 0.5 * gama22 * a22 + 0.5 * gama44 * a44;				   //a11
	/*a22*/ if (j == 2)  return -2 * A * b12 + 2 * B * b23 - gama22 * a22;				//a22
	/*a33*/ if (j == 3)  return 2 * A * b34 - 2 * B * b23 + 0.5 * gama22 * a22 + 0.5 * gama44 * a44;   //a33
	/*a44*/ if (j == 4)  return -2 * A * b34 - gama44 * a44;		   //a44

	/*a12*/ if (j == 5)  return -gama12 * a12 - delta21 * b12 + B * b13; //a12
	/*b12*/ if (j == 6)  return -gama12 * b12 + delta21 * a12 + (a22 - a11) * A - B * a13; //b12
	/*a13*/ if (j == 7)  return -gama13 * a13 - delta31 * b13 + A * b14 - A * b23 + B * b12; //a13
	/*b13*/ if (j == 8)  return -gama13 * b13 + delta31 * a13 - A * a14 + A * a23 - B * a12; //b13
	/*a14*/ if (j == 9)  return -gama14 * a14 - delta41 * b14 + A * b13 - A * b24; //a14
	/*b14*/ if (j == 10) return -gama14 * b14 + delta41 * a14 - A * a13 + A * a24; //b14
	/*a23*/ if (j == 11) return -gama23 * a23 - delta32 * b23 - A * b13 + A * b24; //a43
	/*b23*/ if (j == 12) return -gama23 * b23 + delta32 * a23 + A * a13 - A * a24 + (a33 - a22) * B; //b43
	/*a24*/ if (j == 13) return -gama24 * a24 - delta42 * b24 - A * b14 + A * b23 - B * b34; //a13
	/*b24*/ if (j == 14) return -gama24 * b24 + delta42 * a24 + A * a14 - A * a23 + B * a34; //b13
	/*a34*/ if (j == 15) return -gama34 * a34 - delta43 * b34 - B * b24; //a24
	/*b34*/ if (j == 16) return -gama34 * b34 + delta43 * a34 + B * a24 - (a33 - a44) * A; //b24
}

__global__ void Kernel(double* a11, double* a22, double* a33, double* a44, double* a12, double* b12, double* a13, double* b13, double* a14, double* b14,
	double* a23, double* b23, double* a24, double* b24, double* a34, double* b34)
{
	//Paralelização na dessintonia (variável delta21)

	int j, k;
	double k1[17], k2[17], k3[17], k4[17];

	double delta21, delta32, delta41, delta43;
	double delta31, delta42;

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	delta21 = 2 * Pi * (i - 0.5 * nucleos) * passoFreq * 1e6;
	delta32 = -delta21;
	delta43 =  delta21;
	delta41 =  delta21;

	delta31 = 0;
	delta42 = 0;

	for (k = 1; k <= kMax - 1; k++)    //abre loop de k (temporal)
	{
		for (j = 1; j <= 16; j++)
			k1[j] = f(a11[i], a22[i], a33[i], a44[i], a12[i], b12[i], a13[i], b13[i], a14[i], b14[i], a23[i], b23[i],
				a24[i], b24[i], a34[i], b34[i], delta21, delta32, delta41, delta43, delta31, delta42, j);

		for (j = 1; j <= 16; j++)
			k2[j] = f(a11[i] + k1[1] * h / 2, a22[i] + k1[2] * h / 2, a33[i] + k1[3] * h / 2, a44[i] + k1[4] * h / 2, a12[i] + k1[5] * h / 2, b12[i] + k1[6] * h / 2, a13[i] + k1[7] * h / 2, b13[i] + k1[8] * h / 2,
				a14[i] + k1[9] * h / 2, b14[i] + k1[10] * h / 2, a23[i] + k1[11] * h / 2, b23[i] + k1[12] * h / 2, a24[i] + k1[13] * h / 2, b24[i] + k1[14] * h / 2,
				a34[i] + k1[15] * h / 2, b34[i] + k1[16] * h / 2, delta21, delta32, delta41, delta43, delta31, delta42, j);

		for (j = 1; j <= 16; j++)
			k3[j] = f(a11[i] + k2[1] * h / 2, a22[i] + k2[2] * h / 2, a33[i] + k2[3] * h / 2, a44[i] + k2[4] * h / 2, a12[i] + k2[5] * h / 2, b12[i] + k2[6] * h / 2, a13[i] + k2[7] * h / 2, b13[i] + k2[8] * h / 2,
				a14[i] + k2[9] * h / 2, b14[i] + k2[10] * h / 2, a23[i] + k2[11] * h / 2, b23[i] + k2[12] * h / 2, a24[i] + k2[13] * h / 2, b24[i] + k2[14] * h / 2,
				a34[i] + k2[15] * h / 2, b34[i] + k2[16] * h / 2, delta21, delta32, delta41, delta43, delta31, delta42, j);

		for (j = 1; j <= 16; j++)
			k4[j] = f(a11[i] + k3[1] * h, a22[i] + k3[2] * h, a33[i] + k3[3] * h, a44[i] + k3[4] * h, a12[i] + k3[5] * h, b12[i] + k3[6] * h, a13[i] + k3[7] * h, b13[i] + k3[8] * h, a14[i] + k3[9] * h, b14[i] + k3[10] * h,
				a23[i] + k3[11] * h, b23[i] + k3[12] * h, a24[i] + k3[13] * h, b24[i] + k3[14] * h, a34[i] + k3[15] * h, b34[i] + k3[16] * h, delta21, delta32, delta41, delta43, delta31, delta42, j);

		a11[i] = a11[i] + h * (k1[1] / 6 + k2[1] / 3 + k3[1] / 3 + k4[1] / 6);	   a22[i] = a22[i] + h * (k1[2] / 6 + k2[2] / 3 + k3[2] / 3 + k4[2] / 6);
		a33[i] = a33[i] + h * (k1[3] / 6 + k2[3] / 3 + k3[3] / 3 + k4[3] / 6);	   a44[i] = a44[i] + h * (k1[4] / 6 + k2[4] / 3 + k3[4] / 3 + k4[4] / 6);
		a12[i] = a12[i] + h * (k1[5] / 6 + k2[5] / 3 + k3[5] / 3 + k4[5] / 6);     b12[i] = b12[i] + h * (k1[6] / 6 + k2[6] / 3 + k3[6] / 3 + k4[6] / 6);
		a13[i] = a13[i] + h * (k1[7] / 6 + k2[7] / 3 + k3[7] / 3 + k4[7] / 6);	   b13[i] = b13[i] + h * (k1[8] / 6 + k2[8] / 3 + k3[8] / 3 + k4[8] / 6);
		a14[i] = a14[i] + h * (k1[9] / 6 + k2[9] / 3 + k3[9] / 3 + k4[9] / 6);     b14[i] = b14[i] + h * (k1[10] / 6 + k2[10] / 3 + k3[10] / 3 + k4[10] / 6);
		a23[i] = a23[i] + h * (k1[11] / 6 + k2[11] / 3 + k3[11] / 3 + k4[11] / 6); b23[i] = b23[i] + h * (k1[12] / 6 + k2[12] / 3 + k3[12] / 3 + k4[12] / 6);
		a24[i] = a24[i] + h * (k1[13] / 6 + k2[13] / 3 + k3[13] / 3 + k4[13] / 6); b24[i] = b24[i] + h * (k1[14] / 6 + k2[14] / 3 + k3[14] / 3 + k4[14] / 6);
		a34[i] = a34[i] + h * (k1[15] / 6 + k2[15] / 3 + k3[15] / 3 + k4[15] / 6); b34[i] = b34[i] + h * (k1[16] / 6 + k2[16] / 3 + k3[16] / 3 + k4[16] / 6);
	}  //loop tempo
}

int main()
{
	clock_t begin, end;
	double time_spent;
	begin = clock();

	const int nucleos = blocks * threads;
	FILE* arquivo[arq];

	propriedades();	//Biblioteca externa

	printf("Blocks = %d\n", blocks);
	printf("Threads = %d\n\n", threads);

	printf("Calculando...\n");

	double a[17][nucleos], soma;
	int q;

	double* dev_a11[gpu], * dev_a22[gpu];
	double* dev_a33[gpu], * dev_a44[gpu];
	double* dev_a12[gpu], * dev_b12[gpu];
	double* dev_a13[gpu], * dev_b13[gpu];
	double* dev_a14[gpu], * dev_b14[gpu];
	double* dev_a23[gpu], * dev_b23[gpu];
	double* dev_a24[gpu], * dev_b24[gpu];
	double* dev_a34[gpu], * dev_b34[gpu];

	int bytes = nucleos * sizeof(double);
	cudaStream_t stream[gpu];

	for (int kp = 0; kp <= arq - 1; kp++)   //loop nos arquivos
	{

		char text[] = "dadosX.dat";
		text[5] = kp + '0';

		arquivo[kp] = fopen(text, "w");

		fprintf(arquivo[kp], "\\g(d)\\-(21) \\g(r)\\-(11) \\g(r)\\-(22) \\g(r)\\-(33) \\g(r)\\-(44) Re\\g(s)\\-(12) Im\\g(s)\\-(12) Re\\g(s)\\-(13) Im\\g(s)\\-(13) Re\\g(s)\\-(14) Im\\g(s)\\-(14) Re\\g(s)\\-(23) Im\\g(s)\\-(23) Re\\g(s)\\-(24) Im\\g(s)\\-(24) Re\\g(s)\\-(34) Im\\g(s)\\-(34)\n");
		fprintf(arquivo[kp], "MHz \n");
		fprintf(arquivo[kp], "delta21 rho11 rho22 rho33 rho44 Re_rho12 Im_rho12 Re_rho13 Im_rho13 Re_rho14 Im_rho14 Re_rho23 Im_rho23 Re_rho24 Im_rho24 Re_rho34 Im_rho34\n");


		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);

			cudaMalloc((void**)&dev_a11[pp], bytes); cudaMalloc((void**)&dev_a22[pp], bytes);
			cudaMalloc((void**)&dev_a33[pp], bytes); cudaMalloc((void**)&dev_a44[pp], bytes);
			cudaMalloc((void**)&dev_a12[pp], bytes); cudaMalloc((void**)&dev_b12[pp], bytes);
			cudaMalloc((void**)&dev_a13[pp], bytes); cudaMalloc((void**)&dev_b13[pp], bytes);
			cudaMalloc((void**)&dev_a14[pp], bytes); cudaMalloc((void**)&dev_b14[pp], bytes);
			cudaMalloc((void**)&dev_a23[pp], bytes); cudaMalloc((void**)&dev_b23[pp], bytes);
			cudaMalloc((void**)&dev_a24[pp], bytes); cudaMalloc((void**)&dev_b24[pp], bytes);
			cudaMalloc((void**)&dev_a34[pp], bytes); cudaMalloc((void**)&dev_b34[pp], bytes);

			cudaStreamCreate(&stream[pp]);

			for (q = 0; q <= nucleos - 1; q++)
			{
				a[1][q] = 0.5; a[2][q] = 0;
				a[3][q] = 0.5; a[4][q] = 0;
				for (int k = 5; k <= 16; k++)
					a[k][q] = 0;
			}

			cudaMemcpyAsync(dev_a11[pp], a[1],  bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_a22[pp], a[2],  bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a33[pp], a[3],  bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_a44[pp], a[4],  bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a12[pp], a[5],  bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b12[pp], a[6],  bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a13[pp], a[7],  bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b13[pp], a[8],  bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a14[pp], a[9],  bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b14[pp], a[10], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a23[pp], a[11], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b23[pp], a[12], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a24[pp], a[13], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b24[pp], a[14], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a34[pp], a[15], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b34[pp], a[16], bytes, cudaMemcpyHostToDevice, stream[pp]);
		}

		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);
			Kernel << <blocks, threads, 0, stream[pp] >> > (dev_a11[pp], dev_a22[pp], dev_a33[pp], dev_a44[pp], dev_a12[pp], dev_b12[pp], dev_a13[pp], dev_b13[pp], dev_a14[pp],
				dev_b14[pp], dev_a23[pp], dev_b23[pp], dev_a24[pp], dev_b24[pp], dev_a34[pp], dev_b34[pp]);
		}

		//CudaCheckError();

		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);
			cudaMemcpyAsync(a[1],  dev_a11[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[2],  dev_a22[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[3],  dev_a33[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[4],  dev_a44[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[5],  dev_a12[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[6],  dev_b12[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[7],  dev_a13[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[8],  dev_b13[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[9],  dev_a14[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[10], dev_b14[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[11], dev_a23[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[12], dev_b23[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[13], dev_a24[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[14], dev_b24[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[15], dev_a34[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[16], dev_b34[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);

			for (q = 0; q <= nucleos - 1; q++)
			{
				soma = a[1][q] + a[2][q] + a[3][q] + a[4][q];

				printf("%f %.16f %.16f %.16f %.16f %.16f\n", double((gpu * q + pp) * passoFreq),
					a[1][q], a[2][q], a[3][q], a[4][q], soma);

				fprintf(arquivo[kp], "%f %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", double((gpu * q + pp - 0.5 * nucleos) * passoFreq),
					a[1][q], a[2][q], a[3][q], a[4][q], a[5][q], a[6][q], a[7][q], a[8][q], a[9][q], a[10][q], a[11][q], a[12][q], a[13][q], a[14][q], a[15][q], a[16][q]);
			}

			//cudaDeviceSynchronize();			
		}

		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);
			cudaFree(dev_a11[pp]); cudaFree(dev_a22[pp]);
			cudaFree(dev_a33[pp]); cudaFree(dev_a44[pp]);
			cudaFree(dev_a12[pp]); cudaFree(dev_b12[pp]);
			cudaFree(dev_a13[pp]); cudaFree(dev_b13[pp]);
			cudaFree(dev_a14[pp]); cudaFree(dev_b14[pp]);
			cudaFree(dev_a23[pp]); cudaFree(dev_b23[pp]);
			cudaFree(dev_a24[pp]); cudaFree(dev_b24[pp]);
			cudaFree(dev_a34[pp]); cudaFree(dev_b34[pp]);
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaDeviceReset();

		fclose(arquivo[kp]);
	}

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (time_spent <= 60) printf("\nTempo de execucao = %f s\n\n", time_spent);
	if (time_spent > 60 && time_spent <= 3600) printf("\nTempo de execucao = %f min\n\n", time_spent / 60);
	if (time_spent > 3600) printf("\nTempo de execucao = %f h\n\n", time_spent / 3600);

	printf("\a");
}