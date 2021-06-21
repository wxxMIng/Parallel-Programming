#include <xmmintrin.h>
#define SSE_WIDTH		4
#include <omp.h>
#include <stdio.h>
#include <math.h>


#ifndef ARRAYSIZE
#define ARRAYSIZE 8000000
#endif

#ifndef NUMT
#define NUMT 4
#endif

#ifndef NUMTRIES
#define NUMTRIES 100
#endif

#define NUM_ELEMENTS_PER_CORE	ARRAYSIZE / NUMT
#define USE_MUL					true


float A[ARRAYSIZE];
float B[ARRAYSIZE];



void
SimdMul( float *a, float *b,   float *c,   int len )
{
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;
	register float *pc = c;
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		_mm_storeu_ps( pc,  _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
		pc += SSE_WIDTH;
	}

	for( int i = limit; i < len; i++ )
	{
		c[i] = a[i] * b[i];
	}
}


float
SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

float
MulSum( float *a, float *b, int len)
{
	float sum[4] = { 0., 0., 0., 0. };
	for (int i = 0; i < len; i++)
	{
		sum[0] += a[i] * b[i];
	}
	return sum[0] + sum[1] + sum[2] + sum[3];
}


int
main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n", );
	return 1
#endif

	float sum_sin, sum_simd, sum_mul;

	for (int i = 0; i < ARRAYSIZE; i++)
	{
		A[i] = 1.;
		B[i] = 2.;
	}

  	// FILE* file = fopen("project4_sin.csv", "a");

	// single core for loop
	double maxMegaMultsPerSecond1 = 0.;
	double time_sin = 0.;

	for (int i = 0; i < NUMTRIES; i++)
	{
		double time0_sin = omp_get_wtime();
		sum_sin = MulSum( &A[0], &B[0], ARRAYSIZE );
		double time1_sin = omp_get_wtime();

		double megaMultsPerSecond1 = float(ARRAYSIZE) / (time1_sin - time0_sin) / 1000000.;
		if (megaMultsPerSecond1 > maxMegaMultsPerSecond1)
		{
			maxMegaMultsPerSecond1 = megaMultsPerSecond1;
			time_sin = time1_sin - time0_sin;
		}
	}

	// fprintf(file, "SING, %d, %lf, %lf, %lf, N/A \n", ARRAYSIZE, sum_sin, time_sin, maxMegaMultsPerSecond1);
	// fclose(file);
 	// FILE* file1 = fopen("project4_simd.csv", "a");

	// SIMD
 
	double maxMegaMultsPerSecond2 = 0.;
	double time_simd = 0.;

	for (int i = 0; i < NUMTRIES; i++)
	{
		double time0_simd = omp_get_wtime();
		sum_simd = SimdMulSum( &A[0], &B[0], ARRAYSIZE );
		double time1_simd = omp_get_wtime();

		double megaMultsPerSecond2 = float(ARRAYSIZE) / (time1_simd - time0_simd) / 1000000.;
		if (megaMultsPerSecond2 > maxMegaMultsPerSecond2)
		{
			maxMegaMultsPerSecond2 = megaMultsPerSecond2;
			time_simd = time1_simd - time0_simd;
		}
	}

	// fprintf(file1, "SIMD, %d, %lf, %lf, %lf, %lf \n", ARRAYSIZE, sum_simd, time_simd, maxMegaMultsPerSecond2, time_sin / time_simd);
  	// fclose(file1);

	// combining SIMD with MUlticore
 
	if (USE_MUL)
	{
 		// FILE* file2 = fopen("project4_mul.csv", "a");
		omp_set_num_threads(NUMT);
		double maxMegaMultsPerSecond3 = 0.;
		sum_mul = 0.;
		double time_mul = 0.;

		for (int i = 0; i < NUMTRIES; i++)
		{
			float sum_temp = 0;
      
			double time0_mul = omp_get_wtime();
			#pragma omp parallel reduction(+:sum_temp)
			{
				int first = omp_get_thread_num() * NUM_ELEMENTS_PER_CORE;
				sum_temp += SimdMulSum( &A[first], &B[first], NUM_ELEMENTS_PER_CORE);
			}
			double time1_mul = omp_get_wtime();

			double megaMultsPerSecond3 = (float)ARRAYSIZE / (time1_mul - time0_mul) / 1000000.;
			if( megaMultsPerSecond3 > maxMegaMultsPerSecond3)
			{
				maxMegaMultsPerSecond3 = megaMultsPerSecond3;
				time_mul = time1_mul - time0_mul;
				sum_mul = sum_temp;
			}
		}

    // fprintf(file2, "MULT, %d, %lf, %lf, %lf, %lf \n", ARRAYSIZE, sum_mul, time_mul, maxMegaMultsPerSecond3, time_sin / time_mul);
    // fclose(file2);
	}

	return 0;
}