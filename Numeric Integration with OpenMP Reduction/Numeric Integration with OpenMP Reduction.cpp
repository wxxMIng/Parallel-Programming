#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//setting the number of threads:
#ifndef NUMT
#define NUMT	4
#endif

//setting the number of nodes:
#ifndef NUMNODES 
#define NUMNODES	1000
#endif

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

#define N	0.70

// function prototypes:
float Height( int, int );

float
Height( int iu, int iv )	// iu,iv = 0 .. NUMNODES-1
{
	float x = -1.  +  2.*(float)iu /(float)(NUMNODES-1);	// -1. to +1.
	float y = -1.  +  2.*(float)iv /(float)(NUMNODES-1);	// -1. to +1.

	float xn = pow( fabs(x), (double)N );
	float yn = pow( fabs(y), (double)N );
	float r = 1. - xn - yn;
	if( r <= 0. )
    return 0.;
	float height = pow( r, 1./(float)N );
	return height;
}

// main program:
int
main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	omp_set_num_threads(NUMT);	// set the number of threads to use in parallelizing the for-loop:

	// the area of a single full-sized tile:
	float fullTileArea = (((XMAX - XMIN) / (float)(NUMNODES - 1)) * ((YMAX - YMIN) / (float)(NUMNODES - 1)));

	// sum up the weighted heights into the variable "volume"
	// using an OpenMP for loop and a reduction:
	double volume = 0.;
	double time0 = omp_get_wtime();

#pragma omp parallel for default(none), shared(fullTileArea), reduction(+:volume)
	for (int i = 0; i < NUMNODES * NUMNODES; i++) {
		int iu = i % NUMNODES;
		int iv = i / NUMNODES;
		float z = Height(iu, iv);
		bool iu_is_edge = iu == 0 || iu == NUMNODES - 1;
    bool iv_is_edge = iv == 0 || iv == NUMNODES - 1;

		// Sum up
		if (iv_is_edge && iu_is_edge){
			// Quarter tile at the corners
			volume += 0.25 * z * fullTileArea;
		}
		else if (iv_is_edge || iu_is_edge){
			// Half tile at the edges
			volume += 0.5 * z * fullTileArea;
		}
		else{
			// Full tile in the middle
			volume += z * fullTileArea;
		}
	}
	
  volume *= 2;	

	double time1 = omp_get_wtime();
	double megaHeightPerSecond = (double)(NUMNODES * NUMNODES) / (time1 - time0) / 1000000.;
	fprintf(stderr, "%2d threads , %2d Nodes : Volume = %.6lf ; MegaHeights/Sec : %.6lf \n", NUMT, NUMNODES, volume, megaHeightPerSecond);
}