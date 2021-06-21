#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int	NowYear;		// 2021 - 2026
int	NowMonth;		// 0 - 11
int Index;

float	NowPrecip;		// inches of rain per month
float	NowTemp;		  // temperature this month
float	NowHeight;		// grain height in inches
int		NowNumDeer;		// number of deer in the current population
int 	NowNumFarmer;		// number of wolf in the current population

int 	NumInThreadTeam;
int 	NumAtBarrier;
int 	NumGone;
omp_lock_t Lock;


const float GRAIN_GROWS_PER_MONTH =		9.0;
const float GRAIN_GROWS_PER_FARMER =  	5.0;	
const float ONE_DEER_EATS_PER_MONTH =	1.0;
const float ONE_FARMER_EATS_PER_MONTH = 0.5;

const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =				2.0;	// plus or minus noise

const float AVG_TEMP =					60.0;	// average
const float AMP_TEMP =					20.0;	// plus or minus
const float RANDOM_TEMP =				10.0;	// plus or minus noise

const float MIDTEMP =					40.0;	
const float MIDPRECIP =					10.0;	


// function prototype:
void  InitBarrier( int n );
void  WaitBarrier();
void  Deer();
void  Grain();
void  Watcher();
void  Farmer();
void  Update_Temp_Precip();
float SQR( float x);
float Ranf( unsigned int *seedp, float low, float high );
int   Ranf( unsigned int *seedp, int ilow, int ihigh);


unsigned int seed = 0;
float x = Ranf(&seed, -1.f, 1.f);

FILE* file = fopen("project 3.csv", "w");

// main program:
int
main(int argc, char const *argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif	

	
	fprintf(file, "Index, Year, Month, Temp, Precip, Height, NumDeer, NumFarmer\n");
	// starting date and time:
	NowMonth =    0;
	NowYear  = 2021;
	Index = 0;
	// starting state (feel free to change this if you want):
	NowNumDeer 	 = 20;
	NowNumFarmer = 5;
	NowHeight    = 500.;
	Update_Temp_Precip();

	omp_set_num_threads( 4 );	// same as # of sections
	InitBarrier( 3 );

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			Deer( );
		}

		#pragma omp section
		{
			Grain( );
		}

		#pragma omp section
		{
			Watcher( );
		}

		#pragma omp section
		{
			Farmer( );
		}
	}   // implied barrier -- all functions must return in order
		// to allow any of them to get past here

	fclose(file);
	return 0;
}


void 
InitBarrier( int n)
{
	NumInThreadTeam = n;
	NumAtBarrier = 0;
	omp_init_lock( &Lock );
}

void
WaitBarrier()
{
	omp_set_lock( &Lock );
	{
		NumAtBarrier++;
		if (NumAtBarrier == NumInThreadTeam) 	 //release the waiting threads
		{
			NumGone = 0;
			NumAtBarrier = 0;
			//let all other threads return before this unblocks:
			while( NumGone != NumInThreadTeam -1);
			omp_unset_lock( &Lock );
			return;
		}
	}
	omp_unset_lock( &Lock );

	while( NumAtBarrier != 0 );		// all threads wait here until the last one arrives ...

	#pragma omp atomic		// ... and sets NumAtBarrier to 0
		NumGone++;
}

void
Farmer()
{
	while( NowYear < 2027 )
	{
		// compute the next variables
		int nextNumFarmer = NowNumFarmer;
		int carryingCapacity = (int)( NowNumDeer / 5 ) ;
		
    	if ( nextNumFarmer < carryingCapacity )
			nextNumFarmer++;
      
		if ( nextNumFarmer > carryingCapacity )
			nextNumFarmer--;

		if ( nextNumFarmer < 0 )
			nextNumFarmer = 0;

		WaitBarrier();		// DoneComputing barrier
		
		// copy the next state into the now variables
		NowNumFarmer = nextNumFarmer;
		WaitBarrier();		// DoneAssigning barrier
		
    // do nothing
    WaitBarrier();		// DonePrinting barrier
	}
}

void
Deer()
{
	while( NowYear < 2027 )
	{
		// compute the next variables
		int nextNumDeer = NowNumDeer;
		int carryingCapacity = (int)NowHeight;
		
		if ( nextNumDeer < carryingCapacity )
			nextNumDeer += 2;
   
		if ( nextNumDeer > carryingCapacity )
			nextNumDeer--;

		nextNumDeer -= NowNumFarmer * (int)ONE_FARMER_EATS_PER_MONTH;
		
		if (nextNumDeer < 0)
			nextNumDeer = 0;

		WaitBarrier();		// DoneComputing barrier

		// copy the next state into the now variables
		NowNumDeer = nextNumDeer;
		WaitBarrier();		// DoneAssigning barrier
		
		// do nothing
		WaitBarrier();		// DonePrinting barrier
	}
}

void
Grain()
{
	while( NowYear < 2027 )
	{
		// compute the next variables
		float tempFactor = exp( -SQR( (NowTemp - MIDTEMP) / 10.) );
		float precipFactor = exp( -SQR( (NowPrecip - MIDPRECIP) / 10. ) );
		float nextHeight = NowHeight;

		nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
    	nextHeight += (float)NowNumFarmer * GRAIN_GROWS_PER_FARMER;
		nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
		
		if ( nextHeight < 0. )
			nextHeight = 0.;

		WaitBarrier();		// DoneComputing barrier
		
		// copy the next state into the now variables
		NowHeight = nextHeight;
		WaitBarrier();		// DoneAssigning barrier
		
		// do nothing
		WaitBarrier();		// DonePrinting barrier
	}

}

void
Watcher()
{
	// when finished
	while( NowYear < 2027 )
	{
		// do nothing
		WaitBarrier();		// DoneComputing barrier

		// do nothing
		WaitBarrier();		// DoneAssigning barrier

		// advance time and re-compute all environmental variables
		Update_Temp_Precip();
		WaitBarrier();		// DonePrinting barrier
	}
}

void  
Update_Temp_Precip( )
{
	fprintf(stderr, "%d NowYear %d, NowMonth %d, Temp %f, Precip %f, NowHeight %f, NowNumDeer %d, NowNumFarmer %d\n", Index, NowYear, NowMonth, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumFarmer);
	fprintf(file, "%d, %d,  %d,  %f,  %f,  %f,  %d,  %d\n", Index, NowYear, NowMonth, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumFarmer);

	if ( NowMonth == 12 )
	{
		NowMonth = 1;
		NowYear++;
	}
	else
	{
		NowMonth++;
	}
	Index++;

	float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

	float temp = AVG_TEMP - AMP_TEMP * cos( ang );
	NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

	float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
	NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
	if( NowPrecip < 0. )
		NowPrecip = 0.;

}


float
SQR( float x )
{
	return x*x;
}

float 
Ranf( unsigned int *seedp, float low, float high )
{
	float r = (float)rand_r( seedp );
	return( low + r * (high - low) / (float)RAND_MAX );
}

int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;

    return (int)( Ranf(seedp, low,high) );
}

