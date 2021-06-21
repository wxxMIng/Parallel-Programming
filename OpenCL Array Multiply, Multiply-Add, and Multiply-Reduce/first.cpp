// 1. Program header

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <omp.h>

#include "cl.h"
#include "cl_platform.h"


#ifndef NMB
#define	NMB			8192
#endif

// 1K~8M
#define NUM_ELEMENTS	NMB*1024		

// 32~256
#ifndef LOCAL_SIZE
#define	LOCAL_SIZE		128		
#endif

#define	NUM_WORK_GROUPS		NUM_ELEMENTS/LOCAL_SIZE

const char *			CL_FILE_NAME = { "first.cl" };
const float				TOL = 0.0001f;

void		Wait( cl_command_queue );
int			LookAtTheBits( float );
void		PrintOpenclInfo();
void		SelectOpenclDevice();
char*		Vendor(cl_uint);
char*		Type(cl_device_type);

// globals:
cl_platform_id   Platform;
cl_device_id     Device;

// opencl vendor ids:
#define ID_AMD          0x1002
#define ID_INTEL        0x8086
#define ID_NVIDIA       0x10de

#ifdef MAIN_PROGRAM_TO_TEST
// compile with:
// g++ -o printinfo printinfo.cpp /usr/local/apps/cuda/10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
int
main(int argc, char* argv[])
{
	PrintOpenclInfo();
	SelectOpenclDevice();
	return 0;
}
#endif

int
main( int argc, char *argv[ ] )
{
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE *fp;
#ifdef WIN32
	errno_t err = fopen_s( &fp, CL_FILE_NAME, "r" );
	if( err != 0 )
#else
	fp = fopen( CL_FILE_NAME, "r" );
	if( fp == NULL )
#endif
	{
		fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
		return 1;
	}

	cl_int status;		// returned status from opencl calls
				// test against CL_SUCCESS

	// get the platform id:

	cl_platform_id platform;
	status = clGetPlatformIDs( 1, &platform, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
	
	// get the device id:

	cl_device_id device;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );

	// 2. allocate the host memory buffers:

	float *hA = new float[ NUM_ELEMENTS ];
	float *hB = new float[ NUM_ELEMENTS ];
	float *hC = new float[ NUM_ELEMENTS ];
	//float *hD = new float[ NUM_ELEMENTS ];
	float* hD = new float[ NUM_WORK_GROUPS ];

	size_t dSize  = NUM_WORK_GROUPS * sizeof(float);

	// fill the host memory buffers:

	for( int i = 0; i < NUM_ELEMENTS; i++ )
	{
		hA[i] = hB[i] = hC[i] = (float) sqrt(  (double)i  );
	}

	size_t dataSize = NUM_ELEMENTS * sizeof(float);

	// 3. create an opencl context:

	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer( context, CL_MEM_READ_ONLY,  dataSize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

	cl_mem dB = clCreateBuffer( context, CL_MEM_READ_ONLY,  dataSize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (2)\n" );

	cl_mem dC = clCreateBuffer( context, CL_MEM_READ_ONLY, dataSize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (3)\n" );

	cl_mem dD = clCreateBuffer( context, CL_MEM_WRITE_ONLY, dSize, NULL, &status );
	if ( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (4)\n" );

	// 6. enqueue the 3 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer( cmdQueue, dA, CL_FALSE, 0, dataSize, hA, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

	status = clEnqueueWriteBuffer( cmdQueue, dB, CL_FALSE, 0, dataSize, hB, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (2)\n" );

	status = clEnqueueWriteBuffer( cmdQueue, dC, CL_FALSE, 0, dataSize, hC, 0, NULL, NULL );
	if ( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (3)\n" );

	Wait( cmdQueue );

	// 7. read the kernel code from a file:

	fseek( fp, 0, SEEK_END );
	size_t fileSize = ftell( fp );
	fseek( fp, 0, SEEK_SET );
	char *clProgramText = new char[ fileSize+1 ];		// leave room for '\0'
	size_t n = fread( clProgramText, 1, fileSize, fp );
	clProgramText[fileSize] = '\0';
	fclose( fp );
	if( n != fileSize )
		fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

	// create the text for the kernel program:

	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateProgramWithSource failed\n" );
	delete [ ] clProgramText;

	// 8. compile and link the kernel code:

	char *options = { "" };
	status = clBuildProgram( program, 1, &device, options, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		size_t size;
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
		cl_char *log = new cl_char[ size ];
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
		fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
		delete [ ] log;
	}

	// 9. create the kernel object:

	cl_kernel kernel = clCreateKernel( program, "ArrayMultReduce", &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateKernel failed\n" );

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dA );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (1)\n" );

	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dB );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (2)\n" );

	status = clSetKernelArg( kernel, 2, sizeof(cl_mem), &dC );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (3)\n" );

	status = clSetKernelArg( kernel, 3, LOCAL_SIZE * sizeof(float), NULL ); // tell OpenCL that this is a local array
	status = clSetKernelArg( kernel, 4, sizeof(cl_mem), &dD );
	if ( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (4)\n" );

	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

	Wait( cmdQueue );
	double time0 = omp_get_wtime( );

	time0 = omp_get_wtime( );

	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait( cmdQueue );
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer( cmdQueue, dD, CL_TRUE, 0, NUM_WORK_GROUPS * sizeof(float), hD, 0, NULL, NULL );
	
	if( status != CL_SUCCESS )
			fprintf( stderr, "clEnqueueReadBuffer failed\n" );

	// add up
	float sum = 0.;
	for (int i = 0; i < NUM_WORK_GROUPS; i++)
	{
		sum += hD[i];
	}
	
 
	// did it work?
	/*
	float sum1 = 0.;
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		sum1 += hA[i] * hB[i];
	}

	fprintf(stderr, "sum = %10.3lf\tsum1 = %10.3lf\n", sum, sum1);
	
	/*
	for( int i = 0; i < NUM_ELEMENTS; i++ )
	{
		float expected = (hA[i] * hB[i]) + hC[i];
		if( fabs( (long)(hD[i] - expected) ) > TOL )
		{
			fprintf( stderr, "%4d: %13.6f * %13.6f + %13.6f wrongly produced %13.6f instead of %13.6f (%13.8f)\n",
				i, hA[i], hB[i], hC[i], hD[i], expected, fabs(hD[i]-expected) );
			fprintf( stderr, "%4d:    0x%08x *    0x%08x wrongly produced    0x%08x instead of    0x%08x\n",
				i, LookAtTheBits(hA[i]), LookAtTheBits(hB[i]), LookAtTheBits(hD[i]), LookAtTheBits(expected) );
		}
	}
	*/
	fprintf( stderr, "%8d\t%4d\t%10d\t%10.3lf GigaMultsPerSecond\n",
		NMB, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS/(time1-time0)/1000000000. );

#ifdef WIN32
	Sleep( 2000 );
#endif


	// 13. clean everything up:

	clReleaseKernel(        kernel   );
	clReleaseProgram(       program  );
	clReleaseCommandQueue(  cmdQueue );
	clReleaseMemObject(     dA  );
	clReleaseMemObject(     dB  );
	clReleaseMemObject(     dC  );
	clReleaseMemObject(		dD	);

	delete [ ] hA;
	delete [ ] hB;
	delete [ ] hC;
	delete [ ] hD;

	return 0;
}


int
LookAtTheBits( float fp )
{
	int *ip = (int *)&fp;
	return *ip;
}


// wait until all queued tasks have taken place:

void
Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clWaitForEvents failed\n" );
}


void
PrintOpenclInfo()
{
	cl_int status;		// returned status from opencl calls
						// test against CL_SUCCESS
	fprintf(stderr, "PrintInfo:\n");

	// find out how many platforms are attached here and get their ids:

	cl_uint numPlatforms;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (1)\n");

	fprintf(stderr, "Number of Platforms = %d\n", numPlatforms);

	cl_platform_id* platforms = new cl_platform_id[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2)\n");

	for (int p = 0; p < (int)numPlatforms; p++)
	{
		fprintf(stderr, "Platform #%d:\n", p);
		size_t size;
		char* str;

		clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, size, str, NULL);
		fprintf(stderr, "\tName    = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, size, str, NULL);
		fprintf(stderr, "\tVendor  = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, size, str, NULL);
		fprintf(stderr, "\tVersion = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE, size, str, NULL);
		fprintf(stderr, "\tProfile = '%s'\n", str);
		delete[] str;


		// find out how many devices are attached to each platform and get their ids:

		cl_uint numDevices;

		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		fprintf(stderr, "\tNumber of Devices = %d\n", numDevices);

		cl_device_id* devices = new cl_device_id[numDevices];
		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		for (int d = 0; d < (int)numDevices; d++)
		{
			fprintf(stderr, "\tDevice #%d:\n", d);
			size_t size;
			cl_device_type type;
			cl_uint ui;
			size_t sizes[3] = { 0, 0, 0 };

			clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
			fprintf(stderr, "\t\tType = 0x%04x = ", (unsigned int)type);
			switch (type)
			{
			case CL_DEVICE_TYPE_CPU:
				fprintf(stderr, "CL_DEVICE_TYPE_CPU\n");
				break;
			case CL_DEVICE_TYPE_GPU:
				fprintf(stderr, "CL_DEVICE_TYPE_GPU\n");
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				fprintf(stderr, "CL_DEVICE_TYPE_ACCELERATOR\n");
				break;
			default:
				fprintf(stderr, "Other...\n");
				break;
			}

			clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Vendor ID = 0x%04x ", ui);
			switch (ui)
			{
			case ID_AMD:
				fprintf(stderr, "(AMD)\n");
				break;
			case ID_INTEL:
				fprintf(stderr, "(Intel)\n");
				break;
			case ID_NVIDIA:
				fprintf(stderr, "(NVIDIA)\n");
				break;
			default:
				fprintf(stderr, "(?)\n");
			}

			clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Compute Units = %d\n", ui);

			clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Item Dimensions = %d\n", ui);

			clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Item Sizes = %d x %d x %d\n", sizes[0], sizes[1], sizes[2]);

			clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Group Size = %d\n", size);

			clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Clock Frequency = %d MHz\n", ui);

			size_t extensionSize;
			clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
			char* extensions = new char[extensionSize];
			clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);
			fprintf(stderr, "\nDevice #%d's Extensions:\n", d);
			for (int e = 0; e < (int)strlen(extensions); e++)
			{
				if (extensions[e] == ' ')
					extensions[e] = '\n';
			}
			fprintf(stderr, "%s\n", extensions);
			delete[] extensions;
		}
		delete[] devices;
	}
	delete[] platforms;
	fprintf(stderr, "\n\n");
}

void
SelectOpenclDevice()
{
	// select which opencl device to use:
	// priority order:
	//	1. a gpu
	//	2. an nvidia or amd gpu
	//	3. an intel gpu
	//	4. an intel cpu

	int bestPlatform = -1;
	int bestDevice = -1;
	cl_device_type bestDeviceType;
	cl_uint bestDeviceVendor;
	cl_int status;		// returned status from opencl calls
				// test against CL_SUCCESS

	fprintf(stderr, "\nSelecting the OpenCL Platform and Device:\n");

	// find out how many platforms are attached here and get their ids:

	cl_uint numPlatforms;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (1)\n");

	cl_platform_id* platforms = new cl_platform_id[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2)\n");

	for (int p = 0; p < (int)numPlatforms; p++)
	{
		// find out how many devices are attached to each platform and get their ids:

		cl_uint numDevices;

		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		cl_device_id* devices = new cl_device_id[numDevices];
		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		for (int d = 0; d < (int)numDevices; d++)
		{
			cl_device_type type;
			cl_uint vendor;
			size_t sizes[3] = { 0, 0, 0 };

			clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

			clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);

			// select:

			if (bestPlatform < 0)		// not yet holding anything -- we'll accept anything
			{
				bestPlatform = p;
				bestDevice = d;
				Platform = platforms[bestPlatform];
				Device = devices[bestDevice];
				bestDeviceType = type;
				bestDeviceVendor = vendor;
			}
			else					// holding something already -- can we do better?
			{
				if (bestDeviceType == CL_DEVICE_TYPE_CPU)		// holding a cpu already -- switch to a gpu if possible
				{
					if (type == CL_DEVICE_TYPE_GPU)			// found a gpu
					{										// switch to the gpu we just found
						bestPlatform = p;
						bestDevice = d;
						Platform = platforms[bestPlatform];
						Device = devices[bestDevice];
						bestDeviceType = type;
						bestDeviceVendor = vendor;
					}
				}
				else										// holding a gpu -- is a better gpu available?
				{
					if (bestDeviceVendor == ID_INTEL)			// currently holding an intel gpu
					{										// we are assuming we just found a bigger, badder nvidia or amd gpu
						bestPlatform = p;
						bestDevice = d;
						Platform = platforms[bestPlatform];
						Device = devices[bestDevice];
						bestDeviceType = type;
						bestDeviceVendor = vendor;
					}
				}
			}
		}
		delete[] devices;
	}
	delete[] platforms;


	if (bestPlatform < 0)
	{
		fprintf(stderr, "Found no OpenCL devices!\n");
	}
	else
	{
		fprintf(stderr, "I have selected Platform #%d, Device #%d\n", bestPlatform, bestDevice);
		fprintf(stderr, "Vendor = %s, Type = %s\n", Vendor(bestDeviceVendor), Type(bestDeviceType));
	}
}

char*
Vendor(cl_uint v)
{
	switch (v)
	{
	case ID_AMD:
		return (char*)"AMD";
	case ID_INTEL:
		return (char*)"Intel";
	case ID_NVIDIA:
		return (char*)"NVIDIA";
	}
	return (char*)"Unknown";
}

char*
Type(cl_device_type t)
{
	switch (t)
	{
	case CL_DEVICE_TYPE_CPU:
		return (char*)"CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU:
		return (char*)"CL_DEVICE_TYPE_GPU";
	case CL_DEVICE_TYPE_ACCELERATOR:
		return (char*)"CL_DEVICE_TYPE_ACCELERATOR";
	}
	return (char*)"Unknown";
}