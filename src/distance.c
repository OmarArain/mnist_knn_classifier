#include "distance.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <float.h>

#ifdef DEBUG
  #define dprint(fmt, ...) printf("debug: %s:"  fmt "\n", __func__,  __VA_ARGS__)
#else
  #define dprint(fmt, ...) do {} while(0)
#endif

//stringify
#define _STR(x) #x
#define STR(x) _STR(x)
//argchecking function
#define ARGCHECK(img1data, img2data, x, y) 	\
	if (!img1data || !img2data) 			\
	{ errno = EINVAL; \
	   printf("invalid img passed to distance function."); \
	   return DBL_MAX; } 			\
	if (x==0 || y==0) { errno = EINVAL; \
		printf("invalid dimensions passed to distance function."); \
		return 0; }


static double euclid(const unsigned char * img1data,
        	const unsigned char * img2data, uint x, uint y)
{

	ARGCHECK(img1data, img2data, x, y);

	uint num_dims = x*y;
	double sum = 0;
	double sqdiff = 0;
	for(int p=0;p<num_dims;p++)
	{
		sqdiff = pow((double)(img1data[p] - img2data[p]),2);
		sum +=sqdiff;
	}

	dprint("img1data:%p\timg2data:%p\tx:%u\ty:%u\tsum:%f\tsqrt(sum):%f",
			(void*)img1data, (void*)img2data, x, y, sum, sqrt(sum));
	return sqrt(sum);
}

static double reduced(const unsigned char * img1data,
        	const unsigned char * img2data, uint x, uint y)
{

	ARGCHECK(img1data, img2data, x, y);

	uint num_dims = x*y;
	double img1_sum = 0, img2_sum=0;
	for(int p=0;p<num_dims;p++)
	{
		img1_sum += (double)(img1data[p]);
		img2_sum += (double)(img2data[p]);
	}

	dprint("img1data:%p\timg2data:%p\tx:%u\ty:%u\timg1_sum:%f\timg2_sum:%f\tfabs():%f",
			(void*)img1data, (void*)img2data, x, y, 
			img1_sum, img2_sum, fabs(img1_sum-img2_sum));
	return fabs(img1_sum - img2_sum);	
	
	return 0;
}

// static double downsample(const unsigned char * img1data,
//         	const unsigned char * img2data, uint x, uint y)
// {
// 	ARGCHECK(img1data, img2data, x, y);	
// 	return 0;
// }

// static double crop(const unsigned char * img1data,
//         	const unsigned char * img2data, uint x, uint y)
// {
// 	ARGCHECK(img1data, img2data, x, y);
// 	return 0;
// }

// static double threshold(const unsigned char * img1data,
//         	const unsigned char * img2data, uint x, uint y)
// {
// 	ARGCHECK(img1data, img2data, x, y);
// 	return 0;
// }



distance_t create_distance_function(const char * schemename)
{

	if 		(strcmp(schemename, "euclid") == 0) return euclid;
	else if (strcmp(schemename, "reduced") == 0) return reduced;
	// else if (strcmp(schemename, "downsample") == 0) return downsample;
	// else if (strcmp(schemename, "crop") == 0) return crop;
	// else if (strcmp(schemename, "threshold") == 0) return threshold;
	else	
	{
		printf("%s is not a valid distance function.\n", schemename); 
		return NULL;
	}
}

char * describe_distance_functions()
{
	return DISTANCE_H_LIB_DESC;
}
