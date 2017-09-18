#ifndef DISTANCE_H
#define DISTANCE_H
#include <stdlib.h>
// hardcoded parameters for distance functions
#define DOWNSAMPLE_FREQ 2
#define CROP_SIZE 4
#define THRESHOLD_LVL 127

//descriptions. UPDATE THESE along with
// create_distance_function WHEN ADDING FUNCTIONS
#define DISTANCE_H_FUNCS {"euclid", "reduced"}
#define DISTANCE_H_NUM_FUNCS 2
#define EUCLID_D  		"euclid: euclidean distance of each the pixel values\n"

#define REDUCED_D 		"reduced: absoulute value of the difference of the sum" \
				  		" of pixel values\n"
#define DOWNSAMPLE_D 	"downsample: uniformly reduces the number of pixels " \
				  	 	"by a factor of " STR(DOWNSAMPLE_FREQ) \
				  	 	" and calculates euclidean distance.\n"

#define CROP_D 			"crop: removes the outermost " STR(CROP_SIZE) \
			   			" pixels of the image, and calculates euclidean distance.\n"

#define THRESHOLD_D 	"threshold: counts pixels that have a value greater than " \
			   			STR(THRESHOLD_LVL) ".\n"
#define DISTANCE_H_LIB_DESC EUCLID_D REDUCED_D //DOWNSAMPLE_D CROP_D THRESHOLD_D

typedef unsigned int uint;

typedef double (*distance_t)(const unsigned char * img1data, 
						const unsigned char * img2data, uint x, uint y);

// returns a pointer to a distance function when given a string 
// naming the distance function desired. Returns NULL if distance
// function not implemented.
distance_t create_distance_function(const char * schemename);


// returns a string describing all of the implemented distance functions
// in the library.
char * describe_distance_functions();

#endif

