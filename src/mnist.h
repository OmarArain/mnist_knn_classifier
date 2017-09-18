#ifndef MNIST_H
#define MNIST_H
//specs from http://yann.lecun.com/exdb/mnist/
#define LABELS "-labels-idx1-ubyte"
#define IMAGES "-images-idx3-ubyte"
#define LBL_MAGIC_NUM 0x00000801
#define IMG_MAGIC_NUM 0x00000803
#define LBL_HEADER_SIZE 8
#define IMG_HEADER_SIZE 16
// #define MNIST_X 28
// #define MNIST_Y 28
#define MN_IX 0
#define NUM_IMG_IX 1
#define X_IX 2
#define Y_IX 3

//choose the endian converitng function
#define MY_NTOHL ntohl 
#define MY_HTONL htonl



#include <stdbool.h>

struct mnist_dataset_t;

/// Define struct mnist_dataset_t in your mnist.c file
typedef struct mnist_dataset_t * mnist_dataset_handle;

#define MNIST_DATASET_INVALID ((mnist_dataset_handle) 0)

/// Define struct mnist_image_t in your mnist.c file
struct mnist_image_t;
typedef struct mnist_image_t * mnist_image_handle;

#define MNIST_IMAGE_INVALID ((mnist_image_handle) 0)

/// Open the given MNIST formatted dataset;
/// Returns MNIST_INVALID if the dataset could not be opened,
/// and a mnist_dataset_handle on success.
/// The handle must be freed using mnist_free().
/// 
/// For dataset 'name', the following files must exist:
///   - name-labels-idx1-ubyte
///   - name-images-idx3-ubyte
/// in other words, to open the dataset represented by the files above,
/// the user would call mnist_open("name")
///
/// The open function checks the 'magic number' at the beginning of the file
/// and returns MNIST_DATASET_INVALID if the number does not match
/// expectations.
mnist_dataset_handle mnist_open (const char * name);


/// Create a new empty dataset.
/// This only creates the in-memory representation of the dataset.
/// No actual files need to be created.
/// x and y refer to the image size for this dataset.
/// Returns MNIST_DATASET_INVALID if an error occurred.
///
/// In case of success, the resulting handle needs to be freed
/// using mnist_free.
mnist_dataset_handle mnist_create(unsigned int x, unsigned int y);

/// Free the memory associated with the handle
void mnist_free (mnist_dataset_handle handle);


/// Return the number of images in the datasets
/// Returns <0 if handle is equal to MNIST_DATASET_INVALID
int mnist_image_count (const mnist_dataset_handle handle);

/// Return the size of each image in the dataset; Stores the size in X and Y.
/// Sets *x and *y to 0 if handle == MNIST_DATASET_INVALID
void mnist_image_size (const mnist_dataset_handle handle,
                       unsigned int * x, unsigned int * y);

/// Return a handle to the first image in the dataset;
/// Note: there is no requirement to free the returned handle.
/// If there are no images in the dataset, or if handle ==
/// MNIST_DATASET_INVALID, or any other error occurs,
/// return MNIST_IMAGE_INVALID
mnist_image_handle mnist_image_begin (const mnist_dataset_handle handle);

/// Return a pointer to the data for the image. The data should not be copied
/// and the user of the data should not modify or free it. The return pointer
/// should point to image_size_x * image_size_y bytes, in the same order as
/// specified in the MNIST file format.
/// The pointer must remain valid even after mnist_image_next() is called.
/// However, when mnist_free() is called, all pointers returned by this
/// function become invalid as well.
const unsigned char * mnist_image_data (const mnist_image_handle h);

/// Obtain the label of the image.
/// Return <0 if handle is equal to MNIST_IMAGE_INVALID.
int mnist_image_label (const mnist_image_handle h);

/// Return the handle of the next image in the dataset, or MNIST_IMAGE_INVALID
/// if there are no more images.
mnist_image_handle mnist_image_next (const mnist_image_handle h);


/// Add a new image *after* the image represented by mnist_image_handle
/// Returns a handle referring to the new image
/// 
///  If i == MNIST_IMAGE_INVALID, add as the very first image of the
///             dataset (without replacing an already existing image).
///
///  imagedata: point to the image data. The data is *copied* into
///             the data structure (i.e. the imagedata pointer doesn't
///             need to remain valid after the call returns.
///
///  x:         number of columns in the image
///  y:         number of rows in the image.
///
/// returns MNIST_IMAGE_INVALID if there was an issue.
///  (for example, x and y don't match the sizes for which the dataset
///   was create).
///
mnist_image_handle mnist_image_add_after (mnist_dataset_handle h,
      mnist_image_handle i,
      const unsigned char * imagedata, unsigned int x, unsigned int y,
      unsigned int label);

/// Persist the specified dataset to file.
///
/// Name follows the same convention as for mnist_open.
/// i.e. mnist_save(h, "name"); should create two files (a label and
/// image file) with appropriate
/// file names.
///
/// If successful, returns true. If any error happens (including I/O)
/// returns false.
///
/// If the function fails, there should be no memory leaks.
///
/// If the dataset already exists, this function will overwrite it without
/// warning.
bool mnist_save(const mnist_dataset_handle h, const char * filename);

/// Function that takes a number N ≤ total number of images and 
/// returns a NEW dataset that consists of N randomly selected images from 
// 	old the dataset.

mnist_dataset_handle mnist_create_sample (const mnist_dataset_handle h,
		unsigned int n);


//returns an pseudo-random integer that is uniformly distributed
// in N bins (i.e the range [0,N))
int _uniform_rand_int(int range);


//shuffles an int array using Fisher-Yates algo in O(N) time
void _fisher_yates_shuffle(int *array, int length);

#endif
