#ifndef KNN_H
#define KNN_H
#define KNN_INVALID NULL
#define LABEL_INVALID -1
#include "distance.h"
#include "mnist.h"
/*
For the k-NN algorithm, we will have:
• a set of training images, for which know the corresponding label.
• a set of test images, for which we are trying to determine the label 
	and track the accuracy of the classification.
• a distance metric (represented by a distance_t variable).
• a parameter k

Note that while we do know the label for the test images, 
	we will only use this knowledge to determine if the k-NN algorithm 
	was able to correctly identify the image or not.
Your implementation of the k-NN algorithm should do the following, 
	for each image in the collection of test images:
• Calculate the distance between the selected image of the test set and 
	all the images of the training set.
• For the k images – for which we know the corresponding label – closest 
	to the unknown image, find the label which occurred the most. 
	If all labels occurred the same number of times, pick the label 
	corresponding to the closest image.

For each test image, determine if the guess produced by the k-NN algorithm
 	matches the known label of the image. We define the accuracy as:

*/


typedef struct knn_data * knn_data_t;

int partition(double ix_list[], int data_list[], 
				int left, int right, int pivot_ix);

double quickselect(double ix_list[], int data_list[], 
				int left, int right, int k);

knn_data_t knn_data_create(mnist_image_handle train_img,
						   mnist_dataset_handle test_dataset);

void knn_data_free(knn_data_t k);

double * knn_data_get_distances(knn_data_t knn, distance_t distance);

int knn_data_best_label(knn_data_t knn, int k, distance_t distance);

#endif
