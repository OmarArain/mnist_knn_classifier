#include "knn.h"
#include "mnist.h"
#include "distance.h"
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <float.h>
#include <stdio.h>


#ifndef dprint
	#ifdef DEBUG
	  #define dprint(fmt, ...) printf("debug: %s:"  fmt "\n", __func__, \
	  				 __VA_ARGS__)
	#else
	  #define dprint(fmt, ...) do {} while(0)
	#endif
#endif

#define NUM_IMG_LABELS 10
#define SWAP(x, y, TYPE) do {TYPE _t=x; x=y ; y=_t;} while (0)

//convenience
typedef unsigned int uint;
typedef unsigned char uchar;

struct knn_data
{
	//k nearest distances
	double * distances;
	// k nearest labels
	int * labels;

	//closest distance of each group with label [i] 
	// i.e. min_dist[1] is the smallest distance of a 
	// test_img with label=1.  Use this to break ties.
	double min_dist[NUM_IMG_LABELS];

	mnist_image_handle train_img;
	mnist_dataset_handle test_dataset;

};


knn_data_t knn_data_create(mnist_image_handle train_img,
						   mnist_dataset_handle test_dataset)
{
	int num_imgs = mnist_image_count(test_dataset);
	if (num_imgs==0) return KNN_INVALID;
	if (train_img==MNIST_IMAGE_INVALID) return KNN_INVALID;

	knn_data_t knn = malloc(sizeof(struct knn_data));
	double * distances = malloc(num_imgs*sizeof(double));
	if(!distances) {errno=ENOMEM; return KNN_INVALID;}
	int * labels = malloc(num_imgs*sizeof(int));
	if(!labels) {errno=ENOMEM; free(distances); return KNN_INVALID;}


	for(int i=0; i<num_imgs; i++)
	{
		distances[i] = DBL_MAX;
		labels[i] = LABEL_INVALID;
	}

	//initialize/assign values

	knn->distances = distances;
	knn->labels = labels;
	knn->train_img = (mnist_image_handle) train_img;
	knn->test_dataset = (mnist_dataset_handle) test_dataset;
	for(int i=0 ; i<NUM_IMG_LABELS; i++) {knn->min_dist[i] = DBL_MAX;}

	return knn;
}

void knn_data_free(knn_data_t k)
{
	if(k!=KNN_INVALID)
	{
		free(k->distances);
		free(k->labels);
		//get rid of potentially dangliing pointers
		k->train_img=NULL;
		k->test_dataset=NULL;
		free(k);
	}
}


int partition(double ix_list[], int data_list[], int left, int right, int pivot_ix)
{
	//partition algo used in quickselect. I use the dist_list as the indexed
	// list, and partition the label list along with it
	// given a pivot index, the algo will move everything less than pivot_value
	// to the left of the list and everything greater to the right.
	
	if(left>right) return -1;
	// if((pivot_ix<left) || (pivot_ix > right)) {puts("ERR1");return -1;}

	double pivot_val = ix_list[pivot_ix];

	SWAP(ix_list[pivot_ix], ix_list[right], double);
	SWAP(data_list[pivot_ix], data_list[right], int);

	int store_ix = left;
	for(int i=left; i<right; i++)
	{
		if(ix_list[i] < pivot_val)
		{
			SWAP(ix_list[store_ix], ix_list[i], double);
			SWAP(data_list[store_ix], data_list[i], int);	
		
			store_ix++;
		}
	}

	SWAP(ix_list[right], ix_list[store_ix], double);
	SWAP(data_list[right], data_list[store_ix], int);

	dprint("left:%d\tright:%d\tpivot_ix:%d\tstore_ix:%d\tpivot_val:%f", 
			left, right, pivot_ix, store_ix, pivot_val);
	return store_ix;
}


double quickselect(double ix_list[], int data_list[], int left, int right, int k)
{
	/*quickselect algo.  This is used to speed up knn algo.
	Picks the (k+1) smallest element of a list in O(N) on average. 
	References:
	http://stats.stackexchange.com/questions/219655/k-nn-computational-complexity
	https://en.wikipedia.org/wiki/Quickselect
	*/
	dprint("left: %d\tright: %d\t k:%d",left,right,k);
	if(left>right) 
	{
		printf("error: quickselect: left >right. Swapping values."); 
	 	// SWAP(left, right, int);
	 	exit(EXIT_FAILURE);
	}
	if(k<left)
	{
		printf("error: quickselect: k:%d < left:%d.\n",k,left); 
	 	exit(EXIT_FAILURE);
	}
	if(k>right)
	{
		printf("error: quickselect: k > right.\n"); 
		exit(EXIT_FAILURE);		
	}

	while(true)
	{
		// dprint("left: %d\t right: %d", left, right);
		if (left==right) return ix_list[left];
		 // it is likely that the data may be presorted, from previous
		 // calls to knn_data_best_label, thus set pivot_ix = k. 
		 // otherwise left + floor(rand() % (right - left + 1));
		int pivot_ix = k; 		
		pivot_ix = partition(ix_list, data_list, left, right, pivot_ix);
		// dprint("pivot_ix:%d\tk:%d\tix_list[k]:%f", pivot_ix,k,ix_list[k]);
		if (k == pivot_ix) {return ix_list[k];}
		else if (k < pivot_ix) {right = pivot_ix - 1;}
		else {left = pivot_ix + 1;}
	}
}


double * knn_data_get_distances(knn_data_t knn, distance_t distance)
{
	//calculates the distances between the train image
	// and each test image.  Saves the labels and distances
	// in knn_data_t struct.
	if(knn == KNN_INVALID) return NULL;
	if(knn->test_dataset == MNIST_DATASET_INVALID) return NULL;
	if(knn->train_img == MNIST_IMAGE_INVALID) return NULL;
	// assert(knn->test_dataset);
	// assert(knn->train_img);
	int num_imgs = mnist_image_count(knn->test_dataset);
	uint x,y;
	mnist_image_size(knn->test_dataset, &x, &y);

	const uchar * train_img_data = mnist_image_data(knn->train_img);
	mnist_image_handle test_img = mnist_image_begin(knn->test_dataset);	

	//get all the distances and labels
	for(int i=0; i<num_imgs; i++)
	{
		const uchar * test_img_data = mnist_image_data(test_img);
		double d = distance(train_img_data, test_img_data, x, y);
		int l = mnist_image_label(test_img);
		dprint("d:%f\tl:%d\ti:%d",d,l,i);
		knn->distances[i] = d;
		knn->labels[i] = l;
		if(d<knn->min_dist[l]) knn->min_dist[l] = d;
		test_img = mnist_image_next(test_img);
	}	

	return knn->distances;
}


int knn_data_best_label(knn_data_t knn, int k, distance_t distance)
{
	//gets "best" label. If there are more than
	// k labels that are less than the threshold
	// distance, it picks the label with the 
	// CLOSEST point.


	//find the kth smallest distance (0-indexed!!, so need to subtract 1 when 
	// calling this function!
	// also partially sorts the distances and labels.
	double * distances = knn_data_get_distances(knn, distance);
	if(!distances) return LABEL_INVALID;
	int num_imgs = mnist_image_count(knn->test_dataset);
	if((k<0)||(k>=num_imgs)) return LABEL_INVALID;
	// if(k>=num_imgs)
	// {
	// 	printf("%s: warning: k:%d>num_imgs:%d. Setting k=num_imgs.\n"
	// 			,__func__, k+1,num_imgs);
	// 	k=num_imgs-1;
	// }	
	double k_dist = quickselect(knn->distances,knn->labels, 0, num_imgs-1, k);
	// num of distances <= k_dist
	int n = 0;
	int lblcnt[NUM_IMG_LABELS] = {0};
	//count labels where dist < kdist
	// for(int i=0; i<num_imgs; i++){dprint("label[%d]:%d",i,knn->labels[i]);}
	for (int i=0; i<num_imgs; i++)
	{	
		double d = knn->distances[i];
		int l = knn->labels[i];
		// dprint("d:%f\tl:%d",d,l);
		//add to nearest neighbors if <k_dist
		if(d <= k_dist) {lblcnt[l]++; n++;}
	}

	int max_cnt = 0;
	int best_label = -1;

	for(int i=0; i<NUM_IMG_LABELS; i++)
	{
		if(lblcnt[i]>max_cnt)
		{
			max_cnt = lblcnt[i];
			best_label = i;
		}
		else if (lblcnt[i] == max_cnt)
		{
			if(knn->min_dist[i] < knn->min_dist[best_label])
				best_label = i;
		}
	}
	dprint("n: %d\tk:%d\tbest_label:%d",n,k,best_label);
	return best_label;
}

