#define DEBUG_OLD
// #define DEBUG
#include "knn.h"
#include "mnist.h"
#include "distance.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#define ERRMSG "Usage: ./ocr [train-name] [train-size] [test-name] [k] [distance-scheme]\n"\
    			"The following distance schemes are supported: \n" DISTANCE_H_LIB_DESC
#define PRINT_INTERVAL 1

/*
    Usage: ./ocr [train-name] [train-size] [test-name] [k] [distance-scheme]
*/

struct ocr_results
{
	double accuracy;
	char *distance;
	int k;
	int train_size;
};


void print_ocr_status(char * distance, int num_processed, 
					int num_imgs, int correct)
{
	double processed_pct = ((double) num_processed / (double) num_imgs)*100;
	double correct_pct = ((double) correct/ (double) num_processed)*100;
	printf("[%s] %d/%d (%6.2f%%) %d/%d (%6.2f%%)\n", 
		distance, num_processed, num_imgs,
		processed_pct, correct, num_processed, correct_pct);
}

double ocr(mnist_dataset_handle train_mdh,
	mnist_dataset_handle test_mdh, int k, char * distance)
{
  	time_t print_time = time(0);
	if(k<0)
	{
		puts("Invalid k value. Exiting");
		return -1;
	}
	int correct = 0;
	int num_processed = 0;
	int num_imgs = mnist_image_count(test_mdh);
	if(num_imgs <=0)
	{
		puts("No images in test dataset. Exiting.");

	}
	mnist_image_handle test_img = mnist_image_begin(test_mdh);

	printf("K = %d\n", k+1);
	for(int i=0; i<num_imgs; i++)
	{
		int expected_label = mnist_image_label(test_img);
		if(expected_label==LABEL_INVALID)
		{
			puts("Invalid image. Exiting");
			return -1;
		}
		knn_data_t knn = knn_data_create(test_img, train_mdh);
		if (knn == KNN_INVALID)
		{
			puts("Invalid image or dataset. Exiting.");
			knn_data_free(knn);
			return -1;
		}
		distance_t dist_func = create_distance_function(distance);
		if(!dist_func)
		{
			knn_data_free(knn);
			return -1;
		} 
			
		int label = knn_data_best_label(knn, k, dist_func);
		if(label==LABEL_INVALID)
		{
			puts("Knn_best_label failed. Exiting");
			knn_data_free(knn);
			return -1;
		}

		if (label==expected_label) correct++;
		num_processed++;
		if ((time(0)-PRINT_INTERVAL)>=print_time)
		{
			print_time = time(0);
			print_ocr_status(distance, num_processed, num_imgs, correct);
		}
		test_img = mnist_image_next(test_img);
		knn_data_free(knn);
	}
	print_ocr_status(distance, num_processed, num_imgs, correct);
	double accuracy = (double) correct / (double) num_processed;
	//prints periodically
	//returns accuracy

	return accuracy;
}


int main (int argc, char ** args)
{
	//parse args
	//check for errors
	bool print_results = false;
	if (argc!=6)
	{
		puts(ERRMSG);
		exit(EXIT_FAILURE);
	}

	char * train_name  = args[1]; //"data/train";
	//open train_name
	// check if valid
	mnist_dataset_handle train_mdh = mnist_open(train_name);
	if(train_mdh == MNIST_DATASET_INVALID)
	{
		printf("%s%s or %s%s cannot be opened.\n", 
			train_name, IMAGES, train_name, LABELS);
		exit(EXIT_FAILURE);
	}

	// get num_imgs in train_set
	int num_imgs = mnist_image_count(train_mdh);
	//get train_size(s)
	char * train_size = args[2];//1000;
	int train_sizes[] = {.25*num_imgs,.5*num_imgs,.75*num_imgs,num_imgs};
	int n_train_sizes = 0;
	// if train_size = all, then
	if (strcmp(train_size,"all")==0)
	{
		n_train_sizes = 4;
		print_results=true;
	// else train size[] = {trainsize}		//somethong
	}
	else
	{	
	// if train_size = 0, set train_size = num_imgs(train_name dataset)
		int t = atoi(train_size);
		if(t==0) t = num_imgs;
		train_sizes[0] = t;
		n_train_sizes = 1;
	}

	//open test_name
	//check if valid
	char * test_name = args[3];   //"data/t10k";
	mnist_dataset_handle test_mdh = mnist_open(test_name);
	if(test_mdh == MNIST_DATASET_INVALID)
	{

		printf("%s%s or %s%s cannot be opened.\n", 
			test_name, IMAGES, train_name, LABELS);
		mnist_free(train_mdh);
		exit(EXIT_FAILURE);

	}
	//get k(s), check if valid
	char * k = args[4];//10;
	int ks[] = {1,5,10,15,25};
	int n_ks = 0;
	// if k = all
	if (strcmp(k,"all")==0)
	{	
		print_results=true;
		n_ks = 5;
	}
	// else k[] = {k}
	else
	{

		ks[0] = atoi(k);
		n_ks = 1;
	}

	for(int i=0;i<n_ks;i++)
	{
		for(int j=0;j<n_train_sizes;j++)
		{
			if(ks[i]>train_sizes[j])
			{
				printf("Invalid input k:%d is greater than train_size:%d.\n",
					ks[i], train_sizes[j]);
				mnist_free(train_mdh);
				mnist_free(test_mdh);
				exit(EXIT_FAILURE);
			}			
		}
	}

	//get distance functions
	char * distance = args[5];   //"reduced";
	char * distances[] = DISTANCE_H_FUNCS;
	int n_distances = 0;
	// if distance = all
	if (strcmp(distance, "all")==0)
	{
		n_distances = DISTANCE_H_NUM_FUNCS;
		print_results=true;
	}
	// 	else distances[] = {distance}
	else
	{
		distances[0] = distance;
		n_distances = 1;
	}

	//ocr_results results_set[] = malloc(sizeof(ocr_results)*num_dist*num_k*num_trainsize)
	double *results = malloc(n_distances*n_ks*n_train_sizes*sizeof(double));
	mnist_dataset_handle * sample_mdhs = malloc(n_train_sizes*sizeof(mnist_dataset_handle));
	//i=0, for each train_size in train_size[]
	//create sample sets
	for(int i=0;i<n_train_sizes;i++)
	{
		// sample = mnists_create_sample(train_set, trainsize)
		// printf("%d\n", train_sizes[i]);
		sample_mdhs[i] = mnist_create_sample(train_mdh, train_sizes[i]);
		// check for error
		if (sample_mdhs[i] == MNIST_DATASET_INVALID)
		{
			printf("Can't create valid sample using %s and train_size of %d\n", 
				test_name, train_sizes[i]);
			for(int j=0;j<i;j++) mnist_free(sample_mdhs[j]);
			mnist_free(test_mdh);
			mnist_free(train_mdh);
			free(sample_mdhs);
			free(results);
			exit(EXIT_FAILURE);		
		}
	}
	int ix = 0;
	//for each distance in distances[]
	for(int i=0;i<n_distances;i++)
	{
		// fore each k in k[]
		for(int j=0;j<n_ks;j++)
		{
			//for each sample in samples
			for(int s=0;s<n_train_sizes;s++)
			{
				double accuracy = ocr(sample_mdhs[s], test_mdh, 
								ks[j]-1, distances[i]);
				if (accuracy<0)
				{
					for(int m=0;m<n_train_sizes;m++) mnist_free(sample_mdhs[m]);
					free(results);
			   		free(sample_mdhs);
					mnist_free(test_mdh);
					mnist_free(train_mdh);
					return(EXIT_FAILURE);
				}
				else
				{
					results[ix] = accuracy;
				}
				ix++;
			}
		}	
	}
/*
# distance k 15000 30000
     euclid 1 50.34 60.33
     euclid 5 53.34 79.22
*/

	if(print_results)
	{
		int ix=0;
		printf("# distance k%c", ' ');
		for(int m=0;m<n_train_sizes;m++) printf("%d", train_sizes[m]);
		printf("%c", '\n');
		for(int i=0;i<n_distances;i++)
		{
			// fore each k in k[]
			for(int j=0;j<n_ks;j++)
			{
				printf("%s ", distances[i]);
				printf("%d ", ks[j]);
				//for each sample in samples
				for(int s=0;s<n_train_sizes;s++)
				{
					printf("%.2f", results[ix]*100);
					ix++;
				}
				printf("%c", '\n');
			}
		}
	}

    for(int m=0;m<n_train_sizes;m++) mnist_free(sample_mdhs[m]);
   	free(sample_mdhs);
	free(results);
	mnist_free(test_mdh);
	mnist_free(train_mdh);
	return(EXIT_SUCCESS);
}
