#include "knn.h"
#include "mnist.h"
#include "distance.h"
#include <CUnit/Basic.h>
#include <limits.h>
#include <errno.h>
#include <float.h>

//data used in the tests
// DO NOT CHANGE THESE. The values were chosen carefully to test
// some subtle conditions.
#define DATASET_X 	2
#define DATASET_Y 	2
//Each label group using BASE_IMG has sum(pixels)%10=0, so we can use "reduced" distance.
#define BASE_IMG 	{1,2,3,4} //sum = 10. 
#define NUM_LABELS  10
#define IMG_PER_LBL 3
#define SORTED  	{0,1,2,3,4,5,6,7,8,9}
#define UNSORTED 	{3,1,9,5,4,8,6,2,7,0}
#define SEMISORTED 	{1,2,3,4,5,6,7,8,9,0}
#define SAME_VAL 	{9,9,9,9,9,9,9,9,9,9}
#define REPEATS  	{1,2,0,2,1,0,1,2,0,3}
//test for ties using the folllowing image:
#define IMG_SUM5	{0,1,2,2} //sum = 5, exactly halfway between two images groups
#define IMG_SUM4	{0,1,1,2} // sum = 4, closer to lower label group
#define IMG_SUM6	{0,1,2,3} //sum = 6, closer to higher label group


static mnist_dataset_handle _make_test_dataset(unsigned char base_img[])
{
	mnist_dataset_handle mdh = NULL;
	mdh = mnist_create(DATASET_X,DATASET_Y);
	int img_size = DATASET_X*DATASET_Y;
	mnist_image_handle img = mnist_image_begin(mdh);
	
	//add 3 images for each label
	// label 0 has a sum = sum(baseimg)
	// label 1 has a sum = sum(baseimg)+10
	// .. label 9 has a sum = sum(baseimg)+100
	for (int i =0; i<NUM_LABELS; i++)
	{
		unsigned char img_data[DATASET_X*DATASET_Y];
		//make img_data using base_image
		for (int j=0; j<img_size; j++)
			img_data[j] = base_img[j] + (j+1)*i; //1+2+3+4 = 10

		for(int k=0;k<IMG_PER_LBL;k++)
		{
			//change first and last pixel to make each img unique.
			img = mnist_image_add_after(mdh, img, img_data, 
										DATASET_X, DATASET_Y, i);
			// for (int j=0; j<img_size; j++) printf(":%d",img_data[j]);
			// puts("\n");
			img_data[0]++;
			img_data[(DATASET_X*DATASET_Y)-1]--; 
		}
	}
	return mdh;	
}

static void test_partition()
{
	//test with unsorted data. The values are 1-10 not in order.
	for(int i=0; i<10; i++)
	{
		double ix_list[] = UNSORTED;
		int data_list[] = UNSORTED;
		//test that the value was moved to the right spot
		int pivot_val = (int) ix_list[i];
		int pivot_val_data = (int) data_list[i];

		int new_pivot_ix = partition(ix_list, data_list, 0, 9, i);
		CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val);
		//test that data_list was sorted too
		CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val_data);

		//since data_list and ix_list are the same, test it.
		// also test that the partition is valid
		for(int j=0;j<10;j++)
		{
			CU_ASSERT_EQUAL_FATAL(ix_list[j], data_list[j]);
			if (j<new_pivot_ix)
				{CU_ASSERT_TRUE_FATAL(ix_list[j]<=pivot_val);}
			else
				{CU_ASSERT_TRUE_FATAL(ix_list[j]>=pivot_val);}
		}
	}

	//test with repeats
	for(int i=0; i<10; i++)
	{
		double ix_list[] = REPEATS;
		int data_list[] = REPEATS;
		//test that the value was moved to the right spot
		double pivot_val =  ix_list[i];

		int new_pivot_ix = partition(ix_list, data_list, 0, 9, i);
		// printf("\n%d\t%d\t%d\t%f\n",i, new_pivot_ix, new_pivot_ix/3, pivot_val);

		//since there are repeats, we cannot know what the value of 
		// new_pivot_ix will be, and instead jsut test that
		// the list was partitioned propertly.
		for(int j=0;j<10;j++)
		{
			CU_ASSERT_EQUAL_FATAL(ix_list[j], data_list[j]);
			if (j<new_pivot_ix)
				{CU_ASSERT_TRUE_FATAL(ix_list[j]<=pivot_val);}
			else
				{CU_ASSERT_TRUE_FATAL(ix_list[j]>=pivot_val);}
		}		
	}

	//test with left>right;
	{
	double ix_list[] = UNSORTED;
	int data_list[] = UNSORTED;
	CU_ASSERT_EQUAL_FATAL(partition(ix_list, data_list, 9, 0, 1), -1);
	// //test with pivot_ix < left
	// CU_ASSERT_EQUAL_FATAL(partition(ix_list, data_list, 4, 9, 1), -1);
	// //test with pivot_ix>right
	// CU_ASSERT_EQUAL_FATAL(partition(ix_list, data_list, 0, 4, 5), -1);
	}
	
	//test with one element
	{
	double ix_list[] = {0};
	int data_list[] = {0};
	int pivot_val = (int) ix_list[0];
	int pivot_val_data = (int) data_list[0];
	int new_pivot_ix = partition(ix_list, data_list, 0, 0, 0);
	CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val);
	//test that data_list was sorted too
	CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val_data);
	}	

	//test with two elements
	for(int i=0; i<2; i++)
	{
	double ix_list[] = {0,1};
	int data_list[] = {0,1};
	int pivot_val = (int) ix_list[i];
	int pivot_val_data = (int) data_list[i];
	int new_pivot_ix = partition(ix_list, data_list, 0, 1, i);
	CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val);
	//test that data_list was sorted too
	CU_ASSERT_EQUAL_FATAL(new_pivot_ix, pivot_val_data);
	}	

}


static void test_quickselect()
{
	//test with unsorted data
	for(int k=0; k<10; k++)
	{
		double ix_list[] = UNSORTED;
		int data_list[] = UNSORTED;
		//test that the value was moved to the right spot
		double kth_val = quickselect(ix_list, data_list, 0, 9, k);
		// printf("%f\t%d\n", kth_val, k);
		CU_ASSERT_EQUAL_FATAL(kth_val, (double) k);
	}	

	
	//test with semisorted data
	for(int k=0; k<10; k++)
	{
		double ix_list[] = SEMISORTED;
		int data_list[] = SEMISORTED;
		//test that the value was moved to the right spot
		double kth_val = quickselect(ix_list, data_list, 0, 9, k);
		// printf("%f\t%d\n", kth_val, k);
		CU_ASSERT_EQUAL_FATAL(kth_val, (double) k);
	}	
	//test with repeats
	for(int k=0; k<10; k++)
	{
		double ix_list[] = REPEATS;
		int data_list[] = REPEATS;
		//test that the value was moved to the right spot
		double kth_val = quickselect(ix_list, data_list, 0, 9, k);
		// printf("%f\t%d\n", kth_val, k);
		CU_ASSERT_EQUAL_FATAL(kth_val, (double) (k/3));
	}

}

static void test_knn_data_create_free()
{
	//test with empty dataset
	{
	unsigned char base_img[] = BASE_IMG;
	mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
	mnist_dataset_handle test_mdh = mnist_create(DATASET_X,DATASET_Y);
	mnist_image_handle train_img = mnist_image_begin(train_mdh);
	knn_data_t knn = knn_data_create(train_img, test_mdh);
	CU_ASSERT_EQUAL_FATAL(knn, KNN_INVALID);

	knn_data_free(knn);
	mnist_free(train_mdh);
	mnist_free(test_mdh);	
	}
	//test with normal dataset
	{
	unsigned char base_img[] = BASE_IMG;
	mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
	mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
	mnist_image_handle train_img = mnist_image_begin(train_mdh);
	knn_data_t knn = knn_data_create(train_img, test_mdh);
	CU_ASSERT_NOT_EQUAL_FATAL(knn, KNN_INVALID);

	knn_data_free(knn);
	mnist_free(train_mdh);
	mnist_free(test_mdh);	
	}
	//test with invalid image
	{
	unsigned char base_img[] = BASE_IMG;
	mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
	mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
	mnist_image_handle train_img = MNIST_IMAGE_INVALID;
	knn_data_t knn = knn_data_create(train_img, test_mdh);
	CU_ASSERT_EQUAL_FATAL(knn, KNN_INVALID);

	knn_data_free(knn);
	mnist_free(train_mdh);
	mnist_free(test_mdh);	
	}	

}


static void test_knn_data_get_distances()
{
	//test normal dataset
	//test with normal dataset against itself
	{
	unsigned char base_img[] = BASE_IMG;
	mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
	mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
	mnist_image_handle train_img = mnist_image_begin(train_mdh);
	knn_data_t knn = knn_data_create(train_img, test_mdh);
	distance_t distance = create_distance_function("reduced");
	double * distances = knn_data_get_distances(knn, distance);
	//test distances and to make sure they are correct.
	for(int i = 0; i < NUM_LABELS * IMG_PER_LBL; i++)
	{
		// since label=0 has sum =10
		//  label 1 has sum =20
		//  all the images are IN ORDER
		//  thus we can calc expected distance with ease
		double expected_dist = (double) ((int)(i/IMG_PER_LBL)*10);
		// printf("%f\t%f\n", distances[i], expected_dist);
		CU_ASSERT_EQUAL_FATAL(distances[i], expected_dist);
	}

	knn_data_free(knn);
	mnist_free(train_mdh);
	mnist_free(test_mdh);	
	}

	//test bad image
	{
		unsigned char base_img[] = BASE_IMG;
		// mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = MNIST_IMAGE_INVALID;
		knn_data_t knn = knn_data_create(train_img, test_mdh);
		distance_t distance = create_distance_function("reduced");
		double * distances = knn_data_get_distances(knn, distance);
		CU_ASSERT_EQUAL_FATAL(distances, NULL);

		knn_data_free(knn);
		mnist_free(test_mdh);
	}

	//test empty dataset
	{
		unsigned char base_img[] = BASE_IMG;
		mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
		mnist_dataset_handle test_mdh = mnist_create(DATASET_X,DATASET_Y);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		knn_data_t knn = knn_data_create(train_img, test_mdh);
		distance_t distance = create_distance_function("reduced");
		double * distances = knn_data_get_distances(knn, distance);
		CU_ASSERT_EQUAL_FATAL(distances, NULL);

		knn_data_free(knn);
		mnist_free(train_mdh);
		mnist_free(test_mdh);
	}
}

static void test_knn_data_best_label()
{
	//test normal dataset
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = BASE_IMG;
		mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(offset_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");
		int num_imgs = mnist_image_count(train_mdh);
		// knn_data_get_distances(knn, distance);
		//test to see that EACH image is classified correctly.
		for(int i=1;i<num_imgs;i++)
		{
			knn_data_t knn = knn_data_create(train_img, test_mdh);
			int label = knn_data_best_label(knn, 5, distance);
			int expected_label = mnist_image_label(train_img);
			CU_ASSERT_EQUAL_FATAL(label, expected_label);
			train_img = mnist_image_next(train_img);
			knn_data_free(knn);
		}
		mnist_free(train_mdh);
		mnist_free(test_mdh);	
	}

	//test normal dataset with train image that 
	// is  between two groups, but closer to the CORRECT group.
	// ALL LABELS SHOULD BE ACCURATE
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = IMG_SUM6;
		mnist_dataset_handle train_mdh =_make_test_dataset(offset_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");
		int num_imgs = mnist_image_count(train_mdh);
		// knn_data_get_distances(knn, distance);
		//test to see that EACH image is classified correctly.
		for(int i=1;i<num_imgs;i++)
		{
			knn_data_t knn = knn_data_create(train_img, test_mdh);
			int label = knn_data_best_label(knn, 0, distance);
			int expected_label = mnist_image_label(train_img);
			CU_ASSERT_EQUAL_FATAL(label, expected_label);
			train_img = mnist_image_next(train_img);
			knn_data_free(knn);
		}
		mnist_free(train_mdh);
		mnist_free(test_mdh);	
	}

	//test normal dataset with train image that 
	// is  between two groups, but closer to the INCORRECT group.
	// ALL LABELS SHOULD BE INACCURATE except for the first 3 labels
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = IMG_SUM4;
		mnist_dataset_handle train_mdh =_make_test_dataset(offset_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");
		int num_imgs = mnist_image_count(train_mdh);
		// knn_data_get_distances(knn, distance);
		//test to see that EACH image is classified correctly.
		for(int i=1;i<num_imgs;i++)
		{
			knn_data_t knn = knn_data_create(train_img, test_mdh);
			int label = knn_data_best_label(knn, 0, distance);
			int expected_label = mnist_image_label(train_img);
			if(i<=3)
			{
				CU_ASSERT_EQUAL_FATAL(label, expected_label);
			}
			else
			{
				CU_ASSERT_EQUAL_FATAL(label, expected_label-1);
			}
			train_img = mnist_image_next(train_img);
			knn_data_free(knn);
		}
		mnist_free(train_mdh);
		mnist_free(test_mdh);	
	}

	//test normal dataset with train image that 
	// is  exactly between two groups, but closer to the INCORRECT group.
	// ALL LABELS SHOULD BE INACCURATE except the first 
	// 3, since the algo gets the FIRST best label
	//  and all the labels are in order.
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = IMG_SUM5;
		mnist_dataset_handle train_mdh =_make_test_dataset(offset_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");
		int num_imgs = mnist_image_count(train_mdh);
		// knn_data_get_distances(knn, distance);
		//test to see that EACH image is classified correctly.
		for(int i=1;i<num_imgs;i++)
		{
			knn_data_t knn = knn_data_create(train_img, test_mdh);
			int label = knn_data_best_label(knn, 0, distance);
			int expected_label = mnist_image_label(train_img);
			// printf("label:%d\texpected:%d\n", label, expected_label);
			if(i<=3)
			{
				CU_ASSERT_EQUAL_FATAL(label, expected_label);
			}
			else
			{
				CU_ASSERT_EQUAL_FATAL(label, expected_label-1);
			}
			train_img = mnist_image_next(train_img);
			knn_data_free(knn);
		}
		mnist_free(train_mdh);
		mnist_free(test_mdh);	
	}

	//test bad image
	{
		unsigned char base_img[] = BASE_IMG;
		// mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = MNIST_IMAGE_INVALID;
		knn_data_t knn = knn_data_create(train_img, test_mdh);
		distance_t distance = create_distance_function("reduced");

		int label = knn_data_best_label(knn, 0, distance);
		CU_ASSERT_EQUAL_FATAL(label, LABEL_INVALID);
		knn_data_free(knn);
		mnist_free(test_mdh);
		
	}
	//test k>num_imgs
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = IMG_SUM5;
		mnist_dataset_handle train_mdh =_make_test_dataset(offset_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");
		int num_imgs = mnist_image_count(train_mdh);
		knn_data_t knn = knn_data_create(train_img, test_mdh);
		int label = knn_data_best_label(knn, num_imgs, distance);
		CU_ASSERT_EQUAL_FATAL(label, LABEL_INVALID);
		knn_data_free(knn);
		mnist_free(train_mdh);
		mnist_free(test_mdh);
	}
	
	//test k = -1
	{
		unsigned char base_img[] = BASE_IMG;
		unsigned char offset_img[] = IMG_SUM5;
		mnist_dataset_handle train_mdh =_make_test_dataset(offset_img);
		mnist_dataset_handle test_mdh = _make_test_dataset(base_img);
		mnist_image_handle train_img = mnist_image_begin(train_mdh);
		distance_t distance = create_distance_function("reduced");

		knn_data_t knn = knn_data_create(train_img, test_mdh);
		int label = knn_data_best_label(knn, -1, distance);
		CU_ASSERT_EQUAL_FATAL(label, LABEL_INVALID);
		knn_data_free(knn);
		mnist_free(train_mdh);
		mnist_free(test_mdh);
	}
	//test with empty dataset
	{
	unsigned char base_img[] = BASE_IMG;
	mnist_dataset_handle train_mdh =_make_test_dataset(base_img);
	mnist_dataset_handle test_mdh = mnist_create(DATASET_X,DATASET_Y);
	mnist_image_handle train_img = mnist_image_begin(train_mdh);
	knn_data_t knn = knn_data_create(train_img, test_mdh);
	distance_t distance = create_distance_function("reduced");

	int label = knn_data_best_label(knn, 3, distance);
	// int expected_label = mnist_image_label(train_img);
	CU_ASSERT_EQUAL_FATAL(label, LABEL_INVALID);

	knn_data_free(knn);
	mnist_free(train_mdh);
	mnist_free(test_mdh);	
	}

}

static int init_suite(void)
{
	return 0;
}

static int clean_suite(void)
{
	return 0;
}
int main()
{
	CU_pSuite pSuite = NULL;
	   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Unit Test Suite", init_suite, clean_suite);
   if (NULL == pSuite)
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   /* NOTE - ORDER IS IMPORTANT - MUST TEST fread() AFTER fprintf() */
   if ((   NULL == CU_add_test(pSuite, "partition()\n", test_partition))
       || (NULL == CU_add_test(pSuite, "quickselect()\n", test_quickselect))
       || (NULL == CU_add_test(pSuite, "knn_data_create() and _free()\n", test_knn_data_create_free))
       || (NULL == CU_add_test(pSuite, "knn_data_get_distances()\n", test_knn_data_get_distances))
       || (NULL == CU_add_test(pSuite, "knn_data_best_label()\n", test_knn_data_best_label))
      )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
}
