#include "mnist.h"
#include <CUnit/Basic.h>
#include <limits.h>
#define TEST_T10K "data/t10k"
#define TEST_TRAIN "data/train"
#define TEST_OUTFILE "data/test"
#define TEST_T10K_FIRST_LBL 7
#define TEST_T10k_FILE_SZ 10008

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */

// static mnist_dataset_handle my_mdh;
// static mnist_image_handle my_mih;

static int init_suite(void)
{

	// my_mdh = mnist_open(TEST_T10K);
	// my_mih = mnist_image_begin(my_mdh);
	return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
static int clean_suite(void)
{
	// mnist_free(my_mdh);
  	return 0;
}

static void test_mnist_open()
{
	//valid names are t10k, train

	//test t10k
	mnist_dataset_handle mdh = NULL;
	CU_ASSERT_NOT_EQUAL((mdh=mnist_open(TEST_T10K)), MNIST_DATASET_INVALID);
	mnist_free(mdh);
	//test train
	CU_ASSERT_NOT_EQUAL((mdh=mnist_open(TEST_TRAIN)), MNIST_DATASET_INVALID);
	mnist_free(mdh);
	//test invalid
	CU_ASSERT_EQUAL((mdh=mnist_open("invalid")), MNIST_DATASET_INVALID);
	mnist_free(mdh);
}

static void test_mnist_create()
{
	//test empty valid create
	mnist_dataset_handle mdh = NULL;
	CU_ASSERT_NOT_EQUAL((mdh=mnist_create(32,32)),MNIST_DATASET_INVALID);
	mnist_free(mdh);
	//test invalid create (x=0 || y=0)
	CU_ASSERT_EQUAL((mdh=mnist_create(0,32)),MNIST_DATASET_INVALID);
	//mnist_free(mdh);
}

static void test_mnist_image_count()
{
	mnist_dataset_handle mdh = NULL;
	//test with empty dataset
	mdh=mnist_create(32,32);
	CU_ASSERT_EQUAL(mnist_image_count(mdh), 0);
	mnist_free(mdh);

	//test with t10k dataset
	mdh=mnist_open(TEST_T10K);
	CU_ASSERT_EQUAL(mnist_image_count(mdh), 10000);
	mnist_free(mdh);
	//test with invalid dataset
	mdh=MNIST_DATASET_INVALID;
	CU_ASSERT_TRUE(mnist_image_count(mdh)<0);
	mnist_free(mdh);

}

static void test_mnist_image_size()
{
	mnist_dataset_handle mdh = NULL;	
	unsigned int x = 1, y=1;
	//test with empty dataset
	mdh=mnist_create(32,32);
	mnist_image_size(mdh, &x, &y);
	CU_ASSERT_TRUE((x==32)&&(y==32));
	mnist_free(mdh);
	//test with t10k dataset
	mdh=mnist_open(TEST_T10K);
	x=0,y=0;
	mnist_image_size(mdh, &x, &y);
	CU_ASSERT_TRUE((x==28)&&(y==28));
	mnist_free(mdh);
	//test with invalid dataset
	mdh=MNIST_DATASET_INVALID;
	x=1,y=1;
	mnist_image_size(mdh, &x, &y);
	CU_ASSERT_TRUE(!x&&!y);
	mnist_free(mdh);	
}


static void test_mnist_image_begin()
{
	//test with invalid dataset
	mnist_dataset_handle mdh = NULL;
	mnist_image_handle mih = (mnist_image_handle) 0xDEADBEEF;
	mdh = MNIST_DATASET_INVALID;
	CU_ASSERT_EQUAL((mih=mnist_image_begin(mdh)),MNIST_IMAGE_INVALID);
	mnist_free(mdh);
	//test with t10k dataset
	mdh=mnist_open(TEST_T10K);
	mih=MNIST_IMAGE_INVALID;
	CU_ASSERT_NOT_EQUAL((mih=mnist_image_begin(mdh)),MNIST_IMAGE_INVALID);	
	mnist_free(mdh);
}

static void test_mnist_image_data()
{
	//test with t10k dataset
	mnist_dataset_handle mdh = mnist_open(TEST_T10K);
	mnist_image_handle mih = mnist_image_begin(mdh);
	const unsigned char * t = mnist_image_data(mih);
	CU_ASSERT_NOT_EQUAL(t, NULL);
	//test with invalid image handle
	mih = MNIST_IMAGE_INVALID;
	CU_ASSERT_EQUAL(mnist_image_data(mih), NULL);
	mnist_free(mdh);
}

static void test_mnist_image_label()
{
	mnist_dataset_handle mdh = mnist_open(TEST_T10K);
	mnist_image_handle mih = mnist_image_begin(mdh);
	//test with t10k
	CU_ASSERT_EQUAL(mnist_image_label(mih), TEST_T10K_FIRST_LBL);
	//test with invalid
	CU_ASSERT_EQUAL(mnist_image_label(MNIST_IMAGE_INVALID),-1);
	mnist_free(mdh);
}

static void test_mnist_image_next()
{
	//test with my_mih
	mnist_dataset_handle mdh = mnist_open(TEST_T10K);
	mnist_image_handle mih = mnist_image_begin(mdh);
	CU_ASSERT_NOT_EQUAL(mnist_image_next(mih), MNIST_IMAGE_INVALID);
	mnist_free(mdh);

}

static void test_mnist_image_add_after()
{
	//test with mdh, mih
	mnist_dataset_handle mdh = mnist_open(TEST_T10K);
	mnist_image_handle mih = mnist_image_begin(mdh);
	unsigned int x=28, y=28;
	unsigned char imagedata[28*28] = {0};
	unsigned int label = 9;
	CU_ASSERT_NOT_EQUAL(mnist_image_add_after(mdh, mih, imagedata, 
									x, y, label), MNIST_IMAGE_INVALID);
	//test with my_mdh, my_mih = MNIST_IMAGE_INVALID
	mih = mnist_image_add_after(mdh, MNIST_IMAGE_INVALID, 
							imagedata, x, y, label);
	CU_ASSERT_EQUAL(mih, mnist_image_begin(mdh));

	//test with my_mdh, my_mih, invalid x size
	x--;
	CU_ASSERT_EQUAL(mnist_image_add_after(mdh, mih, imagedata, 
									x, y, label), MNIST_IMAGE_INVALID);
	x++;
	//test with empty dataset, MNIST_IMAGE_INVALID
	mnist_free(mdh);
	mdh = mnist_create(28,28);
	mih = mnist_image_add_after(mdh, MNIST_IMAGE_INVALID, imagedata, x, y, label);
	CU_ASSERT_EQUAL(mih, mnist_image_begin(mdh));
	mnist_free(mdh);

}

static void test_mnist_save()
{
	//test with my_mdh

	//test with empty_dataset
	mnist_dataset_handle mdh = mnist_create(28,28);
	CU_ASSERT_TRUE(mnist_save(mdh, TEST_OUTFILE));
	mnist_free(mdh);
	mdh = mnist_open(TEST_OUTFILE);
	int count = mnist_image_count(mdh);
	CU_ASSERT_EQUAL(count, 0);
	mnist_free(mdh);
	//test with invalid
	CU_ASSERT_FALSE(mnist_save(MNIST_DATASET_INVALID, TEST_OUTFILE));
}

static void test_mnist_create_sample()
{
	mnist_dataset_handle mdh = mnist_open(TEST_T10K);
	mnist_dataset_handle sample1 = NULL, sample2 = NULL, invalid_sample=NULL;
	unsigned int n = 100;
	unsigned int too_many = UINT_MAX;

	
	//test that TOO large is invalid
	CU_ASSERT_EQUAL_FATAL((invalid_sample=mnist_create_sample(mdh,too_many)), 
					MNIST_DATASET_INVALID);
	mnist_free(invalid_sample);

	//test that negative is invalid (since -1 evaluated to MAX_UNSIGNED, it will
	// be too large)
	CU_ASSERT_EQUAL_FATAL((invalid_sample=mnist_create_sample(mdh,-1)), 
					MNIST_DATASET_INVALID);
	mnist_free(invalid_sample);

	//test that 0 works but is empty
	invalid_sample=mnist_create_sample(mdh,0);
	CU_ASSERT_EQUAL_FATAL(mnist_image_count(invalid_sample), 0);
	mnist_free(invalid_sample);


	//compare sample1 and sample2
	sample1 = mnist_create_sample(mdh, n);
	sample2 = mnist_create_sample(mdh, n);

	//test if samples are valid datasets
	CU_ASSERT_NOT_EQUAL_FATAL(sample1, NULL);
	CU_ASSERT_NOT_EQUAL_FATAL(sample1, MNIST_DATASET_INVALID);
	CU_ASSERT_NOT_EQUAL_FATAL(sample2, NULL);
	CU_ASSERT_NOT_EQUAL_FATAL(sample2, MNIST_DATASET_INVALID);

	mnist_image_handle s1_img=mnist_image_begin(sample1);
	mnist_image_handle s2_img=mnist_image_begin(sample2);

	//num images are the same
	unsigned int diff_img_count = 0;
	CU_ASSERT_EQUAL_FATAL(mnist_image_count(sample1), mnist_image_count(sample2));
	//num of images that are the different > 0
	unsigned int x, y, num_pixels;
	mnist_image_size(sample1, &x, &y);
	num_pixels = x*y;

	//go through each image, count number of diff images
	for(unsigned int i=0; i<mnist_image_count(sample1); ++i) //for each image
	{
		const unsigned char *s1_img_data,  *s2_img_data;
		s1_img_data = mnist_image_data(s1_img);
		s2_img_data = mnist_image_data(s2_img);
		bool same_val = true;
		for(unsigned int p=0; p<num_pixels; ++p)//each pixel in 
		{
			//compare values
			same_val &= (s1_img_data[p] == s2_img_data[p]);
			//break if false, no need to check all pixels
			if(!same_val) break;
		}

		if (!same_val) diff_img_count++;
		s1_img = mnist_image_next(s1_img);
		s2_img = mnist_image_next(s2_img);
	}


	// num different should be >95 with greater than 99% probability
	CU_ASSERT_TRUE((diff_img_count>95));

	mnist_free(mdh);
	mnist_free(sample1);
	mnist_free(sample2);
}


/* The main() function for setting up and running the tests.
 * Returns a CUE_SUCCESS on successful running, another
 * CUnit error code on failure.
 */
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
   if (
       (NULL == CU_add_test(pSuite, "mnist_open()\n", test_mnist_open))
       || (NULL == CU_add_test(pSuite, "mnist_create()\n", test_mnist_create))
	   || (NULL == CU_add_test(pSuite, "mnist_image_count()\n", test_mnist_image_count))
	   || (NULL == CU_add_test(pSuite, "mnist_image_size()\n", test_mnist_image_size))
	   || (NULL == CU_add_test(pSuite, "mnist_image_begin()\n", test_mnist_image_begin))
	   || (NULL == CU_add_test(pSuite, "mnist_image_data()\n", test_mnist_image_data))
	   || (NULL == CU_add_test(pSuite, "mnist_image_label()\n", test_mnist_image_label))
	   || (NULL == CU_add_test(pSuite, "mnist_image_next()\n", test_mnist_image_next))
	   || (NULL == CU_add_test(pSuite, "mnist_image_add_after()\n", test_mnist_image_add_after))
	   || (NULL == CU_add_test(pSuite, "mnist_save()\n", test_mnist_save))
	   || (NULL == CU_add_test(pSuite, "mnist_create_sample()\n", test_mnist_create_sample))
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
