#include "distance.h"
#include <CUnit/Basic.h>
#include <limits.h>
#include <errno.h>
#include <float.h>
#define TEST_T10K "data/t10k"
#define TEST_TRAIN "data/train"
#define TEST_OUTFILE "data/test"
#define TEST_T10K_FIRST_LBL 7
#define TEST_T10k_FILE_SZ 10008

#define XSIZE1 3
#define YSIZE1 3
#define XSIZE2 4
#define YSIZE2 4
#define IMGDATA1_1 {0}
#define IMGDATA1_2 {1,3,3,4,4,5,6,7,8};
#define IMGDATA1_3 {2,4,4,5,5,6,7,8,9};

static int init_suite(void)
{
	return 0;
}

static int clean_suite(void)
{
	return 0;
}

static void test_describe_distance_functions()
{
	char * desc = describe_distance_functions(); 
	CU_ASSERT_NOT_EQUAL_FATAL(desc, NULL);
	puts(desc);

}

static void test_euclid()
{

	//sample images
	unsigned char img1_data[XSIZE1*YSIZE1] = IMGDATA1_1;
	unsigned char img2_data[XSIZE1*YSIZE1] = IMGDATA1_2;
	unsigned char img3_data[XSIZE1*YSIZE1] = IMGDATA1_3;

	// ans_xy = the euclidean distance betwen imgx and imgy,
	// calculated by hand/excel spreadsheet
	// i don't think i should calc answers computationally
	// in the test script because I will be checking against 
	// my own algorithm
	double ans11 = 0;
	double ans12 = 15;
	double ans21 = ans12;
	double ans23 = 3;
	double ans32 = ans23;
	double ans31 = sqrt(2*2 + 
						4*4 + 
						4*4 +
						5*5 +
						5*5 +
						6*6 +
						7*7 +
						8*8 +
						9*9);
	double ans13 = ans31;
	//get euclid function, test that it works
	distance_t euclid = create_distance_function("euclid");
	CU_ASSERT_NOT_EQUAL_FATAL(euclid, NULL);
	//test for errors
	//invalid image
	double d1i = euclid(img1_data, NULL, XSIZE1, YSIZE1); // should be -1
	CU_ASSERT_EQUAL_FATAL(d1i, DBL_MAX);
	CU_ASSERT_EQUAL_FATAL(errno, EINVAL);
	errno = 0;
	//invalid dimensions
	double d0i = euclid(img1_data, img2_data, 0, 0); // should be -1
	CU_ASSERT_EQUAL_FATAL(d0i, 0);
	CU_ASSERT_EQUAL_FATAL(errno, EINVAL);
	errno = 0;
	//test for accuracy
	double d12 = euclid(img1_data, img2_data, XSIZE1, YSIZE1); //should be 15
	CU_ASSERT_EQUAL(d12, ans12);
	double d21 = euclid(img2_data, img1_data, XSIZE1, YSIZE1); //should be 15
	CU_ASSERT_EQUAL(d21, ans21);
	double d23 = euclid(img2_data, img3_data, XSIZE1, YSIZE1); //should be 3
	CU_ASSERT_EQUAL(d23, ans23);
	double d32 = euclid(img3_data, img2_data, XSIZE1, YSIZE1); //shoule be 3
	CU_ASSERT_EQUAL(d32, ans32);
	double d11 = euclid(img1_data, img1_data, XSIZE1, YSIZE1); //should be 0
	CU_ASSERT_EQUAL(d11, ans11);
	double d13 = euclid(img1_data, img3_data, XSIZE1, YSIZE1); //should be ~17.77
	CU_ASSERT_EQUAL(d13, ans13);
	double d31 = euclid(img3_data, img1_data, XSIZE1, YSIZE1); //should be ~17.77
	CU_ASSERT_EQUAL(d31, ans31);

}


static void test_reduced()
{
	//sample images
	unsigned char img1_data[XSIZE1*YSIZE1] = IMGDATA1_1;
	unsigned char img2_data[XSIZE1*YSIZE1] = IMGDATA1_2;
	unsigned char img3_data[XSIZE1*YSIZE1] = IMGDATA1_3;

	//ansxy = the euclidean distance betwen imgx and imgy,
	// calculatted by hand
	double ans11 = 0;
	double ans12 = 41;
	double ans21 = ans12;
	double ans23 = 9;
	double ans32 = ans23;
	double ans31 = 50;
	double ans13 = ans31;

	//get euclid function, test that it works
	distance_t reduced = create_distance_function("reduced");
	CU_ASSERT_NOT_EQUAL_FATAL(reduced, NULL);

	//test for error
	double d1i = reduced(img1_data, NULL, XSIZE1, YSIZE1); // should be -1
	CU_ASSERT_EQUAL_FATAL(d1i, DBL_MAX);
	//test for accuracy
	double d12 = reduced(img1_data, img2_data, XSIZE1, YSIZE1); //should be 41
	CU_ASSERT_EQUAL(d12, ans12);
	double d21 = reduced(img2_data, img1_data, XSIZE1 ,YSIZE1); //should be 41
	CU_ASSERT_EQUAL(d21, ans21);
	double d23 = reduced(img2_data, img3_data, XSIZE1, YSIZE1); //should be 9
	CU_ASSERT_EQUAL(d23, ans23);
	double d32 = reduced(img3_data, img2_data, XSIZE1, YSIZE1); //shoule be 9
	CU_ASSERT_EQUAL(d32, ans32);
	double d11 = reduced(img1_data, img1_data, XSIZE1, YSIZE1); //should be 0
	CU_ASSERT_EQUAL(d11, ans11);
	double d13 = reduced(img1_data, img3_data, XSIZE1, YSIZE1); //should be 50
	CU_ASSERT_EQUAL(d13, ans13);
	double d31 = reduced(img3_data, img1_data, XSIZE1, YSIZE1); //should be 50
	CU_ASSERT_EQUAL(d31, ans31);
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
   if ((   NULL == CU_add_test(pSuite, "describe_distance_functions()\n", test_describe_distance_functions))
       || (NULL == CU_add_test(pSuite, "create_distance_function(\"euclid\")\n", test_euclid))
       || (NULL == CU_add_test(pSuite, "create_distance_function(\"reduced\")\n", test_reduced))
       // || (NULL == CU_add_test(pSuite, "create_distance_function(\"downsample\")\n", test_downsample))
       // || (NULL == CU_add_test(pSuite, "create_distance_function(\"crop\")\n", test_crop))
       // || (NULL == CU_add_test(pSuite, "create_distance_function(\"threshold\")\n", test_threshold))
       // || (NULL == CU_add_test(pSuite, "create_distance_function(\"manhattan\")\n", test_manhattan))
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
