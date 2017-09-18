#include "mnist.h"
#include <arpa/inet.h> // for ntoh and hton functions
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

// #define DEBUG //use "gcc -D DEBUG" to run with debug
// #define DEBUG_OLD
#ifdef DEBUG
  #define debug_print(fmt, ...) printf("debug: " fmt,  __VA_ARGS__)
#else
  #define debug_print(fmt, ...) do {} while(0)
#endif


struct mnist_dataset_t
{
	// raw data from label file
	uint8_t * lblbuf;

	// raw data from image file
	uint8_t * imgbuf;

	//pointer to first image
	struct mnist_image_t * head;

};

struct mnist_image_t
{
	//handle of dataset image belongs to
	mnist_dataset_handle mdh;

	//offset of image data in mnist_dataset_handle->imgbuf 
	//		and mnist_dataset_handle->lblbuf. 
	// ranges from 0 to num_images-1
	uint32_t idx;

	//pointer to next image
	struct mnist_image_t * next;
};

//returns an pseudo-random integer that is uniformly distributed
// in N bins (i.e the range [0,N))
int _uniform_rand_int(int range){
	int upper_rand = (RAND_MAX / range)*range;
	//discard anything above upper_rand
	int i;
	while( (i=rand()) >= upper_rand){/*do nothing*/}
	// debug_print("i%%range:%drange:%d\n", i%range, range);
	return i%range;
}

//shuffles an int array using Fisher-Yates algo in O(N) time
void _fisher_yates_shuffle(int *array, int length)
{
	int r, t;
	for(int i=length-1; i>=0; i--)
	{
		r = _uniform_rand_int(i+1);
		t = array[i];
		array[i] = array[r];
		array[r] = t;
	}
}

int _compare_int(const void* a, const void* b)
{
	return *(int*)a - *(int*)b;
}

void _populate_mnist_dataset(mnist_dataset_handle mdh, uint8_t * lblbuf, uint8_t * imgbuf)
{
	//internal helper function used by mnist_open
	//populate the fields of mdh as well as make the linked list

	assert(mdh);
	assert(imgbuf);
	assert(lblbuf);
	mdh->imgbuf = imgbuf;
	mdh->lblbuf = lblbuf;

	int img_cnt = mnist_image_count(mdh);
	unsigned int x=0, y = 0;
	mnist_image_size(mdh, &x, &y);

	if((img_cnt<=0)||(x<=0)||(y<=0))
	{
		mdh->head = MNIST_IMAGE_INVALID;
		return;
	}

	//set head
	mnist_image_handle mih = (mnist_image_handle) malloc(sizeof(struct mnist_image_t));
	mih->idx = 0;
	mih->next = MNIST_IMAGE_INVALID;
	mih->mdh = mdh;
	mdh->head = mih;
	
	//iterate through rest of images
	mnist_image_handle prev_mih = NULL;
	int i = 1;
	while(i<img_cnt)
	{
		prev_mih = mih;
		mih = (mnist_image_handle) malloc(sizeof(struct mnist_image_t));
		prev_mih->next = mih;
		//populate data structure
		mih->idx = i;
		mih->next = MNIST_IMAGE_INVALID;
		mih->mdh = mdh;
		i++;
	} 
}

int _read_file(uint8_t ** buf, char * path)
{	
	// adapted from stackoverflow.com
	FILE * fp = fopen(path, "rb");  	// Open the file in binary mode
	if (fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);          // Jump to the end of the file
	size_t flen = ftell(fp);             // Get the current byte offset in the file
	rewind(fp);                      // Jump back to the beginning of the file
	debug_print("_read_file: path=%s\tflen=%lu\n", path, flen);
	*buf = (uint8_t *)malloc(flen*sizeof(uint8_t));
	if (*buf==NULL)
		return 0;
	// Read in the entire file, return 1 if success, 0 otherwise
	int _read = fread(*buf, flen, 1, fp); 
	fclose(fp); // Close the file
	return _read;
}


mnist_dataset_handle mnist_open(const char * name)
{
	char * imgpath = (char *) malloc(strlen(name)+strlen(IMAGES)+1);
	strcpy(imgpath, name);
	strcat(imgpath,IMAGES);

	char * lblpath = (char *) malloc(strlen(name)+strlen(LABELS)+1);
	strcpy(lblpath, name);
	strcat(lblpath, LABELS);

	uint8_t * lblbuf = NULL, * imgbuf = NULL;
	uint32_t imn=0, lmn=0;	//magic numbers

	// indicator if file readreturns 0 if error or no data, 1 otherwise.

	int ibr = _read_file(&imgbuf, imgpath);
	int lbr = _read_file(&lblbuf, lblpath);
	debug_print("mnist_open: ibr=%d\tlbr=%d\n", ibr, lbr);
	debug_print("mnist_open: imgbuf=%p\tlblbuf=%p\n",
				(void *) imgbuf, (void *) lblbuf);
	if (!ibr|| !lbr)
		{
			free(imgpath);
			free(lblpath);
			return MNIST_DATASET_INVALID;
		}

	//read magic number from ifp
	//cast buf to 32 bit
	lmn = (uint32_t) MY_NTOHL(((uint32_t * ) lblbuf)[MN_IX]);
	imn = (uint32_t) MY_NTOHL(((uint32_t * ) imgbuf)[MN_IX]);

	if(imn!=IMG_MAGIC_NUM && lmn!=LBL_MAGIC_NUM)
		return MNIST_DATASET_INVALID;
	debug_print("mnist_open: imn=%" PRIu32 "\tlmn=%" PRIu32 "\n", imn, lmn);

	//make handle to dataset
	mnist_dataset_handle mdh = (mnist_dataset_handle)
								malloc(sizeof(struct mnist_dataset_t)); 

	_populate_mnist_dataset(mdh, lblbuf, imgbuf);

	free(imgpath);
	free(lblpath);
	return mdh;
}

void mnist_free(mnist_dataset_handle handle)
{
	debug_print("mnist_free: handle=%p\n", (void*) handle);
	if((handle!=MNIST_DATASET_INVALID) && (handle!=NULL))
	{
		debug_print("mnist_free: handle=%p\timgbuf=%p\tlblbuf=%p\n",
			(void*) handle, (void*)handle->imgbuf, (void*)handle->lblbuf);
		
		//free image handles
		mnist_image_handle mih = handle->head;
		mnist_image_handle prev_mih = MNIST_IMAGE_INVALID;

		while(mih!=MNIST_IMAGE_INVALID)
		{
			prev_mih = mih;
			mih = prev_mih->next;
			// debug_print("mnist_free: prev_mih->idx:%d\n",prev_mih->idx);
			free(prev_mih);
		}

		free(handle->imgbuf);
		free(handle->lblbuf);
		free(handle);
	}
}

mnist_dataset_handle mnist_create(unsigned int x, unsigned int y)
{
	//test if x==0 or y==0
	if (!x || !y)
		return MNIST_DATASET_INVALID;

	//malloc for dataset handle
	mnist_dataset_handle mdh = (mnist_dataset_handle)
								malloc(sizeof(struct mnist_dataset_t)); 

	//allocate memory for img and lbl
	uint32_t * lblbuf32 = NULL, * imgbuf32 = NULL;
	lblbuf32 = (uint32_t *)malloc(LBL_HEADER_SIZE*sizeof(uint8_t));
	imgbuf32 = (uint32_t *)malloc(IMG_HEADER_SIZE*sizeof(uint8_t));

	//malloc checks
	if (!mdh || !lblbuf32 || !imgbuf32)
	{
		if(mdh) free(mdh);
		if(lblbuf32) free(lblbuf32);
		if(imgbuf32) free(imgbuf32);
		return MNIST_DATASET_INVALID;
	}

	//assign vals as per mnist data specs in network byte order
	lblbuf32[MN_IX] = MY_HTONL(LBL_MAGIC_NUM);
	lblbuf32[NUM_IMG_IX] = MY_HTONL(0);

	imgbuf32[MN_IX] = MY_HTONL(IMG_MAGIC_NUM);
	imgbuf32[NUM_IMG_IX] = MY_HTONL(0);
	imgbuf32[X_IX] = MY_HTONL(x);
	imgbuf32[Y_IX] = MY_HTONL(y);

	//add to dataset
	_populate_mnist_dataset(mdh, (uint8_t*)lblbuf32, (uint8_t*)imgbuf32);
	debug_print("mnist_create: mdh->imgbuf=%p\tmdh->lblbuf=%p\n",
	 			(void*)mdh->imgbuf, (void*)mdh->lblbuf);
	return mdh;
}

int mnist_image_count (const mnist_dataset_handle handle)
{
	if(handle==MNIST_DATASET_INVALID || !handle)
		return -1;
	uint32_t count = MY_NTOHL(((uint32_t*)(handle->lblbuf))[NUM_IMG_IX]);
	debug_print("mnist_image_count: count:%"PRIu32"\n",count);
	return count;
}

void mnist_image_size (const mnist_dataset_handle handle,
                       unsigned int * x, unsigned int * y)
{
	if(handle==MNIST_DATASET_INVALID || !handle)
		*x=0, *y=0;
	else
	{
		*x = (unsigned int) MY_NTOHL(((uint32_t*)(handle->imgbuf))[X_IX]);
		*y = (unsigned int) MY_NTOHL(((uint32_t*)(handle->imgbuf))[Y_IX]);
	}
	debug_print("mnist_image_size: *x=%u\t*y=%u\n", *x, *y);
}

mnist_image_handle mnist_image_begin (const mnist_dataset_handle handle)
{
	if(handle==MNIST_DATASET_INVALID)
		return MNIST_IMAGE_INVALID;
	if(mnist_image_count(handle)<=0)
		return MNIST_IMAGE_INVALID;
	debug_print("mnist_image_begin: handle:%p\thandle->head->idx:%d\n", 
				(void*) handle->head,handle->head->idx);
	return handle->head;
}

const unsigned char * mnist_image_data (const mnist_image_handle h)
{
	if (h==MNIST_IMAGE_INVALID)
		return NULL;
	//get pointer to start of imgbuf
	debug_print("mnist_image_data: h:%p\th->mdh:%p\n", (void*)h, (void*)h->mdh);
	uint8_t * imgbuf = h->mdh->imgbuf;
	assert(imgbuf);

	//get idx value
	uint32_t idx = h->idx;
	//get size of images
	unsigned int x=-1, y=-1, size=0;
	mnist_image_size(h->mdh, &x, &y);
	size = x*y; //size in bytes
	const unsigned char * img_data = ((const unsigned char*)imgbuf)+IMG_HEADER_SIZE+idx*size; 
	debug_print("mnist_image_data: IMG_HEADER_SIZE+idx*size:%d\n",IMG_HEADER_SIZE+idx*size);
	debug_print("mnist_image_data: img_data:%p\n",(void*)img_data);
	debug_print("mnist_image_data: *(img_data):%d\n",*img_data);
	//add (h->idx * imagesize) bytes to imgbuf+headersize to get start of img
	return img_data;
}

int mnist_image_label (const mnist_image_handle h)
{
	if (h==MNIST_IMAGE_INVALID)
		return -1;
	debug_print("mnist_image_label: h:%p\th->mdh:%p\n", (void*)h, (void*)h->mdh);
	uint8_t * lblbuf = h->mdh->lblbuf;
	assert(lblbuf);

	//get idx val
	uint32_t idx = h->idx;
	debug_print("mnist_image_label: *(lblbuf+LBL_HEADER_SIZE+idx):%d\n", *(lblbuf+LBL_HEADER_SIZE+idx));
	return *(lblbuf+LBL_HEADER_SIZE+idx);
}

mnist_image_handle mnist_image_next (const mnist_image_handle h)
{
	if(h==MNIST_IMAGE_INVALID)
		return MNIST_IMAGE_INVALID;
	return h->next;
}

mnist_image_handle mnist_image_add_after (mnist_dataset_handle h,
      mnist_image_handle i,
      const unsigned char * imagedata, unsigned int x, unsigned int y,
      unsigned int label)
{

	//make useful variables
	unsigned int y_sz = MY_NTOHL(((uint32_t*)(h->imgbuf))[Y_IX]);
	unsigned int x_sz = MY_NTOHL(((uint32_t*)(h->imgbuf))[X_IX]);
	//check x,y against h->imgdata[2], h->imgdata[3]
	if ( (x!=x_sz) || (y!=y_sz) )
		return MNIST_IMAGE_INVALID;
	uint32_t num_images_old = MY_NTOHL(((uint32_t*)(h->imgbuf))[NUM_IMG_IX]);
	unsigned int imgbuf_old_sz = (y_sz*x_sz)*num_images_old + IMG_HEADER_SIZE;
	unsigned int lblbuf_old_sz = num_images_old+ LBL_HEADER_SIZE;
	
	debug_print("mnist_image_add_after: x:%d\ty:%d\tx_sz:%d\ty_sz:%d\tnum_images_old:%d\n",
				x,y,x_sz,y_sz,num_images_old);	
	assert(h);
	assert(imagedata);
	debug_print("mnist_image_add_after: imgbuf_old_sz:%d\tlblbuf_old_sz:%d\tnum_images_old:%d\n",
				imgbuf_old_sz, lblbuf_old_sz, num_images_old);
	uint8_t * realloc_check = NULL;
	assert(imagedata);
	
	//append imagedata to mdh->imgbuf
		//realloc mdh->imgbuf
	realloc_check = (uint8_t*)realloc(h->imgbuf, imgbuf_old_sz+(x*y));
	if(!realloc_check)
		return MNIST_IMAGE_INVALID;
	h->imgbuf = realloc_check;

		//memcpy imagedata to imgbuf
	memcpy((h->imgbuf)+imgbuf_old_sz, imagedata, (x*y));
	
	//append label to mdh->lblbuf
		//realloc mdh->lblbuf
	realloc_check = NULL;
	realloc_check = (uint8_t*)realloc(h->lblbuf, lblbuf_old_sz+1);
	if(!realloc_check)
		return MNIST_IMAGE_INVALID;
	h->lblbuf = realloc_check;
		//add label to lblbuf
	memcpy((h->lblbuf)+lblbuf_old_sz, &label, 1);
	
	//set mdh->lblbuf[1] = num_images + 1
	uint32_t num_images_new = (uint32_t) (num_images_old+1);
	((uint32_t*)(h->lblbuf))[NUM_IMG_IX] = MY_HTONL(num_images_new);
	//set mdh->imgbug[1] = num_images + 1
	((uint32_t*)(h->imgbuf))[NUM_IMG_IX] = MY_HTONL(num_images_new);
	debug_print("mnist_image_add_after: num_images_new: %"PRIu32
				"\th->lblbuf[NUM_IMG_IX]:%"PRIu32"\n", 
				num_images_new, MY_NTOHL(h->lblbuf[NUM_IMG_IX]));

	//make new_mih
	mnist_image_handle new_mih = (mnist_image_handle) malloc(sizeof(struct mnist_image_t));
		//set new_mih->idx = num_images
	new_mih->idx = num_images_old;
		//set new_mih->mdh = mdh
	new_mih->mdh = h;
		//set new_mih->next = i->nex
	if(i==MNIST_IMAGE_INVALID)
	{	
		//insert at beginning of linked list
		//set mdh->head = new_mih
		// debug_print("mnist_image_add_after: h->head: %d", (int) h->head);
		mnist_image_handle t_mih = h->head;
		h->head = new_mih;
		new_mih->next = t_mih;
	}
	else
	{
		mnist_image_handle t_mih = i->next;
		new_mih->next = t_mih;
		i->next = new_mih;
	}
	return new_mih;
}

bool mnist_save(const mnist_dataset_handle h, const char * filename)
{
	if(h==MNIST_DATASET_INVALID)
		return false;
	//get useful variables
	int num_img = mnist_image_count(h);
	unsigned int x=0,y=0;
	mnist_image_size(h, &x, &y);
	void *lblbuf =(void*)(h->lblbuf), *imgbuf = (void*) (h->imgbuf);
	size_t img_write_sz = (x*y*num_img)+IMG_HEADER_SIZE;
	size_t lbl_write_sz = num_img+LBL_HEADER_SIZE;
	size_t ibw=0, lbw=0;

	if((num_img<0)||!x||!y)
		return false;

	char * imgpath = (char *) malloc(strlen(filename)+strlen(IMAGES)+1);
	strcpy(imgpath, filename);
	strcat(imgpath,IMAGES);
	
	char * lblpath = (char *) malloc(strlen(filename)+strlen(LABELS)+1);
	strcpy(lblpath, filename);
	strcat(lblpath, LABELS);

	FILE * ifp = fopen(imgpath, "wb");  	// Open the file in binary mode
	FILE * lfp = fopen(lblpath, "wb");	

	if ((ifp == NULL) ||(lfp==NULL))
	{	
		free(imgpath);
		free(lblpath);
		return false;
	}

	//size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
	lbw = fwrite(lblbuf, lbl_write_sz, 1, lfp);
	ibw = fwrite(imgbuf, img_write_sz, 1, ifp);
	fclose(ifp);
	fclose(lfp);
	free(imgpath);
	free(lblpath);
	if (lbw&&ibw)
		return true;
	else
		return false;
}

mnist_dataset_handle mnist_create_sample(const mnist_dataset_handle h, 
	unsigned int n)
{
	assert(h);
	unsigned int x, y, num_imgs;
	num_imgs = mnist_image_count(h);
	assert(num_imgs);
	mnist_image_size(h, &x, &y);
	
	if (n>num_imgs) return MNIST_DATASET_INVALID;

	//get randomly selected list of n out of num_img numbers.
	// first populate array fully

	mnist_dataset_handle s_mdh = mnist_create(x,y);
	if(n==0) return s_mdh;
	debug_print(" mnist_create_sample: num_imgs:%d\tn:%d\n", num_imgs, n);

	int *sample_idx = malloc(num_imgs*sizeof(int));
	int i;
	for(i=0;i<num_imgs;i++) { sample_idx[i] = i; }
	
	//shuffle list
	_fisher_yates_shuffle(sample_idx, num_imgs);
	
	//take first n elements of shuffled array
	void * rcheck = realloc(sample_idx, n*sizeof(int));
	if(!rcheck)
	{	
		free(sample_idx);
		if ( n!=0) 
		{	
			//out of memory, n is too large
			mnist_free(s_mdh); 
			return MNIST_DATASET_INVALID;
		}
		// n=0, so s_mdh is empty.
		return s_mdh;
	}
	sample_idx = rcheck;
	// sort array (to make buikding dataset more efficient)
	qsort((void*)sample_idx, n, sizeof(int), _compare_int);
	for(i=0;i<n;i++) {debug_print("mnist_create_sample: sample_idx[i]:%d\n", sample_idx[i]);}


	//traverse the original dataset in order, add the index to the 
	// sample dataset if the idx is in the sample_idx array
	mnist_image_handle s_img = mnist_image_begin(s_mdh);
	mnist_image_handle img = mnist_image_begin(h);
	const unsigned char * img_data = mnist_image_data(img);
	int img_lbl = mnist_image_label(img);
	int r=0, j=0;
	while (j<n)
	{
		// debug_print("r:%d\tsample_idx[r]:%d\n", r ,sample_idx[r]);
		if (r==(sample_idx[j]))
		{	
			debug_print(" mnist_create_sample: found img - r:%d\tsample_idx[j]:%d\n", r ,sample_idx[j]);
			img_data = mnist_image_data(img);
			img_lbl = mnist_image_label(img);
			s_img = mnist_image_add_after(s_mdh, s_img, img_data, x,y, img_lbl);
			assert(s_img);
			img = img->next;
			j++;
			r++;
			if(j<n) assert((img!=MNIST_IMAGE_INVALID));
		}
		else
		{
			r++;
			img=img->next;
			assert((img!=MNIST_IMAGE_INVALID));
			// debug_print("not found img - img->idx:%d\n",img->idx);
		}
	}
	free(sample_idx);
	debug_print(" mnist_create_sample: s_mdh:%p\n", (void*) s_mdh);
	return s_mdh;
}
