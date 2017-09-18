#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>



int main (int argc, char ** args)
{
   if (argc != 3)
   {
      fprintf (stderr, "Need 2 arguments:\n  - dataset name\n  - number of images to convert. Use 0 for all.");
      exit(1);
   }

   const char * name = args[1];
   unsigned int count = atoi(args[2]);

   // Open dataset
   mnist_dataset_handle h = mnist_open (name);
   if (!h)
   {
      fprintf (stderr, "Couldn't open dataset '%s'!\n", name);
      return 1;
   }

   const unsigned int imagecount = mnist_image_count (h);
   unsigned int width, height;
   mnist_image_size (h, &width, &height);

   printf ("Images=%u, width=%u, height=%u\n", imagecount, width, height);

   if (!count)
      count = imagecount;

   char filename[255];
   mnist_image_handle img = mnist_image_begin(h);
   for (unsigned int i=0; i<count; ++i)
   {
      if (!img)
      {
         fprintf (stderr, "Not enough images in dataset!\n");
         break;
      }

      snprintf (filename, sizeof(filename)-1, "%s-%05u.pgm", name, i);
      // Ensure 0-terminated
      filename[sizeof(filename)-1]=0;

      FILE * f = fopen (filename, "w");
      if (!f)
      {
         perror ("Couldn't write to file: ");
         mnist_free (h);
         return 1;
      }

      printf ("Writing %s...\n", filename);

      fputs ("P5\n", f);
      fprintf (f, "# %s %u/%u\n", name, i, imagecount);
      fprintf (f, "%u %u\n", width, height);
      fputs ("255\n", f);

      const unsigned char * data = mnist_image_data (img);
      for (unsigned int y=0; y<height; ++y)
      {
         for (unsigned int x=0; x<width; ++x)
         {
            unsigned int c = *data;
            fprintf (f, "%u ", c);
            ++data;
         }
         fputs ("\n", f);
      }
      fclose (f);

      img = mnist_image_next (img);
   }

   mnist_free (h);
   exit(EXIT_SUCCESS);
}
