CFLAGS = -std=c11 -pedantic -Wall -Werror -g
CC = gcc
LFLAGS = -lcunit -lm
MNIST_FILES = src/mnist.h src/mnist.c
DIST_FILES = src/distance.h src/distance.c $(MNIST_FILES)
KNN_FILES = src/knn.h src/knn.c $(DIST_FILES)
TEST_FILES = src/test_mnist.c src/test_distance.c src/test_knn.c

all: src/main.c $(TEST_FILES) $(KNN_FILES)
	make test_mnist
	make test_distance
	make test_knn
	make ocr

mnist2pgm: src/mnist2pgm.c %(MNIST_FILES)
	$(CC) $(CFLAGS) -o $@ $(filter %.c, $^)

test_mnist_debug: src/test_mnist.c $(MNIST_FILES)
	$(CC) $(CFLAGS) -D DEBUG -o $@ $(filter %.c,$^) $(LFLAGS)

test_mnist: src/test_mnist.c $(MNIST_FILES)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LFLAGS)

test_distance_debug: src/test_distance.c $(DIST_FILES)
	$(CC) $(CFLAGS) -D DEBUG -o $@ $(filter %.c,$^) $(LFLAGS)

test_distance: src/test_distance.c $(DIST_FILES)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LFLAGS)

test_knn_debug: src/test_knn.c $(KNN_FILES)
	$(CC) $(CFLAGS) -D DEBUG -o $@ $(filter %.c,$^) $(LFLAGS)

test_knn: src/test_knn.c $(KNN_FILES)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LFLAGS)

ocr: src/main.c $(KNN_FILES)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LFLAGS)


.PHONY: clean test debug

test: $(TEST_FILES) $(KNN_FILES)
	make test_mnist
	make test_distance
	make test_knn
	./test_mnist
	./test_distance
	./test_knn

debug: $(TEST_FILES) $(KNN_FILES)
	make test_mnist_debug
	make test_distance_debug
	make test_knn_debug
	./test_mnist_debug
	./test_distance_debug
	./test_knn_debug

valgrind_test: $(TEST_FILES) $(KNN_FILES)
	make test_mnist
	make test_distance
	make test_knn
	valgrind --leak-check=full --show-reachable=yes --track-origins=yes ./test_mnist
	valgrind --leak-check=full --show-reachable=yes --track-origins=yes ./test_distance
	valgrind --leak-check=full --show-reachable=yes --track-origins=yes ./test_knn

clean:
	-rm ocr
	-rm test_distance
	-rm test_knn
	-rm test_mnist
	-rm test_distance_debug
	-rm test_knn_debug
	-rm test_mnist_debug
	-rm -R *.dSYM
	-rm src/*.o /*.o