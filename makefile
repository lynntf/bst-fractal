coverage: coverage.o
	nvcc -o coverage coverage.o

coverage.o: coverage.cu
	nvcc -c coverage.cu

clean:
	-rm coverage
	-rm *.o
