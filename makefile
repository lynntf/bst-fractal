coverage: coverage.o
	nvcc -o coverage_exe coverage.o

coverage.o: coverage.cu
	nvcc -c coverage.cu

clean:
	-rm coverage
	-rm *.o
