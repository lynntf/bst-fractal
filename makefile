coverage: coverage.o
	nvcc -o coverage_exe -arch sm_35 coverage.o

coverage.o: coverage.cu
	nvcc -c -arch sm_35 coverage.cu

clean:
	-rm coverage
	-rm *.o
