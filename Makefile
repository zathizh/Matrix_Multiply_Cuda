multiplication : multiplication.cu
	nvcc -o multiplication multiplication.cu

all : multiplication
clean :
	rm multiplication
