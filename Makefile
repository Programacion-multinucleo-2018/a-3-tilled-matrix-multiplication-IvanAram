CUDA_C = nvcc

CFLAGS = -std=c++11

EXE1 = tilledMatrixMult
EXE2 = tilledMatrixMultComparison

PROG1 = tilledMatrixMult.cu
PROG2 = tilledMatrixMultComparison.cu

all:
	$(CUDA_C) -o $(EXE1) $(PROG1) $(CFLAGS)
	$(CUDA_C) -o $(EXE2) $(PROG2) $(CFLAGS)

rebuild: clean all

clean:
	rm -f $(EXE1) $(EXE2)
