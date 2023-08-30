NVCC=nvcc
CC=gcc
CFLAGS=-Ofast -march=native
CUDAFLAGS=-arch=native
OPT=

OBJS=lenia.o

all: lenia

lenia: lenia.o
	${NVCC} ${OPT} $(CUDAFLAGS) -o lenia $(OBJS) $(LDLIBS)

%.o: %.cu
	${NVCC} -dc ${OPT} $(CUDAFLAGS) $< -o $@