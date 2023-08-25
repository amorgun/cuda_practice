CC=gcc
CFLAGS=-Ofast -march=native
LDLIBS = -lm

SRCS=lenia.c
OBJS=lenia.o


all: lenia

lenia: ${OBJS}
	$(CC) $(LDFLAGS) -o lenia $(OBJS) $(LDLIBS)

lenia.o: lenia.c
