FLAGS = -DDEBUG
LIBS  = -lm
ALWAYS_REBUILD = makefile

TARGET = nbody

NVCC = nvcc
GCC  = gcc

all: $(TARGET)

$(TARGET): nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS) -lcudart

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(GCC) $(FLAGS) -c $<

compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $<

clean:
	rm -f *.o $(TARGET)

#FLAGS= -DDEBUG
#LIBS= -lm
#ALWAYS_REBUILD=makefile

#nbody: nbody.o compute.o
#	gcc $(FLAGS) $^ -o $@ $(LIBS)
#nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
#	gcc $(FLAGS) -c $< 
#compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
#	gcc $(FLAGS) -c $< 
#clean:
#	rm -f *.o nbody 
