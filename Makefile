CC=g++
NVCC=nvcc
OPENCV = `pkg-config opencv --cflags --libs`
LIBS=-lcuda -lcudart $(OPENCV)
CFLAGS=-I. -L/opt/cuda/lib64 $(LIBS) -g
CUFLAGS=-lcuda -g
DEPS = *.h
OBJ = patchnorm_cpu.o patchnorm_cuda.o
# CU = add_arr.cu
# CUOBJ = add_arr.o
OBJDIR = out/obj
OUTDIR = out/

all: $(OBJ) | $(OUTDIR) $(OBJDIR)
	$(CC) -o $(OUTDIR)/patchnorm_test patchnorm.cpp $(OBJDIR)/patchnorm_cpu.o $(OBJDIR)/patchnorm_cuda.o $(CFLAGS)

$(OUTDIR):
	mkdir $(OUTDIR)

$(OBJDIR):| $(OUTDIR)
	mkdir $(OBJDIR)

%.o: %.cpp $(DEPS) 
	$(CC) -c -o $(OBJDIR)/$@ $< $(CFLAGS)

%.o: %.cu $(DEPS) 
	$(NVCC) -c -o $(OBJDIR)/$@ $< $(CUFLAGS)


