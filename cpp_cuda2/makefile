PROGRAM_NAME=../bin/aco_cuda2
SOURCES=aco.cpp ant.cu Matrix.cpp
CXX = nvcc

all: compile

compile:
	$(CXX) $(SOURCES) -o $(PROGRAM_NAME) --use_fast_math