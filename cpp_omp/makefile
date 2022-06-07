PROGRAM_NAME=../bin/aco_omp
SOURCES=aco.cpp Ant.cpp Matrix.cpp
CXX = g++

all: compile

compile:
	$(CXX) $(SOURCES) -o $(PROGRAM_NAME) -Wall -fopenmp -O2
