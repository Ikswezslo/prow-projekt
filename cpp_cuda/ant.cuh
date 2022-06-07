#ifndef ANT_CUH
#define ANT_CUH

#include <stdio.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <curand.h>
#include "Matrix.h"

struct AntData {
    //Road *matrix
    bool* is_visited;
    //int* solution;
    int solution_size;
    int n;
    double pheromone_influence;
    double length_influence;
    double solution_cost;
    int actual_city;
};

void cuda_all_malloc(int ants, int n, double pheromone_influence, double length_influence, double pheromone_evaporation, double Q, Road *matrix);

void cuda_all_free();

void cuda_ants_update(int n, double* result, int* solutions, int ants);

void cuda_pheromone_update(int n, int ants);

__global__ void pheromone_decrease(Road* matrix, double* pheromone_evaporation, int *n);

__global__ void pheromone_increase(Road* matrix, int* solutions, double* Q, double* costs, int *n);

__global__ void ants_update(Road *matrix, int* n, 
                       double* pheromone_influence, 
                       double* length_influence, double* result, int* solutions,
                       curandState *states);

__device__ int select_next_city(Road* matrix, AntData* data, curandState *state);           

__device__ double calculate_solution_cost(Road* matrix, AntData* data,  int* solutions);

__device__ void clear(AntData* data, curandState *state,  int* solutions);
#endif