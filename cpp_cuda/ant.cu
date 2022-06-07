#include  "ant.cuh"

Road * d_array;
int* d_n;
double* d_pheromone_influence;
double* d_length_influence;
double* d_pheromone_evaporation;
double* d_Q;
double* d_result;
int* d_solutions;
curandState *d_random;

void cuda_all_malloc(int ants, int n, double pheromone_influence, double length_influence, double pheromone_evaporation, double Q, Road *matrix) {
    cudaMalloc((void**)&d_random, ants * sizeof(curandState));
    cudaMalloc((void **)&d_array, n * n * sizeof(Road));
    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMalloc((void **)&d_pheromone_influence, sizeof(double));
    cudaMalloc((void **)&d_pheromone_evaporation, sizeof(double));
    cudaMalloc((void **)&d_Q, sizeof(double));
    cudaMalloc((void **)&d_length_influence, sizeof(double));
    cudaMalloc((void **)&d_result, ants * sizeof(double));
    cudaMalloc((void **)&d_solutions, ants * n * sizeof(int));

    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pheromone_influence, &pheromone_influence, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_length_influence, &length_influence, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pheromone_evaporation, &pheromone_evaporation, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, &Q, sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_array, matrix, n * n * sizeof(Road), cudaMemcpyHostToDevice);

}

void cuda_all_free() {
    cudaFree(d_random);
    cudaFree(d_array);
    cudaFree(d_n);
    cudaFree(d_pheromone_influence);
    cudaFree(d_pheromone_evaporation);
    cudaFree(d_Q);
    cudaFree(d_length_influence);    
    cudaFree(d_result);
    cudaFree(d_solutions);
}


void cuda_ants_update(int n, double* result, int* solutions, int ants) {
    ants_update<<<ants,1>>>(d_array, d_n, d_pheromone_influence, d_length_influence, d_result, d_solutions, d_random);
    cudaMemcpy(result, d_result, ants * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(solutions, d_solutions, ants * n * sizeof(int), cudaMemcpyDeviceToHost);
}

void cuda_pheromone_update(int n, int ants) {
    pheromone_decrease<<<n,1>>>(d_array, d_pheromone_evaporation, d_n);
    pheromone_increase<<<ants,1>>>(d_array, d_solutions, d_Q, d_result, d_n);
}

__global__ void pheromone_decrease(Road* matrix, double* pheromone_evaporation, int *n) {
    int j;
    double local_pheromone_evaporation  = *pheromone_evaporation;
    int local_n =  *n;
    for(j = 0; j < local_n; j++) {
        matrix[blockIdx.x*local_n + j].pheromone = (1 - local_pheromone_evaporation) * matrix[blockIdx.x*local_n + j].pheromone;
    }
}

__global__ void pheromone_increase(Road* matrix, int* solutions, double* Q, double* costs, int *n) {
    int i,j;
    double local_Q = *Q;
    int local_n =  *n;
    for(i = 0; i < local_n; i++) {
        j = (i+1) % local_n;
        matrix[(&solutions[blockIdx.x*local_n])[i]*local_n + (&solutions[blockIdx.x*local_n])[j]].pheromone += local_Q / costs[blockIdx.x];
        matrix[(&solutions[blockIdx.x*local_n])[j]*local_n + (&solutions[blockIdx.x*local_n])[i]].pheromone += local_Q / costs[blockIdx.x];
    }
}

__global__ void ants_update(Road *matrix, int* n, 
                       double* pheromone_influence, 
                       double* length_influence, double* result, int* solutions,
                       curandState *states) {
    int id = blockIdx.x;
	int seed = id;
    curand_init(seed, id, 0, &states[id]); 
    curandState* state = &states[id];
    AntData *data = new AntData;
    data->n = *n;
    data->pheromone_influence = *pheromone_influence;
    data->length_influence = *length_influence;
    data->length_influence = *length_influence;
    data->is_visited = new bool[data->n];
    data->solution_size = 0;
    data->solution_cost = 0;
    data->actual_city = -1;
    clear(data, state, solutions);
    for(int i = 1; i < data->n; i++) {
        data->actual_city = select_next_city(matrix, data, state);
        data->is_visited[data->actual_city] = true;
        solutions[blockIdx.x * (data->n) + i] = data->actual_city;
        data->solution_size++;
    }
    data->solution_cost = calculate_solution_cost(matrix, data, solutions);
    result[blockIdx.x] = data->solution_cost;
    delete data;
    delete[] data->is_visited;
}

__device__ int select_next_city(Road* matrix, AntData* data, curandState *state) {
    float* probability = new float[data->n - data->solution_size];
    int* candidate = new int[data->n - data->solution_size];
    int number_of_candidates = 0;
    float candidate_probability, sum_probability = 0;

    for(int i = 0; i < data->n; i++) {
        if(!(data->is_visited[i])) {
            candidate_probability = powf(matrix[data->actual_city * data->n + i].pheromone,data->pheromone_influence) * powf(1/matrix[data->actual_city * data->n + i].length,data->length_influence);
            sum_probability += candidate_probability;
            candidate[number_of_candidates] = i;
            probability[number_of_candidates] = candidate_probability;
            number_of_candidates++;
        }
    }

    for(int i = 0; i < number_of_candidates; i++)
        probability[i] /= sum_probability;

    float cumulative_probability = 0;
    int selected_candidate = -1;
    float random = curand_uniform(state);
    for(int i = 0; i < number_of_candidates; i++){
        cumulative_probability += probability[i];
        if(random <= cumulative_probability) {
            selected_candidate = candidate[i];
            break;
        }
    }
    if(selected_candidate == -1)
        selected_candidate = candidate[number_of_candidates - 1];

    delete[] probability;
    delete[] candidate;
    return selected_candidate;
}

__device__ double calculate_solution_cost(Road* matrix, AntData* data, int* solutions) {
    double result = 0;
    int j;
    for(int i = 0; i < data->n; i++) {
        j = (i+1) % data->n;
        result += matrix[solutions[blockIdx.x * (data->n) + i] * data->n + solutions[blockIdx.x * (data->n) + j]].length;
    }
    return result;
}

__device__ void clear(AntData* data, curandState *state, int* solutions) {
    data->actual_city = int(curand_uniform(state) * data->n);
    if(data->actual_city > data->n -1)
        data->actual_city = data->n - 1;
    for(int i = 0; i < data->n; i++)
        data->is_visited[i] = false;
    data->is_visited[data->actual_city] = true;
    solutions[blockIdx.x * (data->n) + 0] = data->actual_city;
    data->solution_size = 1;
}