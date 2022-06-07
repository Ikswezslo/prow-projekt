#include "Ant.h"
using namespace std;

Ant::Ant() {
    is_visited = new bool[n];
    solution = new int[n];
    solution_size = 0;
}

Ant::~Ant() {
    delete[] is_visited;
    delete[] solution;
}

void Ant::update(Road** matrix) {
    clear();
    for(int i = 1; i < n; i++) {
        actual_city = select_next_city(matrix);
        is_visited[actual_city] = true;
        solution[i] = actual_city;
        solution_size++;
    }
    solution_cost = calculate_solution_cost(matrix);
}

int Ant::select_next_city(Road** matrix) {
    double* probability = new double[n - solution_size];
    int* candidate = new int[n - solution_size];
    int number_of_candidates = 0;
    double candidate_probability, sum_probability = 0;
    for(int i = 0; i < n; i++) {
        if(!is_visited[i]) {
            candidate_probability = powf(matrix[actual_city][i].pheromone,pheromone_influence) * powf(1/matrix[actual_city][i].length,length_influence);
            sum_probability += candidate_probability;
            candidate[number_of_candidates] = i;
            probability[number_of_candidates] = candidate_probability;
            number_of_candidates++;
        }
    }

    for(int i = 0; i < number_of_candidates; i++)
        probability[i] /= sum_probability;

    double cumulative_probability = 0;
    int selected_candidate = -1;
    double random = (double)rand_r(&seed)/RAND_MAX;
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

double Ant::calculate_solution_cost(Road** matrix) {
    double result = 0;
    int j;
    for(int i = 0; i < n; i++) {
        j = (i+1) % n;
        result += matrix[solution[i]][solution[j]].length;
    }
    return result;
}

void Ant::clear() {
    actual_city = rand_r(&seed) % n;
    for(int i = 0; i < n; i++)
        is_visited[i] = false;
    is_visited[actual_city] = true;
    solution[0] = actual_city;
    solution_size = 1;
}

void Ant::setSeed(unsigned int seed) {
    this->seed = seed;
}
