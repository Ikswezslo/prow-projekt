#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "Matrix.h"
#include "ant.cuh"

using namespace std;

int n;

int ant_number;
int iteration_number;
double pheromone_evaporation;
double Q;
double pheromone_start_value;
double pheromone_influence;
double length_influence;
int num_threads;

int* best_solution = NULL;
double best_solution_cost = 0;

void generate_solutions(double* result, int* solutions) {
    cuda_ants_update(n, result, solutions, ant_number);
}

void compare_solution(double* costs, int* solutions) {
    if(best_solution == NULL) {
        best_solution = &solutions[0];
        best_solution_cost = costs[0];
    }

    double cost;
    for(int i = 0; i < ant_number; i++) {
        
        cost = costs[i];
        if(cost < best_solution_cost) {
            best_solution_cost = cost;
            best_solution = &solutions[i*n];
        }
    }
}

void pheromone_update() {
    cuda_pheromone_update(n, ant_number);
}

int main() {
    srand( time( NULL ) );

    string temp;
    ifstream config("aco_config.txt");
    config >> temp >> ant_number;
    config >> temp >> iteration_number;
    config >> temp >> pheromone_start_value;
    config >> temp >> pheromone_evaporation;
    config >> temp >> pheromone_influence;
    config >> temp >> length_influence;
    config >> temp >> Q;
    config >> temp >> num_threads;
    config.close();

    ifstream inst("inst.txt");
    inst >> n;
    
    City* cities = new City[n];
    for(int i = 0; i < n; i++)
        inst >> temp >> cities[i].x  >> cities[i].y;
    inst.close();
    
    double start = clock();
    initMatrix(cities);
    delete[] cities;
    Road* matrix = getMatrix();
    double* result = new double[ant_number]; 
    int* solutions = new int[ant_number*n]; 
    cuda_all_malloc(ant_number, n, pheromone_influence, length_influence, pheromone_evaporation, Q, matrix);
    for(int i = 0; i < iteration_number; i++) {
        generate_solutions(result, solutions);
        compare_solution(result, solutions);
        pheromone_update();
    }
    cuda_all_free();
    
    double stop = clock();

    cout << "Czas: " << (stop - start) / (double)CLOCKS_PER_SEC  << endl;
    cout << "Wynik: " << (best_solution_cost * getMaxMatrixLength()) / 100.0 << endl;
    cout << "Rozwiazanie: ";
    for(int i = 0; i < n; i++)
        cout << best_solution[i] << ' ';
    cout << endl;

    delete[] result;
    delete[] solutions;
    deleteMatrix();
    return 0;
}
