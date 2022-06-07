#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "Matrix.h"
#include "Ant.h"

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

void generate_solutions(Road** matrix, Ant* ant) {
    int i;
    unsigned int ziarno;
    unsigned int seed = time(NULL);

    ziarno = seed;
    for(i = 0; i < ant_number; i++) {
        ant[i].setSeed(ziarno);
        ant[i].update(matrix);
    }
}

void compare_solution(Ant* ant) {
    if(best_solution == NULL) {
        best_solution = ant[0].solution;
        best_solution_cost = ant[0].solution_cost;
    }

    double cost;
    for(int i = 0; i < ant_number; i++) {
        cost = ant[i].solution_cost;
        if(cost < best_solution_cost) {
            best_solution_cost = cost;
            best_solution = ant[i].solution;
        }
    }
}

void pheromone_update(Road** matrix, Ant* ant) {

    int i,j,k;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            matrix[i][j].pheromone = (1 - pheromone_evaporation) * matrix[i][j].pheromone;
        }
    }
    
    for(k = 0; k < ant_number; k++) {
        for(i = 0; i < n; i++) {
            int j = (i+1) % n;
            matrix[ant[k].solution[i]][ant[k].solution[j]].pheromone += Q / ant[k].solution_cost;
            matrix[ant[k].solution[j]][ant[k].solution[i]].pheromone += Q / ant[k].solution_cost;
        }
    }
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
    Road** matrix = getMatrix();
    Ant* ant = new Ant[ant_number];
    for(int i = 0; i < iteration_number; i++) {
        generate_solutions(matrix, ant);
        compare_solution(ant);
        pheromone_update(matrix, ant);
    }
    double stop = clock();

    cout << "Czas: " << (stop - start) / (double)CLOCKS_PER_SEC  << endl;
    cout << "Wynik: " << (best_solution_cost * getMaxMatrixLength()) / 100.0 << endl;
    cout << "Rozwiazanie: ";
    for(int i = 0; i < n; i++)
        cout << best_solution[i] << ' ';
    cout << endl;

    deleteMatrix();
    delete[] ant;
    return 0;
}
