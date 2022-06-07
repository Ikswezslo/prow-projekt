#ifndef ANT_HPP
#define ANT_HPP

#include <iostream>
#include <cstdlib>
#include <math.h>
#include "Matrix.h"

extern int n;
extern double pheromone_influence;
extern double length_influence;

class Ant {
private:
    int actual_city;
    bool* is_visited;
    unsigned int seed;
    int select_next_city(Road** matrix);
    double calculate_solution_cost(Road** matrix);
    void clear();
public:
    Ant();
    ~Ant();
    int* solution;
    double solution_cost;
    int solution_size;
    void update(Road** matrix);
    void setSeed(unsigned int seed);
};
#endif
