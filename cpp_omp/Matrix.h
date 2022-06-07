#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <math.h>

extern int n;
extern double pheromone_start_value;
extern double pheromone_influence;
extern double length_influence;

struct City {
    double x;
    double y;
};

struct Road {
    double pheromone;
    double length;
};

void initMatrix(City* cities);
void deleteMatrix();
void normalizeMatrix();
Road** getMatrix();
double getMaxMatrixLength();

#endif
