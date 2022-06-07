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
    float pheromone;
    float length;
};

void initMatrix(City* cities);
void deleteMatrix();
void normalizeMatrix();
Road* getMatrix();
float getMaxMatrixLength();

#endif
