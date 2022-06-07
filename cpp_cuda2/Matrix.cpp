#include "Matrix.h"
using namespace std;

float max_length;
Road* matrix;

void initMatrix(City* cities) {
    matrix = new Road[n*n];
    int i,j;

    max_length = matrix[0 + 1].length = sqrt(pow(cities[0].x - cities[1].x, 2) + pow(cities[0].y - cities[1].y, 2));

    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i*n + j].length = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
            matrix[i*n + j].pheromone = pheromone_start_value;
            max_length = max(max_length, matrix[i*n + j].length);
        }
    }
    normalizeMatrix();
}

void normalizeMatrix() {
    int i,j;

    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i*n + j].length = (matrix[i*n + j].length * 100.0) / max_length;
        }
    }
}

void deleteMatrix(){
    delete[] matrix;
}

Road* getMatrix(){
    return matrix;
}

float getMaxMatrixLength() {
    return max_length;
}
