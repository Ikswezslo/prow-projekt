#include "Matrix.h"
using namespace std;

double max_length;
Road** matrix;

void initMatrix(City* cities) {
    matrix = new Road*[n];
    int i,j;

    for(i = 0; i < n; i++)
        matrix[i] = new Road[n];

    max_length = matrix[0][1].length = sqrt(pow(cities[0].x - cities[1].x, 2) + 
                 pow(cities[0].y - cities[1].y, 2));

    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i][j].length = sqrt(pow(cities[i].x - cities[j].x, 2) + 
                                  pow(cities[i].y - cities[j].y, 2));
            matrix[i][j].pheromone = pheromone_start_value;
            max_length = max(max_length, matrix[i][j].length);
        }
    }
    normalizeMatrix();
}

void normalizeMatrix() {
    int i,j;
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i][j].length = (matrix[i][j].length * 100.0) / max_length;
        }
    }
}

void deleteMatrix(){
    for(int i = 0; i < n; i++)
        delete[] matrix[i];
    delete[] matrix;
}

Road** getMatrix(){
    return matrix;
}

double getMaxMatrixLength() {
    return max_length;
}
