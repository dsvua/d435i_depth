#include <vector> // for 2D vector

struct Point{
    float x, y, z;
};

std::vector<Point> convertFloatPointsToVectorPoint(float *_points, int count);
void writeToPly(std::vector<Point> points, const char* fileName);