#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>

#include "helperFunctions.h"

using namespace std;

vector<float3> convertFloatPointsToVectorPoint(float3 *_points, int count){
    vector<float3> points;
    for(int i=0; i<count; i++){
        float3 point;
        point.x = _points[i].x;
        point.y = _points[i].x;
        point.z = _points[i].x;
        if (point.z > 0){
            points.push_back(point);
        }
    }
    return points;
};

//Write to Ply file
void writeToPly(vector<float3> points, const char* fileName){
    
    string s = "ply\nformat ascii 1.0\nelement vertex "+to_string(points.size())+"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    FILE * bfile;
    bfile = fopen (fileName, "wb");
    unsigned int N(s.size());
    fwrite(s.c_str(),1, N ,bfile);
    for(int i = 0; i < points.size(); i++){
        string pointS = to_string(points[i].x)+" "+to_string(points[i].y)+" "+to_string(points[i].z);
        if(i != points.size()-1) pointS = pointS+"\n";
        unsigned int Ns(pointS.size());
        fwrite(pointS.c_str(),1,Ns,bfile);
    }
    fflush(bfile);
    fclose (bfile);
};
