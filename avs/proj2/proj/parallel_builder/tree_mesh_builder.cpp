/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Vladimir Marcin <xmarci10@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    15.12.2019
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::octreeDecomposition(const Vec3_t<float> &position, 
                                              const ParametricScalarField &field, 
                                              unsigned currentGridSize)
{
    unsigned trianglesCount = 0;

    if(currentGridSize <= 1) {
        return buildCube(position, field);
    } else {
        Vec3_t<float> centerOfCube((position.x + currentGridSize/2) * mGridResolution,
                                   (position.y + currentGridSize/2) * mGridResolution,
                                   (position.z + currentGridSize/2) * mGridResolution);
        float split = (mIsoLevel + ((sqrt(3)/2)) * (currentGridSize * mGridResolution)) - evaluateFieldAt(centerOfCube, field);
        
        if(split > 0) {
            for(unsigned i = 0; i < 8; i++) {
                #pragma omp task shared(trianglesCount)
                {
                    Vec3_t<float> childPosition;
                    childPosition.x = sc_vertexNormPos[i].x * (currentGridSize/2) + position.x;
                    childPosition.y = sc_vertexNormPos[i].y * (currentGridSize/2) + position.y;
                    childPosition.z = sc_vertexNormPos[i].z * (currentGridSize/2) + position.z;
                    #pragma omp atomic update
                    trianglesCount += octreeDecomposition(childPosition, field, currentGridSize/2);
                }
            }
            #pragma omp taskwait 
            return trianglesCount;
        } else {
            return 0;
        }
    }
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

    #pragma omp parallel 
    {
        #pragma omp single 
        {
            totalTriangles = octreeDecomposition(Vec3_t<float>(), field, mGridSize);
        }
    }

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    // const Vec3_t<float> *pPoints = field.getPoints().data();
    // const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    // #pragma omp simd simdlen(8)
    for(auto &point: field.getPoints())
    {
        float distanceSquared  = (pos.x - point.x) * (pos.x - point.x);
        distanceSquared       += (pos.y - point.y) * (pos.y - point.y);
        distanceSquared       += (pos.z - point.z) * (pos.z - point.z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    #pragma omp critical (saveTriangle)
    mTriangles.push_back(triangle);
}
