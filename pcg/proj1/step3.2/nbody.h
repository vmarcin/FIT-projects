/**
 * @File nbody.h
 *
 * Header file of your implementation to the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xmarci10
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#define FULL_WARP_MASK 0xffffffff

/**
 * WARNING:   IF U ARE USING BLOCKDIM.X > 1024 CHANGE MAX_BLOCKDIMX VALUE.
 *            OTHERWISE THE RESULT WILL NOT BE CORRECT.
 */
#define MAX_BLOCKDIMX 1024

#include <cstdlib>
#include <cstdio>
#include "h5Helper.h"

/* Gravitation constant */
constexpr float G =  6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * Particles data structure
 */
typedef struct
{
    float *pos_x;
    float *pos_y;
    float *pos_z;

    float *vel_x;
    float *vel_y;
    float *vel_z;

    float *weight;
} t_particles;

/**
/* Velocities data structure (to be used as buffer for partial results)
 */
typedef struct
{
    float *x;
    float *y;
    float *z;
} t_velocities;

/**
 * CUDA kernel to calculate velocity and new position for each particle
 * @param p_in  - input particles
 * @param p_out - output particles
 * @param N     - Number of particles
 * @param dt    - Size of the time step
 */
__global__ void calculate_velocity( const t_particles p_in, 
                                    t_particles p_out, 
                                    int N, 
                                    float dt);
/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void centerOfMass(t_particles p,
                             float*      comX,
                             float*      comY,
                             float*      comZ,
                             float*      comW,
                             int*        lock,
                             const int   N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
