/**
 * @file      nbody.h
 *
 * @author    Vladimir Marcin \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xmarci10@fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2021
 *
 *
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include  <cmath>
#include "h5Helper.h"

#define INTEGRATE_QUEUE 1
#define COM_QUEUE       2
#define MEMCPY_QUEUE    3
#define UPDATE_QUEUE    4

/// Gravity constant
constexpr float G = 6.67384e-11f;

/// Collision distance threshold
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * @struct float4
 * Structure that mimics CUDA float4
 */
struct float4
{
  float x;
  float y;
  float z;
  float w;
};

/// Define sqrtf from CUDA libm library
#pragma acc routine(sqrtf) seq

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Structure with particle data
 */
struct Particles
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  size_t n;
  
  float *pos_x;
  float *pos_y;
  float *pos_z;

  float *vel_x;
  float *vel_y;
  float *vel_z;

  float *weight;

  Particles(size_t n) : n(n) {
    pos_x = new float[n];
    pos_y = new float[n];
    pos_z = new float[n];

    vel_x = new float[n];
    vel_y = new float[n];
    vel_z = new float[n];

    weight = new float[n];

    #pragma acc enter data copyin(this)

    #pragma acc enter data create(pos_x[0:n])
    #pragma acc enter data create(pos_y[0:n])
    #pragma acc enter data create(pos_z[0:n])

    #pragma acc enter data create(vel_x[0:n])
    #pragma acc enter data create(vel_y[0:n])
    #pragma acc enter data create(vel_z[0:n])

    #pragma acc enter data create(weight[0:n])
  }

  void updateHost() {
    #pragma acc update self(pos_x[0:n])
    #pragma acc update self(pos_y[0:n])
    #pragma acc update self(pos_z[0:n])

    #pragma acc update self(vel_x[0:n])
    #pragma acc update self(vel_y[0:n])
    #pragma acc update self(vel_z[0:n])

    #pragma acc update self(weight[0:n])
  }

  void updateHostAsync() {
    #pragma acc update self(pos_x[0:n]) async(MEMCPY_QUEUE)
    #pragma acc update self(pos_y[0:n]) async(MEMCPY_QUEUE)
    #pragma acc update self(pos_z[0:n]) async(MEMCPY_QUEUE)

    #pragma acc update self(vel_x[0:n]) async(MEMCPY_QUEUE)
    #pragma acc update self(vel_y[0:n]) async(MEMCPY_QUEUE)
    #pragma acc update self(vel_z[0:n]) async(MEMCPY_QUEUE)

    #pragma acc update self(weight[0:n]) async(MEMCPY_QUEUE)
  }

  void updateDevice() {
    #pragma acc update device(pos_x[0:n])
    #pragma acc update device(pos_y[0:n])
    #pragma acc update device(pos_z[0:n])

    #pragma acc update device(vel_x[0:n])
    #pragma acc update device(vel_y[0:n])
    #pragma acc update device(vel_z[0:n])

    #pragma acc update device(weight[0:n])
  }

  void copyData(const Particles &p) {
    for (unsigned int i = 0; i < n; i++) {
      pos_x[i] = p.pos_x[i];
      pos_y[i] = p.pos_y[i];
      pos_z[i] = p.pos_z[i];

      vel_x[i] = p.vel_x[i];
      vel_y[i] = p.vel_y[i];
      vel_z[i] = p.vel_z[i];

      weight[i] = p.weight[i];
    }
  }

  ~Particles() {
    #pragma acc exit data delete(pos_x)
    #pragma acc exit data delete(pos_y)
    #pragma acc exit data delete(pos_z)

    #pragma acc exit data delete(vel_x)
    #pragma acc exit data delete(vel_y)
    #pragma acc exit data delete(vel_z)

    #pragma acc exit data delete(weight)

    #pragma acc exit data delete(this)

    delete [] pos_x;
    delete [] pos_y;
    delete [] pos_z;

    delete [] vel_x;
    delete [] vel_y;
    delete [] vel_z;

    delete [] weight;
  }
};// end of Particles
//----------------------------------------------------------------------------------------------------------------------

/**
 * @struct Velocities
 * Velocities of the particles
 */
struct Velocities
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  size_t n;
  
  float *x;
  float *y;
  float *z;

  Velocities(size_t n) : n(n) {
    #pragma acc enter data copyin(this)

    #pragma acc enter data create(x[0:n])
    #pragma acc enter data create(y[0:n])
    #pragma acc enter data create(z[0:n])
  }

  ~Velocities() {
    #pragma acc exit data delete(x)
    #pragma acc exit data delete(y)
    #pragma acc exit data delete(z)

    #pragma acc exit data delete(this)
  }
};// end of Velocities
//----------------------------------------------------------------------------------------------------------------------

struct CenterOfMass
{
  float x;
  float y;
  float z;
  float w;

  CenterOfMass() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {
    #pragma acc enter data copyin(this)
  }

  void updateHostAsync() {
    #pragma acc update self(this) wait(UPDATE_QUEUE) async(MEMCPY_QUEUE)
  }

  void updateHost() {
    #pragma acc update self(this)
  }

  ~CenterOfMass() {
    #pragma acc exit data delete(this)
  }
};

/**
 * @brief  Calculate velocity
 * 
 * @param p_in  - input particles
 * @param p_out - output particles
 * @param N     - Number of particles
 * @param dt    - Size of the time step
 */
void calculate_velocity(const Particles& p_in,
                        Particles&       p_out,
                        const int        N,
                        const float      dt);

/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param [in] p - Particles
 * @param [in] N - Number of particles
 * @return Center of Mass [x, y, z] and total weight[w]
 */


/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param p [in] - Particles
 * @param com [out] - Center Of Mass [x, y, z] and total weight[w]
 * @param N [N] - Number of particles
 */
void centerOfMassGPU( const Particles&  p,
                      CenterOfMass&     com,
                      const int         N);
                 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compute center of mass on CPU
 * @param memDesc
 * @return centre of gravity
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
