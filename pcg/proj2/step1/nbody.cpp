/**
 * @file      nbody.cpp
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

#include <math.h>
#include <cfloat>
#include "nbody.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute gravitation velocity
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt)
{
  float F, r, dx, dy, dz;
  
  #pragma acc parallel loop present(p, tmp_vel) gang worker vector
  for (unsigned int i = 0; i < N; i++) {
    /**
     * Ressetting the registers for partial results.
     * note:  Using registers reduces the number of accesses to global memory.
     *        Partial results are saved at the end of the calculation. Using
     *        registers also eliminates the need to reset the global memory (tmp_vel) 
     *        after each integration step (Instead we just reset the registers).     
     */
    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    #pragma acc loop seq
    for (unsigned int j = 0; j < N; j++) {
      /**
       * The calculation of the gravitational force is divided into several
       * instructions in order to eliminate data dependencies, and thus
       * we have increased the ILP.
       */
      F = -G * dt * p.weight[j];

      dx = p.pos_x[i] - p.pos_x[j];
      dy = p.pos_y[i] - p.pos_y[j];
      dz = p.pos_z[i] - p.pos_z[j];

      r = sqrtf(dx*dx + dy*dy + dz*dz);

      // see previous comment
      F /= (r * r * r + FLT_MIN);

      tmp_vel_x += (r > COLLISION_DISTANCE) ? F * dx : 0.0f;
      tmp_vel_y += (r > COLLISION_DISTANCE) ? F * dy : 0.0f;
      tmp_vel_z += (r > COLLISION_DISTANCE) ? F * dz : 0.0f;
    }
    /**
     * Write partial results into the global memory
     * note:  Only one store to global memory instead of store in each 
     *        cycle's iteration
     */
    tmp_vel.x[i] = tmp_vel_x;
    tmp_vel.y[i] = tmp_vel_y;
    tmp_vel.z[i] = tmp_vel_z;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt)
{
  float r, dx, dy, dz;

  #pragma acc parallel loop present(p, tmp_vel) gang worker vector
  for (unsigned int i = 0; i < N; i++) {
    /**
     * Ressetting the registers for partial results.
     * note:  Using registers reduces the number of accesses to global memory.
     *        Partial results are saved at the end of the calculation. Using
     *        registers also eliminates the need to reset the global memory (tmp_vel) 
     *        after each integration step (Instead we just reset the registers).     
     */
    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    #pragma acc loop seq
    for (unsigned int j = 0; j < N; j++) {
      dx = p.pos_x[i] - p.pos_x[j];
      dy = p.pos_y[i] - p.pos_y[j];
      dz = p.pos_z[i] - p.pos_z[j];

      r = sqrtf(dx*dx + dy*dy + dz*dz);

      if (r < COLLISION_DISTANCE) {
        /**
         * Reuseage of the registers of distances.
         * note:  The values are calculated only once and then used several times, see below.
         */
        dx = p.weight[i] - p.weight[j];
        dy = 2 * p.weight[j];
        dz = p.weight[i] + p.weight[j];

        tmp_vel_x += (r > 0.0f) ? ((dx * p.vel_x[i] + dy * p.vel_x[j]) / dz) - p.vel_x[i] : 0.0f;
        tmp_vel_y += (r > 0.0f) ? ((dx * p.vel_y[i] + dy * p.vel_y[j]) / dz) - p.vel_y[i] : 0.0f;
        tmp_vel_z += (r > 0.0f) ? ((dx * p.vel_z[i] + dy * p.vel_z[j]) / dz) - p.vel_z[i] : 0.0f;
      }
    }
    /**
     * Write partial results in global memory
     * note:  Only one store to global memory instead of store in each 
     *        cycle's iteration
     */
    tmp_vel.x[i] += tmp_vel_x;
    tmp_vel.y[i] += tmp_vel_y;
    tmp_vel.z[i] += tmp_vel_z;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt)
{
  #pragma acc parallel loop present(p, tmp_vel) 
  for (unsigned int i = 0; i < N; i++) {
    p.vel_x[i] += tmp_vel.x[i];
    p.pos_x[i] += p.vel_x[i] * dt;

    p.vel_y[i] += tmp_vel.y[i];
    p.pos_y[i] += p.vel_y[i] * dt;

    p.vel_z[i] += tmp_vel.z[i];
    p.pos_z[i] += p.vel_z[i] * dt;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------



/// Compute center of gravity
float4 centerOfMassGPU(const Particles& p,
                       const int        N)
{

  return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
