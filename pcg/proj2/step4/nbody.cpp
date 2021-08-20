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

void calculate_velocity(const Particles& p_in,
                        Particles&       p_out,
                        const int        N,
                        const float      dt)
{
  float F, r, dx, dy, dz;

  #pragma acc parallel loop present(p_in, p_out) gang worker vector async(INTEGRATE_QUEUE)
  for(unsigned int i = 0; i < N; i++) {
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

    /**
     * Since we do not have shared memory, we have extracted the most used 
     * elements from the global memory to the regiters.
     */
    float vel_x = p_in.vel_x[i];
    float vel_y = p_in.vel_y[i];
    float vel_z = p_in.vel_z[i];

    #pragma acc loop seq
    for (unsigned int j = 0; j < N; j++) {
      /**
       * The calculation of the gravitational force is divided into several
       * instructions in order to eliminate data dependencies, and thus
       * we have increased the ILP.
       */
      F = -G * dt * p_in.weight[j];

      dx = p_in.pos_x[i] - p_in.pos_x[j];
      dy = p_in.pos_y[i] - p_in.pos_y[j];
      dz = p_in.pos_z[i] - p_in.pos_z[j];

      r = sqrtf(dx*dx + dy*dy + dz*dz);

      // see previous comment
      F /= (r * r * r + FLT_MIN);

      // Add the velocity obtained by the gravitational action of the body 'j'.
      tmp_vel_x += (r > COLLISION_DISTANCE) ? F * dx : 0.0f;
      tmp_vel_y += (r > COLLISION_DISTANCE) ? F * dy : 0.0f;
      tmp_vel_z += (r > COLLISION_DISTANCE) ? F * dz : 0.0f;

      if (r < COLLISION_DISTANCE) {
      /**
       * Reuseage of the registers of distances.
       * note:  The values are calculated only once and then used several times, see below.
       */
        dx = p_in.weight[i] - p_in.weight[j];
        dy = 2 * p_in.weight[j];
        dz = p_in.weight[i] + p_in.weight[j];

        // Add the velocity obtained by the collision with the body 'j'.
        tmp_vel_x += (r > 0.0f) ? ((dx * vel_x + dy * p_in.vel_x[j]) / dz) - vel_x : 0.0f;
        tmp_vel_y += (r > 0.0f) ? ((dx * vel_y + dy * p_in.vel_y[j]) / dz) - vel_y : 0.0f;
        tmp_vel_z += (r > 0.0f) ? ((dx * vel_y + dy * p_in.vel_z[j]) / dz) - vel_y : 0.0f;
      }
    }
    /**
     * Update particle
     * note:  Write to global memory only once at the end of the cycle.
     */
    dx = vel_x + tmp_vel_x;
    p_out.vel_x[i] = dx;
    p_out.pos_x[i] = dx * dt + p_in.pos_x[i];

    dy = vel_y + tmp_vel_y;
    p_out.vel_y[i] = dy;
    p_out.pos_y[i] = dy * dt + p_in.pos_y[i];

    dz = vel_z + tmp_vel_z;
    p_out.vel_z[i] = dz;
    p_out.pos_z[i] = dz * dt + p_in.pos_z[i];
  }
}

/// Compute center of gravity
void centerOfMassGPU( const Particles&  p,
                      CenterOfMass&   com,
                      const int         N)
{

  float comx = 0.0f;
  float comy = 0.0f;
  float comz = 0.0f;
  float comw = 0.0f;

  #pragma acc parallel loop present(p) reduction(+:comx, comy, comz, comw) async(COM_QUEUE)
  for (unsigned int i = 0; i < N; i++) {
    comx += p.pos_x[i] * p.weight[i];
    comy += p.pos_y[i] * p.weight[i];
    comz += p.pos_z[i] * p.weight[i];

    comw += p.weight[i];
  }

  /**
   * Before saving the resulting values from local variables to the global GPU memory,
   * we need to wait for the reduction to complete => wait(COM_QUEUE). But since we 
   * want the save to be asynchronous as well, we add it to UPDATE_QUEUE. 
   * Then before copying the COM data back to the CPU, we have to wait for this 
   * queue to be emptied ( see CenterOfMass::updateHostAsync() ).
   */
  #pragma acc serial present(com) wait(COM_QUEUE) async(UPDATE_QUEUE)
  {
    com.x = comx / comw;
    com.y = comy / comw;
    com.z = comz / comw;
    com.w = comw;
  }
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
