/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xmarci10
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate velocity and new position for each particle
 * @param p_in  - input particles
 * @param p_out - output particles
 * @param N     - Number of particles
 * @param dt    - Size of the time step
 */
__global__ void calculate_velocity(const t_particles p_in, t_particles p_out, int N, float dt)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (thread_id < N) {
    float r, dx, dy, dz;
    float posx, posy, posz;
    float velx, vely, velz;
    float weight; 
    float F;

    /**
     * Ressetting the registers for partial results.
     * note:  Using registers reduces the number of accesses to global memory.
     *        Partial results are saved at the end of the calculation.      
     */
    float tmpvelx = 0.0f;
    float tmpvely = 0.0f;
    float tmpvelz = 0.0f;

    /**
     * Loading positions, velocities and weights from the global memory into the registers.
     * note:  Pre-reading data from the global memory, reduces the number of 
     *        memory accesses and thus signigicantly speeds up the calculation.
     */
    posx = p_in.pos_x[thread_id];
    posy = p_in.pos_y[thread_id];    
    posz = p_in.pos_z[thread_id];
    
    velx = p_in.vel_x[thread_id];
    vely = p_in.vel_y[thread_id];
    velz = p_in.vel_z[thread_id];

    weight = p_in.weight[thread_id];
    
    for (int j = 0; j < N; j++) {
      /**
       * Loading the weight of the second particle as it will be used multiple times.
       * note:  It reduces the number of accesses to the global memory.
       */
      float weight_j = p_in.weight[j];
      
      /**
       * The calculation of the gravitational force is divided into several
       * several instructions in order to eliminate data dependencies, and thus
       * we have increased the ILP.
       */
      F = -G * dt * weight_j;

      dx = posx - p_in.pos_x[j];
      dy = posy - p_in.pos_y[j];
      dz = posz - p_in.pos_z[j];

      r = sqrt(dx*dx + dy*dy + dz*dz);

      // see previous comment
      F /= (r * r * r + FLT_MIN);

      // Add the velocity obtained by the gravitational action of the body 'j'.
      tmpvelx += (r > COLLISION_DISTANCE) ? F * dx : 0.0f;
      tmpvely += (r > COLLISION_DISTANCE) ? F * dy : 0.0f;
      tmpvelz += (r > COLLISION_DISTANCE) ? F * dz : 0.0f;

      if (r < COLLISION_DISTANCE) {
        /**
         * Reuseage of the registers of distances.
         * note:  The values are calculated only once and then used several times, see below.
         */
        dx = weight - weight_j;
        dy = 2 * weight_j;
        dz = weight + weight_j;

        // Add the velocity obtained by the collision with the body 'j'.
        tmpvelx += (r > 0.0f) ? ((dx * velx + dy * p_in.vel_x[j]) / dz) - velx : 0.0f;
        tmpvely += (r > 0.0f) ? ((dx * vely + dy * p_in.vel_y[j]) / dz) - vely : 0.0f;
        tmpvelz += (r > 0.0f) ? ((dx * velz + dy * p_in.vel_z[j]) / dz) - velz : 0.0f;
      }
    }
    /**
     * Update particle
     * note:  Write to global memory only once at the end of the cycle.
     */
    velx += tmpvelx;
    p_out.vel_x[thread_id] = velx;
    p_out.pos_x[thread_id] = velx * dt + posx;

    vely += tmpvely;
    p_out.vel_y[thread_id] = vely;
    p_out.pos_y[thread_id] = vely * dt + posy;

    velz += tmpvelz;
    p_out.vel_z[thread_id] = velz;
    p_out.pos_z[thread_id] = velz * dt + posz;
  }
}// end of calculate_velocity
//-----------------------------------------------------------------------------------------------------------------------

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
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc)
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
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
