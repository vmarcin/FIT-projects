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
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (thread_id < N) {
    float F, r, dx, dy, dz;
    float posx, posy, posz;

    /**
     * Ressetting the registers for partial results.
     * note:  Using registers reduces the number of accesses to global memory.
     *        Partial results are saved at the end of the calculation. Using
     *        registers also eliminates the need to reset the global memory (tmp_vel) 
     *        after each integration step (Instead we just reset the registers).     
     */
    float tmpvelx = 0.0f;
    float tmpvely = 0.0f;
    float tmpvelz = 0.0f;

    /**
     * Loading positions from the global memory into the registers.
     * note:  Pre-reading data from the global memory, reduces the number of 
     *        memory accesses and thus signigicantly speeds up the calculation.
     */
    posx = p.pos_x[thread_id];
    posy = p.pos_y[thread_id];
    posz = p.pos_z[thread_id];

    for (int j = 0; j < N; j++) {
      /**
       * The calculation of the gravitational force is divided into several
       * several instructions in order to eliminate data dependencies, and thus
       * we have increased the ILP.
       */
      F = -G * dt * p.weight[j];

      dx = posx - p.pos_x[j];
      dy = posy - p.pos_y[j];
      dz = posz - p.pos_z[j];

      r = sqrt(dx*dx + dy*dy + dz*dz);

      // see previous comment
      F /= (r * r * r + FLT_MIN);

      tmpvelx += (r > COLLISION_DISTANCE) ? F * dx : 0.0f;
      tmpvely += (r > COLLISION_DISTANCE) ? F * dy : 0.0f;
      tmpvelz += (r > COLLISION_DISTANCE) ? F * dz : 0.0f;
    }
    /**
     * Write partial results in global memory
     * note:  Only one store to global memory instead of store in each 
     *        cycle's iteration
     */
    tmp_vel.x[thread_id] = tmpvelx; 
    tmp_vel.y[thread_id] = tmpvely;
    tmp_vel.z[thread_id] = tmpvelz;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (thread_id < N) {
    float r, dx, dy, dz;
    float posx, posy, posz;
    float velx, vely, velz;
    float weight;

    /**
     * Ressetting the registers for partial results.
     * note:  Using registers reduces the number of accesses to global memory.
     *        Partial results are saved at the end of the calculation. Using
     *        registers also eliminates the need to reset the global memory (tmp_vel) 
     *        after each integration step (Instead we just reset the registers).     
     */
    float tmpvelx = 0.0f;
    float tmpvely = 0.0f;
    float tmpvelz = 0.0f;

    /**
     * Loading positions, velocities and weights from the global memory into the registers.
     * note:  Pre-reading data from the global memory, reduces the number of 
     *        memory accesses and thus signigicantly speeds up the calculation.
     */
    posx = p.pos_x[thread_id];
    posy = p.pos_y[thread_id];
    posz = p.pos_z[thread_id];
   
    velx = p.vel_x[thread_id];
    vely = p.vel_y[thread_id];
    velz = p.vel_z[thread_id];
   
    weight = p.weight[thread_id];

    for (int j = 0; j < N; j++) {
      /**
       * Loading the weight of the second particle as it will be used multiple times.
       * note:  It reduces the number of accesses to the global memory.
       */
      float weight_j = p.weight[j];

      dx = posx - p.pos_x[j];
      dy = posy - p.pos_y[j];
      dz = posz - p.pos_z[j];

      r = sqrt(dx*dx + dy*dy + dz*dz);

      if (r < COLLISION_DISTANCE) {
        /**
         * Reuseage of the registers of distances.
         * note:  The values are calculated only once and then used several times, see below.
         */
        dx = weight - weight_j;
        dy = 2 * weight_j;
        dz = weight + weight_j;

        tmpvelx += (r > 0.0f) ? ((dx * velx + dy * p.vel_x[j]) / dz) - velx : 0.0f;
        tmpvely += (r > 0.0f) ? ((dx * vely + dy * p.vel_y[j]) / dz) - vely : 0.0f;
        tmpvelz += (r > 0.0f) ? ((dx * velz + dy * p.vel_z[j]) / dz) - velz : 0.0f;
      }    
    }
    /**
     * Write partial results in global memory
     * note:  Only one store to global memory instead of store in each 
     *        cycle's iteration
     */
    tmp_vel.x[thread_id] += tmpvelx; 
    tmp_vel.y[thread_id] += tmpvely;
    tmp_vel.z[thread_id] += tmpvelz;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (thread_id < N) {
    p.vel_x[thread_id] += tmp_vel.x[thread_id];
    p.pos_x[thread_id] += p.vel_x[thread_id] * dt;

    p.vel_y[thread_id] += tmp_vel.y[thread_id];
    p.pos_y[thread_id] += p.vel_y[thread_id] * dt;

    p.vel_z[thread_id] += tmp_vel.z[thread_id];
    p.pos_z[thread_id] += p.vel_z[thread_id] * dt;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

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
