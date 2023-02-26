#include <stdbool.h>
#include "sdecomp.h"
#include "internal.h"


MPI_Comm sdecomp_internal_get_comm_cart(const sdecomp_info_t *info){
  return info->comm_cart;
}

/*** process distributions ***/

typedef enum {
  TYPE_NPROCS = 0,
  TYPE_MYRANK = 1
} type_t;

// valid directions:
//   2D: SDECOMP_XDIR, SDECOMP_YDIR
//   3D: SDECOMP_XDIR, SDECOMP_YDIR, SDECOMP_ZDIR
static bool dir_is_invalid(const size_t ndims, const sdecomp_dir_t dir){
  if(2 == ndims){
    const bool is_valid
      =  SDECOMP_XDIR == dir
      || SDECOMP_YDIR == dir;
    return !is_valid;
  }else{
    const bool is_valid
      =  SDECOMP_XDIR == dir
      || SDECOMP_YDIR == dir
      || SDECOMP_ZDIR == dir;
    return !is_valid;
  }
}

// valid pencils:
//   2D: SDECOMP_X1PENCIL, SDECOMP_Y1PENCIL
//   3D: SDECOMP_X1PENCIL, SDECOMP_Y1PENCIL, SDECOMP_Z1PENCIL
//       SDECOMP_X2PENCIL, SDECOMP_Y2PENCIL, SDECOMP_Z2PENCIL
static bool pencil_is_invalid(const size_t ndims, const sdecomp_pencil_t pencil){
  if(2 == ndims){
    const bool is_valid
      =  SDECOMP_X1PENCIL == pencil
      || SDECOMP_Y1PENCIL == pencil;
    return !is_valid;
  }else{
    const bool is_valid
      =  SDECOMP_X1PENCIL == pencil
      || SDECOMP_Y1PENCIL == pencil
      || SDECOMP_Z1PENCIL == pencil
      || SDECOMP_X2PENCIL == pencil
      || SDECOMP_Y2PENCIL == pencil
      || SDECOMP_Z2PENCIL == pencil;
    return !is_valid;
  }
}

static int get_process_config_2d(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const type_t type){
#define SDECOMP_INTERNAL_NDIMS 2
  // compute
  //   type == TYPE_NPROCS: number of processes
  // or
  //   type == TYPE_MYRANK: my location
  // of the given decomposition "info"
  //   in the given dimension "dim"
  // x1 pencil, get info using MPI_Cart_get
  int x1pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int         periods[SDECOMP_INTERNAL_NDIMS] = {0};
  int x1pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  MPI_Cart_get(
      sdecomp.get_comm_cart(info),
      SDECOMP_INTERNAL_NDIMS,
      x1pencil_nprocs,
      periods,
      x1pencil_myrank
  );
  if(pencil == SDECOMP_X1PENCIL){
    // return x1pencil info
    return type == TYPE_NPROCS ? x1pencil_nprocs[dir] : x1pencil_myrank[dir];
  }
  // y1 pencil, determined by x1 pencil
  int y1pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int y1pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  y1pencil_nprocs[0] = x1pencil_nprocs[1];
  y1pencil_nprocs[1] = x1pencil_nprocs[0];
  y1pencil_myrank[0] = x1pencil_myrank[1];
  y1pencil_myrank[1] = x1pencil_myrank[0];
  if(pencil == SDECOMP_Y1PENCIL){
    // return y1pencil info
    return type == TYPE_NPROCS ? y1pencil_nprocs[dir] : y1pencil_myrank[dir];
  }
  return -1;
#undef SDECOMP_INTERNAL_NDIMS
}

static int get_process_config_3d(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const type_t type){
#define SDECOMP_INTERNAL_NDIMS 3
  // compute
  //   type == TYPE_NPROCS: number of processes
  // or
  //   type == TYPE_MYRANK: my location
  // of the given decomposition "info"
  //   in the given dimension "dim"
  // x1 pencil, get info using MPI_Cart_get
  int x1pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int         periods[SDECOMP_INTERNAL_NDIMS] = {0};
  int x1pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  MPI_Cart_get(
      sdecomp.get_comm_cart(info),
      SDECOMP_INTERNAL_NDIMS,
      x1pencil_nprocs,
      periods,
      x1pencil_myrank
  );
  if(pencil == SDECOMP_X1PENCIL){
    // return x1pencil info
    return type == TYPE_NPROCS ? x1pencil_nprocs[dir] : x1pencil_myrank[dir];
  }
  // y1 pencil, determined by x1 pencil
  int y1pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int y1pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  y1pencil_nprocs[0] = x1pencil_nprocs[1];
  y1pencil_nprocs[1] = x1pencil_nprocs[0];
  y1pencil_nprocs[2] = x1pencil_nprocs[2];
  y1pencil_myrank[0] = x1pencil_myrank[1];
  y1pencil_myrank[1] = x1pencil_myrank[0];
  y1pencil_myrank[2] = x1pencil_myrank[2];
  if(pencil == SDECOMP_Y1PENCIL){
    // return y1pencil info
    return type == TYPE_NPROCS ? y1pencil_nprocs[dir] : y1pencil_myrank[dir];
  }
  // z1 pencil, determined by y1 pencil
  int z1pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int z1pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  z1pencil_nprocs[0] = y1pencil_nprocs[0];
  z1pencil_nprocs[1] = y1pencil_nprocs[2];
  z1pencil_nprocs[2] = y1pencil_nprocs[1];
  z1pencil_myrank[0] = y1pencil_myrank[0];
  z1pencil_myrank[1] = y1pencil_myrank[2];
  z1pencil_myrank[2] = y1pencil_myrank[1];
  if(pencil == SDECOMP_Z1PENCIL){
    // return z1pencil info
    return type == TYPE_NPROCS ? z1pencil_nprocs[dir] : z1pencil_myrank[dir];
  }
  // x2 pencil, determined by z1 pencil
  int x2pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int x2pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  x2pencil_nprocs[0] = z1pencil_nprocs[2];
  x2pencil_nprocs[1] = z1pencil_nprocs[1];
  x2pencil_nprocs[2] = z1pencil_nprocs[0];
  x2pencil_myrank[0] = z1pencil_myrank[2];
  x2pencil_myrank[1] = z1pencil_myrank[1];
  x2pencil_myrank[2] = z1pencil_myrank[0];
  if(pencil == SDECOMP_X2PENCIL){
    // return x2pencil info
    return type == TYPE_NPROCS ? x2pencil_nprocs[dir] : x2pencil_myrank[dir];
  }
  // y2 pencil, determined by x2 pencil
  int y2pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int y2pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  y2pencil_nprocs[0] = x2pencil_nprocs[1];
  y2pencil_nprocs[1] = x2pencil_nprocs[0];
  y2pencil_nprocs[2] = x2pencil_nprocs[2];
  y2pencil_myrank[0] = x2pencil_myrank[1];
  y2pencil_myrank[1] = x2pencil_myrank[0];
  y2pencil_myrank[2] = x2pencil_myrank[2];
  if(pencil == SDECOMP_Y2PENCIL){
    // return y2pencil info
    return type == TYPE_NPROCS ? y2pencil_nprocs[dir] : y2pencil_myrank[dir];
  }
  // z2 pencil, determined by y2 pencil
  int z2pencil_nprocs[SDECOMP_INTERNAL_NDIMS] = {0};
  int z2pencil_myrank[SDECOMP_INTERNAL_NDIMS] = {0};
  z2pencil_nprocs[0] = y2pencil_nprocs[0];
  z2pencil_nprocs[1] = y2pencil_nprocs[2];
  z2pencil_nprocs[2] = y2pencil_nprocs[1];
  z2pencil_myrank[0] = y2pencil_myrank[0];
  z2pencil_myrank[1] = y2pencil_myrank[2];
  z2pencil_myrank[2] = y2pencil_myrank[1];
  if(pencil == SDECOMP_Z2PENCIL){
    // return z2pencil info
    return type == TYPE_NPROCS ? z2pencil_nprocs[dir] : z2pencil_myrank[dir];
  }
  return -1;
#undef SDECOMP_INTERNAL_NDIMS
}

/**
 * @brief get number of processes in the given dimension
 * @param[in] info   : struct containing information of process distribution
 * @param[in] pencil : type of pencil (e.g., SDECOMP_X1PENCIL)
 * @param[in] dir    : direction which I am interested in
 * @return           : (success) number of processes in the given dimension
 *                     (failure) errorcode -1
 */
int sdecomp_internal_get_nprocs(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir){
  const size_t ndims = info->ndims;
  if(2 == ndims){
    return get_process_config_2d(info, pencil, dir, TYPE_NPROCS);
  }else{
    return get_process_config_3d(info, pencil, dir, TYPE_NPROCS);
  }
}

/**
 * @brief get my process position in the whole domain
 * @param[in] info   : struct containing information of process distribution
 * @param[in] pencil : type of pencil (e.g., SDECOMP_X1PENCIL)
 * @param[in] dir    : direction which I am interested in
 * @return           : (success) my position in the given dimension
 *                     (failure) errorcode -1
 */
int sdecomp_internal_get_myrank(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir){
  const size_t ndims = info->ndims;
  if(2 == ndims){
    return get_process_config_2d(info, pencil, dir, TYPE_MYRANK);
  }else{
    return get_process_config_3d(info, pencil, dir, TYPE_MYRANK);
  }
}

/**
 * @brief wrapper function to get pencil size or offset
 * @param[in] info   : struct containing information of process distribution
 * @param[in] pencil : type of pencil (e.g., SDECOMP_X1PENCIL)
 * @param[in] dir    : direction which I am interested in
 * @param[in] nitems : number of total grid points in the dimension
 * @param[in] kernel : function pointer to kernel functions,
 *                       one of "kernel_get_mysize" or "kernel_get_offset"
 * @return           : (success) number of local grid points in the dimension
 *                     (failure) errorcode -1
 */
static int get_pencil_config(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const int nitems, int (*kernel)(const int nitems_total, const int nprocs, const int myrank)){
  const size_t ndims = info->ndims;
  // sanitise inputs
  if(pencil_is_invalid(ndims, pencil)){
    return -1;
  }
  if(dir_is_invalid(ndims, dir)){
    return -1;
  }
  // compute number of process in the given direction
  const int nprocs = sdecomp_internal_get_nprocs(info, pencil, dir);
  // compute my location       in the given direction
  const int myrank = sdecomp_internal_get_myrank(info, pencil, dir);
  // call one of "number of grid calculator" and "offset calculator"
  return kernel(nitems, nprocs, myrank);
}

/**
 * @brief get number of grid points of the given pencil in the given direction
 * @param[in] info   : struct containing information of process distribution
 * @param[in] pencil : type of pencil (e.g., SDECOMP_X1PENCIL)
 * @param[in] dir    : direction which I am interested in
 * @param[in] nitems : number of total grid points in the direction
 * @return           : (success) number of local grid points in the direction
 *                     (failure) errorcode -1
 */
int sdecomp_internal_get_pencil_mysize(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const int nitems){
  return get_pencil_config(info, pencil, dir, nitems, sdecomp_internal_kernel_get_mysize);
}

/**
 * @brief get offset of grid points of the given pencil in the given direction
 * @param[in] info   : struct containing information of process distribution
 * @param[in] pencil : type of pencil (e.g., SDECOMP_X1PENCIL)
 * @param[in] dir    : direction which I am interested in
 * @param[in] nitems : number of total grid points in the direction
 * @return           : (success) offset in the give direction
 *                     (failure) errorcode -1
 */
int sdecomp_internal_get_pencil_offset(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const int nitems){
  return get_pencil_config(info, pencil, dir, nitems, sdecomp_internal_kernel_get_offset);
}

