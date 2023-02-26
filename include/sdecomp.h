#if !defined(SDECOMP_H)
#define SDECOMP_H

/*** public structures and APIs ***/

#include <stddef.h>  // size_t
#include <stdbool.h> // bool
#include <mpi.h>     // MPI_Comm, MPI_Datatype

// struct storing domain decomposition information
typedef struct sdecomp_info_t_ sdecomp_info_t;
// struct storing pencil transpose plan
typedef struct sdecomp_transpose_plan_t_ sdecomp_transpose_plan_t;

// directions
typedef enum {
  SDECOMP_XDIR = 0,
  SDECOMP_YDIR = 1,
  SDECOMP_ZDIR = 2,
} sdecomp_dir_t;

// pencil types
typedef enum {
  SDECOMP_X1PENCIL = 0,
  SDECOMP_Y1PENCIL = 1,
  SDECOMP_Z1PENCIL = 2,
  SDECOMP_X2PENCIL = 3,
  SDECOMP_Y2PENCIL = 4,
  SDECOMP_Z2PENCIL = 5,
} sdecomp_pencil_t;

/* APIs of sdecomp_transpose_t */
// accessed by sdecomp.transpose.xxx
typedef struct {
  // constructor
  sdecomp_transpose_plan_t * (* const construct)(
      const sdecomp_info_t *info,
      const bool is_forward,
      const sdecomp_pencil_t pencil,
      const size_t *glsizes,
      const size_t size_of_element,
      const MPI_Datatype mpi_datatype
  );
  // executor
  int (* const execute)(
      sdecomp_transpose_plan_t * restrict plan,
      const void * restrict sendbuf,
      void * restrict recvbuf
  );
  // destructor
  int (* const destruct)(
      sdecomp_transpose_plan_t *plan
  );
  // tester
  int (* const test)(
      sdecomp_transpose_plan_t *plan
  );
} sdecomp_transpose_t;

/* APIs of sdecomp_t */
// accessed by sdecomp.xxx
typedef struct {
  // constructor
  sdecomp_info_t * (* const construct)(
      const MPI_Comm comm_default,
      const size_t ndims,
      const size_t *dims,
      const bool *periods
  );
  // destructor
  int (* const destruct)(
      sdecomp_info_t *sdecomp
  );
  // getter, get number of processes in one dimension
  int (* const get_nprocs)(
      const sdecomp_info_t *info,
      const sdecomp_pencil_t pencil,
      const sdecomp_dir_t dir
  );
  // getter, get position of my process
  int (* const get_myrank)(
      const sdecomp_info_t *info,
      const sdecomp_pencil_t pencil,
      const sdecomp_dir_t dir
  );
  // getter, get local size of my pencil
  int (* const get_pencil_mysize)(
      const sdecomp_info_t *info,
      const sdecomp_pencil_t pencil,
      const sdecomp_dir_t dir,
      const int nitems
  );
  // getter, get offset of my pencil
  int (* const get_pencil_offset)(
      const sdecomp_info_t *info,
      const sdecomp_pencil_t pencil,
      const sdecomp_dir_t dir,
      const int nitems
  );
  // getter, get default communicator
  MPI_Comm (* const get_comm_cart)(
      const sdecomp_info_t *info
  );
  // transpose functions
  const sdecomp_transpose_t transpose;
} sdecomp_t;

extern const sdecomp_t sdecomp;

#endif // SDECOMP_H
