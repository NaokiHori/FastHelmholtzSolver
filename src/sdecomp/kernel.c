#include <stdio.h>
#include "internal.h"


static int sanitise(const int nitems_total, const int nprocs, const int myrank){
  if(nprocs <= 0){
    FILE *fp = sdecomp_internal_fopen();
    fprintf(fp, "ERROR: total number of processes is non-positive (%d)\n", nprocs);
    sdecomp_internal_fclose(fp);
    return 1;
  }
  if(myrank < 0){
    FILE *fp = sdecomp_internal_fopen();
    fprintf(fp, "ERROR: MPI rank is negative (%d)\n", myrank);
    sdecomp_internal_fclose(fp);
    return 1;
  }
  if(nitems_total < nprocs){
    FILE *fp = sdecomp_internal_fopen();
    fprintf(fp, "ERROR: trying to decompose %d items by %d processes\n", nitems_total, nprocs);
    sdecomp_internal_fclose(fp);
    return 1;
  }
  return 0;
}

/* compute number of grid points taken care of by the process */
int sdecomp_internal_kernel_get_mysize(const int nitems_total, const int nprocs, const int myrank){
  if(0 != sanitise(nitems_total, nprocs, myrank)){
    return -1;
  }
  // example: nitems_total: 10, nprocs: 3 (3 processes in total)
  // myrank = 0 -> nitems_local = 3
  // myrank = 1 -> nitems_local = 3
  // myrank = 2 -> nitems_local = 4
  // NOTE: sum of "nitems_local"s is "nitems_total"
  const int nitems_local = (nitems_total + myrank) / nprocs;
  return nitems_local;
}

/* compute offset of the grid point taken care of by the process */
int sdecomp_internal_kernel_get_offset(const int nitems_total, const int nprocs, const int myrank){
  if(0 != sanitise(nitems_total, nprocs, myrank)){
    return -1;
  }
  // example: nitems_total: 10, nprocs: 3 (3 processes in total)
  // myrank = 0 -> offset = 0
  // myrank = 1 -> offset = 3
  // myrank = 2 -> offset = 6
  int offset = 0;
  for(int i = 0; i < myrank; i++){
    offset += sdecomp_internal_kernel_get_mysize(nitems_total, nprocs, i);
  }
  return offset;
}

