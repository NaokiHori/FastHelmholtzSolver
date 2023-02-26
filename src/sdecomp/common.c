// Utility functions which are internally used
//   1. file IOs of log files
//   2. memory managements

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>
#include "sdecomp.h"
#include "internal.h"


FILE *sdecomp_internal_fopen(void){
  const char log_filename[] = {"sdecomp.log"};
  const char mode[] = {"a"};
  errno = 0;
  FILE *stream = fopen(log_filename, mode);
  if(NULL == stream){
    perror(log_filename);
    return NULL;
  }
  return stream;
}

int sdecomp_internal_fclose(FILE *stream){
  if(NULL == stream){
    return 1;
  }
  fclose(stream);
  return 0;
}

void *sdecomp_internal_calloc(const size_t count, const size_t size){
  errno = 0;
  void *ptr = calloc(count, size);
  if(NULL == ptr){
    const char *message = strerror(errno);
    FILE *stream = sdecomp_internal_fopen();
    fprintf(stream, "%s", message);
    sdecomp_internal_fclose(stream);
    // fatal, abort
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  return ptr;
}

void sdecomp_internal_free(void *ptr){
  free(ptr);
}

