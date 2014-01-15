#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{
  int times, i;
  if (argc < 2) {
    printf("need at least one parameter, the number of times to print\n");
    return 1;
  }
  times = atoi(argv[1]);
  for (i=0;i<times;++i)
    printhello();
  return 0;
}
