#include "utils.h"
#include "utils2.h"
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
  if (times < 2) {
    printf("param needs to be at least 2\n");
    return 2;
  } 
  for (i=0;i<times/2;++i)
    printhello();
  for (i=times/2;i<times;++i)
    printgoodbye();
  return 0;
}
