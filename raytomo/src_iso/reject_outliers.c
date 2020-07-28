#include <stdio.h>
#include <math.h>

char buff[300];
char buffd[300];

int main(int na, char *arg[])
{
  FILE *fd, *fr, *fo;
  float dt, nsigma;
  double sigma = 0.;
  int n;

  if ( na < 4 )
    {
      fprintf(stderr,"usage reject_outliers  residfile  nsigma\n");
      exit(1);
    }

  sscanf(arg[3],"%g", &nsigma );

  fr = fopen(arg[1],"r");

  for ( n = 0; ; n++ )
    {
      if ( !fgets(buff,300,fr) ) break;

      sscanf(buff,"%*g%*g%*g%*g%*g%*g%*g%g%", &dt);

      sigma += dt*dt;
    }

  fclose(fr);

  sigma = sqrt(sigma/(double)(n-1));


  /*fd = fopen(arg[1],"r");*/
  fr = fopen(arg[1],"r");
  fo = fopen(arg[2],"w");

  for (;;)
    {
      if ( !fgets(buff,300,fr) ) break;
      /*if ( !fgets(buffd,300,fd) ) break;*/

      sscanf(buff,"%*g%*g%*g%*g%*g%*g%*g%g%", &dt);

      if ( fabs(dt) < nsigma*sigma ) fputs(buff,fo);
    }

  /*fclose(fd);*/
  fclose(fr);
  fclose(fo);
}

