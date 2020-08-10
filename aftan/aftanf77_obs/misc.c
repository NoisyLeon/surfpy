/*
 * Read SAC header and seismogram and make output
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include "aftan.h"
#include "mysac64.h"

void readdata(int sac,char *name,int *n,double *dt,double *delta,
              double *t0,float *sei)
{
int   i,nn, iswap, *ih, *nsam;
float hed[158];       // the SAC header
float dumm,dum[2];
int   i1=0,i2=0,idelta=0;
char *p;
FILE  *fd;

      for(i=0; i < 6000; i++) sei[i]=0.0f;
      iswap = 0;
      if((fd = fopen(name,"r")) == NULL) {
          printf("Can not find file %s.\n",name);
          exit(1);
      }
   if(sac) {
//       The header
      fread(hed,sizeof(float),158,fd);
      ih = (int *)(&hed[76]);
      if(*ih > 100 || *ih < -100) iswap = 1;
      if(iswap) swapn((unsigned char *)hed,(int)(sizeof(float)),158);
      *dt = hed[0];
      *delta = hed[50];
      nsam = (int *)(&hed[79]);
      *t0 = hed[5];
//       The body
      fread(sei,sizeof(float),*nsam,fd);
      if(iswap) swapn((unsigned char *)sei,(int)(sizeof(float)),*nsam);
      *n = *nsam;
   } else {
/*
 * Read test data from ascii file
 */
      i = 0;
      while((nn = fscanf(fd,"%f %f",&dumm,&sei[i])) == 2) {
          if(i < 2) dum[i] = dumm; i++;
      }
// extract delta from file name
      p = strrchr(name,'/');
      sscanf(p+1,"proba_%d_%d_%d.SAC",&i1,&i2,&idelta);
      *delta = idelta;
      *dt = dum[1]-dum[0];
      *n = i;
      *t0 = 0.0;
   }
      fclose(fd);

// seismogram is ready

      printf("Dist= %lf, Dt= %lf, Nsamples= %d\n",*delta,*dt,*n);
}

SAC_HD *read_sac (char *fname, float **sig, SAC_HD *SHD) {
   FILE *fsac;
   if((fsac = fopen(fname, "rb"))==NULL) return NULL;
   //pthread_mutex_lock(&fiolock);
   if ( !SHD ) SHD = &SAC_HEADER;
   fread(SHD,sizeof(SAC_HD),1,fsac);
   if( *sig == NULL) *sig = (float *) malloc (SHD->npts * sizeof(float));
   //else *sig = (float *) realloc (*sig, SHD->npts * sizeof(float));
   fread(*sig,sizeof(float),SHD->npts,fsac);
   fclose (fsac);
   //pthread_mutex_unlock(&fiolock);

   /*-------------  calcule de t0  ----------------*/
   {
        int eh, em ,i;
        float fes;
        char koo[9];

        for ( i = 0; i < 8; i++ ) koo[i] = SHD->ko[i];
        koo[8] = 0;

        SHD->o = SHD->b + SHD->nzhour*3600. + SHD->nzmin*60 +
         SHD->nzsec + SHD->nzmsec*.001;

        sscanf(koo,"%d%*[^0123456789]%d%*[^.0123456789]%g",&eh,&em,&fes);

        SHD->o  -= (eh*3600. + em*60. + fes);
   /*-------------------------------------------*/}
   return SHD;
}

void write_sac (char *fname, float *sig, SAC_HD *SHD) {
   FILE *fsac;
   if( (fsac = fopen(fname, "wb"))==NULL ) {
      fprintf(stderr, "ERROR(write_sac): Cannot open file %s\n", fname);
      return;
   }

   if ( !SHD ) SHD = &SAC_HEADER;
   SHD->iftype = (int)ITIME;
   SHD->leven = (int)TRUE;
   SHD->lovrok = (int)TRUE;
   SHD->internal4 = 6L;

  /*+++++++++++++++++++++++++++++++++++++++++*/
   SHD->depmin = sig[0];
   SHD->depmax = sig[0];
   int i;
   for ( i = 0; i < SHD->npts ; i++ )
   {
    if ( SHD->depmin > sig[i] ) SHD->depmin = sig[i];
    else if ( SHD->depmax < sig[i] ) SHD->depmax = sig[i];
   }

//memset(SHD, 0, sizeof(SAC_HD));
	 //pthread_mutex_lock(&fiolock);
         fwrite(SHD,sizeof(SAC_HD),1,fsac);
         fwrite(sig,sizeof(float),SHD->npts,fsac);
	 //pthread_mutex_unlock(&fiolock);

        fclose (fsac);
}

extern int pflag;
void printres(double dt,int nfout1,double arr1[100][8],int nfout2,
              double arr2[100][7],double tamp, int nrow,int ncol,
              double ampo[32][NMAX],int ierr, char *name,char *pref,double delta)
{
	int i,j;
	FILE *out;
	char name2[160];
	if(ierr  ==  0 ) {
		printf("REESULT: O.K.\n");
	} else if(ierr == 1) {
		printf("RESULT: SOME PROBLEMS\n");
	} else {
		printf("RESULT: NO FINAL RESULT\n");
	}
	//printf("Start:\n");
	//for(i = 0; i < nfout1; i++) {
	//    printf("%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf %8.3lf\n",
	//                i+1,arr1[i][0],arr1[i][1],arr1[i][2],arr1[i][3],
	//                    arr1[i][4],arr1[i][5], arr1[i][6], arr1[i][7]);
	//}
	//if(nfout2 != 0) {
	//printf("Final:\n");
	//    for(i = 0; i < nfout2; i++) {
	//        printf("%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf %8.3lf\n",
	//             i+1,arr2[i][0],arr2[i][1],arr2[i][2],arr2[i][3],
	//                 arr2[i][4], arr2[i][5], arr2[i][6]);
	//    }
	//}
	//  write results to hard drive
	// file ...DISP.0 contains preliminary result
	if( pflag < 2 ) {
		strcpy(name2,name);
		strcat(name2,pref);
		strcat(name2,"_DISP.0");
		if((out = fopen(name2,"w")) == NULL) {
			printf("Can not open file %s.\n",name2);
			exit(1);
		}
		for(i = 0; i < nfout1; i++) {
			fprintf(out,"%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf\n",
					i,arr1[i][0],arr1[i][1],arr1[i][2],arr1[i][3],
					arr1[i][4],arr1[i][5],arr1[i][6]);
		}
		fclose(out);
	}
	// file ...DISP.1 includes final results
	if( pflag!=1 && nfout2!=0) {
		strcpy(name2,name);
		strcat(name2,pref);
		strcat(name2,"_DISP.1");
		if((out = fopen(name2,"w")) == NULL) {
			printf("Can not open file %s.\n",name2);
			exit(1);
		}
		for(i = 0; i < nfout2; i++) {
			fprintf(out,"%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf\n",
					i,arr2[i][0],arr2[i][1],arr2[i][2],arr2[i][3],
					arr2[i][4],arr2[i][5]);
		}
		fclose(out);
	}
	//*/ Output amplitude array into file on hard drive
	if( pflag==0 ) {
		strcpy(name2,name);
		strcat(name2,pref);
		strcat(name2,"_AMP");
		if((out = fopen(name2,"w")) == NULL) {
			printf("Can not open file %s.\n",name2);
			exit(1);
		}
		for(i = 0; i < nrow; ++i)
			for(j = 0; j < ncol; ++j) {
				fprintf(out,"%8.6lf %8.6lf %15.6e\n", arr1[i][1],delta/(tamp+j*dt),ampo[i][j]);
				//printf("%d %d\n",i,j);
			}
		fclose(out);
	}
}

