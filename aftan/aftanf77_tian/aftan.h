#ifndef AFTAN_H
#define AFTAN_H

/* Finction prorotypes */

#define NMAX 32768
#define NPER 500
extern "C" {
void aftanipg_(double *piover4,int *n,float *sei,double *t0,double *dt,
           double *delta,double *vmin,double *vmax,double *tmin,double *tmax,
           double *tresh,double *ffact,double *perc,int *npoints,
           double *taperl,int *nfin,double *snr,double *fmatch,
           int *npred, double pred[2][NPER], int *cuttype,int *nprpv,
           double prpvper[NPER],double prpvvel[NPER],float *seiout,
           int *nfout1,double arr1[100][8],int *nfout2,double arr2[100][7],
           double *tamp, int *nrow,int *ncol, double ampo[32][NMAX],int *ierr);
void aftanpg_(double *piover4,int *n,float *sei,double *t0,double *dt,
           double *delta,double *vmin,double *vmax,double *tmin,double *tmax,
           double *tresh,double *ffact,double *perc,int *npoints,
           double *taperl,int *nfin,double *snr,
           int *nprpv,double prpvper[NPER],double prpvvel[NPER],
           int *nfout1,double arr1[100][8],int *nfout2,double arr2[100][7],
           double *tamp, int *nrow,int *ncol, double ampo[32][NMAX],int *ierr);
void readdata(int sac,char *name,int *n,double *dt,double *delta,
              double *t0,float sei[NMAX]);
void filter4_(double *f1,double *f2,double *f3,double *f4,
              double *dt,int *n, float seis_in[],
              int *ns,double *dom,float amp_rec[]);
};

void swapn(unsigned char *b, int N, int nn);
void printres(double dt,int nfout1,double arr1[100][8],int nfout2,
           double arr2[100][7],double tamp, int nrow,int ncol,
           double ampo[32][NMAX],int ierr, char *name,char *pref,double delta);

#endif /* !AFTAN_H */
//void gaufilt_(double *alpha,double *c_per,
//              double *dt,int *n, float seis_in[], float seis_out[]);

