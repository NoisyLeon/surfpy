#include <stdio.h>
#include <math.h>

#define SLEN 400000

void Filter (double f1, double f2, double f3, double f4, double dt, int n, float *seis_in, float *seis_out);

int precur_noise(float fhlen, float *sei, int nsample, double dt, double dist, double b, double *c_per, double *g_vel, int nper, double *amp_max, double *pre_noise)
{
    // precursor noise level measurement
	double minT, maxT, precurT, signalmax, noiserms, num, e;
	//double fhlen=0.008;
	int k, i, ib, ie;
	float seis_out[SLEN];
	e=b+(nsample-1)*dt;
	for(k = 0; k < nper; k++) {
		// apply gaussian filter of freq-half-width = fhlen
		Filter(-1., 1./c_per[k], fhlen, -1, dt, nsample, sei, seis_out);
		// define signal window
		minT = dist/g_vel[k]-c_per[k]/2.;
		maxT = dist/g_vel[k]+c_per[k]/2.;
		if(minT<b) minT=b;
		if(maxT>e) maxT=e;
		ib = (int)floor(minT/dt);
		ie = (int)ceil(maxT/dt);
		// compute maximum signal
		signalmax=0;
		for(i=ib;i<ie;i++) {
			if(seis_out[i] < 0) num = seis_out[i]*(-1);
			else num = seis_out[i];
			if(num>signalmax)
				signalmax=num;
		}
		amp_max[k]=signalmax;
		// precursor noise window, from 0 to precurT
        precurT = dist/g_vel[k] - 3.*c_per[k];
        if (precurT>100.) precurT=100; // the precursory noise must be between 20 - 100 sec
        if(precurT<20) precurT=20;
        ib = (int)floor(precurT/dt);
        noiserms=0.;
        for(i=0;i<ib;i++) noiserms += seis_out[i] * seis_out[i];
        if (noiserms==0) {
            pre_noise[k]=-1.;
            continue;
        }
        noiserms=sqrt(noiserms/(ib-1.));
        pre_noise[k] = signalmax/noiserms;
	}
	return 1;
}

