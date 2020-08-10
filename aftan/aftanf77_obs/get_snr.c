#include <stdio.h>
#include <math.h>

#define SLEN 400000

void Filter (double f1, double f2, double f3, double f4, double dt, int n, float *seis_in, float *seis_out);

int get_snr(float fhlen, float *sei, int nsample, double dt, double dist, double b, double *c_per, double *g_vel, int nper, double *amp_max, double *snr2)
{
	double minT, maxT, signalmax, noiserms, num, e;
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
		// define noise window ( to at least 50 sec and at most 1000 sec )
		minT = maxT + c_per[k] * 5 + 500.; // move out of the signal
		if( (e - minT) < 50. ) { // no enough noise for computing rms
			snr2[k] = -1.;
			continue;
		}
		else if( (e - minT) < 1100. ) { maxT = e - 10.; }
		else {
			minT = e - 1100.;
			maxT = e - 100.;
		}
		// compute rms noise
		ib = (int)floor(minT/dt);
		ie = (int)ceil(maxT/dt);
		noiserms=0.;
		for(i=ib;i<ie;i++) noiserms += seis_out[i] * seis_out[i];
		noiserms=sqrt(noiserms/(ie-ib-1.));
		snr2[k]=signalmax/noiserms;
	}
	return 1;
}

