# surfpy
A Python package for surface wave tomography and receiver function analysis.
This is an all-in-one package which includes several functions:
1. Data request for earthquake surface wave waveforms, teleseismic P wave (receiver function), continous waveforms (for ambient noise)
2. Ambient noise cross-correlation (two-station ambient noise interferometry)
3. Three-station ambient noise interferometry
4. Surface wave dispersion measurements (aftan)
5. Receiver function (rf) computation and harmonic stripping
6. Surface wave tomography (ray-based tomography and eikonal/Helmholtz tomography)
7. Bayesian Monte Carlo inversion(1. isotropic inversion using rf and Rayleigh wave; 2. VTI inversion using rf, Rayleigh waves and Love waves)
8. A least-square inversion for azimuthally anisotropic Vs model

If you use this package, please consider citing the following papers:

Feng, L. and Ritzwoller, M.H., 2017. The effect of sedimentary basins on surface waves that pass through them. Geophysical Journal International, 211(1), pp.572-592.

Feng, L. and Ritzwoller, M.H., 2019. A 3‐D shear velocity model of the crust and uppermost mantle beneath Alaska including apparent radial anisotropy. Journal of Geophysical Research: Solid Earth, 124(10), pp.10468-10497.

Ritzwoller, M.H., Feng, L.I.L.I., Nakata, N., Gualtieri, L. and Fichtner, A., 2019. Overview of pre-and post-processing of ambient noise correlations. Seismic Ambient Noise, pp.144-187.

Zhang, S., Feng, L. and Ritzwoller, M.H., 2020. Three-station interferometry and tomography: coda versus direct waves. Geophysical Journal International, 221(1), pp.521-541.

Feng, L., Liu, C. and Ritzwoller, M.H., 2020. Azimuthal anisotropy of the crust and uppermost mantle beneath Alaska. Journal of Geophysical Research: Solid Earth, 125(12), p.e2020JB020076.

Feng, L., 2021. High‐resolution crustal and uppermost mantle structure beneath central Mongolia from Rayleigh waves and receiver functions. Journal of Geophysical Research: Solid Earth, 126(4), p.e2020JB021161.

Feng, L. and Díaz, J., 2022. Azimuthal anisotropy of the westernmost Mediterranean: New constraints on lithospheric deformation and geodynamical evolution. Earth and Planetary Science Letters, 593, p.117689.
