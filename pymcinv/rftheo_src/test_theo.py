import theo

tvs 	= np.zeros(100)
tvpvs 	= np.zeros(100)
tqs 	= np.zeros(100)
tqp 	= np.zeros(100)
trho 	= np.zeros(100)
tthick 	= np.zeros(100)


rx 			= theo.theo(newnlayer, tvs, tthick, tvpvs, tqp, tqs, rt, din, 2.5, 0.005, 0, nn)