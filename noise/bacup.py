 

# def _merge_traces(st, tbtime, tetime, tlen):
#     """merge traces
#     """
#     st.sort(keys=['starttime', 'endtime'])
#     gaplst  = st.get_gaps()
#     if len(st) > 1:
#         tr      = st[0]
#         startt1 = tr.stats.starttime
#         endt1   = tr.stats.endtime
#         if endt1 < tetime:
#             tmptb = startt1 - tbtime
#             tmpte = endt1 - tbtime
#             if tmptb > 0.:
#                 Nreclst.append([int(np.ceil(tmptb/dt)), int(np.ceil(tmpte/dt))])
#             else:
#                 Nreclst.append([0, int(np.ceil(tmpte/dt))])
        
    
    
    
    
    
    # 
    # igap    = 0
    # Nreclst = []
    # jrec    = 1
    # dt      = st[0].stats.delta
    # if len(st) > 1:
    #     tr      = st[0]
    #     startt1 = tr.stats.starttime
    #     endt1   = tr.stats.endtime
    #     if endt1 < tetime:
    #         tmptb = startt1 - tbtime
    #         tmpte = endt1 - tbtime
    #         if tmptb > 0.:
    #             Nreclst.append([int(np.ceil(tmptb/dt)), int(np.ceil(tmpte/dt))])
    #         else:
    #             Nreclst.append([0, int(np.ceil(tmpte/dt))])
    #         for i in range(len(st)):
    #             if i == 0:
    #                 continue
    #             tmptr   = st[i]
    #             startt2 = tmptr.stats.starttime
    #             endt2   = tmptr.stats.endtime
    #             if startt1 > startt2:
    #                 raise xcorrHeaderError('Starttime not sorted!')
    #             if endt1 >= endt2: # igonore the shorter trace
    #                 continue
    #             if int((startt2-endt1)/dt) >= 2: # gap
    #                 # generate random data for filling
    #                 Nfill   = int((startt2-endt1)/dt) - 1
    #                 sigmean1= st[i-1].data.mean()
    #                 sigmean2= st[i].data.mean()
    #                 sigstd  = (np.append(st[i-1].data, st[i].data)).std()
    #                 tmprand = np.random.uniform(low = -sigstd, high = sigstd, size = Nfill)
    #                 if Nfill > 1:
    #                     tmpmean = np.arange(Nfill, dtype=np.float64) * (sigmean2 - sigmean1)/(Nfill-1) + sigmean1
    #                 else:
    #                     tmpmean = np.ones(Nfill, dtype=np.float64) *(sigmean1 + sigmean2)/2.
    #                 tmpdata     = tmprand + tmpmean
    #                 tr.data     = np.append(tr.data, tmpdata) # append random filled values, endtime also changed
    #                 # merge trace
    #                 tr          = tr.__add__(st[i],\
    #                                 method = 1, interpolation_samples=2, fill_value='interpolate')
    #                 tmptb   = startt2 - tbtime
    #                 tmpte   = endt2 - tbtime
    #                 # append the rec list
    #                 if tmpte >= tlen:
    #                     Nreclst.append([int(np.ceil(tmptb/dt)), int(tlen/dt)])
    #                 else:
    #                     Nreclst.append([int(np.ceil(tmptb/dt)), int(np.ceil(tmpte/dt))])
    #                 jrec    += 1
    #                 # check with gap list
    #                 
    #             else: # merge data directly
    #                 tr              = tr.__add__(st[i],\
    #                                     method = 1, interpolation_samples=2, fill_value='interpolate')
    #                 tmpte           = endt2 - tbtime
    #                 if tmpte > tlen: 
    #                     Nreclst[jrec-1][1]  = int(tlen/dt)
    #                 else:
    #                     Nreclst[jrec-1][1]  = int(np.ceil(tmpte/dt))
    #                 # check with gaplst
    #                 
    #     else:
    #         tmptb = startt1 - tbtime
    #         if tmptb > 0.:
    #             Nreclst.append([int(np.ceil(tmptb/dt)), int(tlen/dt)])
    #         else:
    #             Nreclst.append([0, int(tlen/dt)])
    # else:
    #     tr      = st[0]
    #     startt1 = tr.stats.starttime
    #     endt1   = tr.stats.endtime
    #     tmptb   = startt1 - tbtime
    #     tmpte   = endt1 - tbtime
    #     if tmptb > 0.:
    #         Ntmpb   = int(np.ceil(tmptb/dt))
    #     else:
    #         Ntmpb   = 0
    #     if tmpte > tlen:
    #         Ntmpe   = int(tlen/dt)
    #     else:
    #         Ntmpe   = int(np.ceil(tmpte/dt))
    #     Nreclst.append([Ntmpb, Ntmpe])
    # return tr, Nreclst

# def _merge_nreclst(Nreclst1, Nreclst2):
#     Nreclst = []
#     irec1   = 0
#     irec2   = 0
#     while(irec1 < len(Nreclst1) or irec2 < len(Nreclst2)):
#         tmprec1 = Nreclst1[irec1]
#         tmprec2 = Nreclst2[irec2]