
def write_mtx_file(filename, sample_name, FTage, FTage_error, TL, NS, NI,
                    zeta, rhod):

    f = open(filename, "w")
    f.write("{name:s}\n".format(name=sample_name))
    f.write("{value:s}\n".format(value=str(-999)))
    f.write("{nconstraints:d} {ntl:d} {nc:d} {zeta:5.1f} {rhod:12.1f} {totco:d}\n".format(
             nconstraints=0, ntl=len(TL), nc=NS.size, zeta=zeta, rhod=rhod,
            totco=2000))
    f.write("{age:5.1f} {age_error:5.1f}\n".format(age=FTage,
                                                   age_error=FTage_error))
    TLmean = (float(sum(TL))/len(TL) if len(TL) > 0 else float('nan'))
    TLmean_sd = np.std(TL)

    f.write("{mtl:5.1f} {mtl_error:5.1f}\n".format(mtl=TLmean,
                                                   mtl_error=TLmean*0.05))
    f.write("{mtl_std:5.1f} {mtl_std_error:5.1f}\n".format(mtl_std=TLmean_sd,
                                                           mtl_std_error=TLmean_sd*0.05))
    for i in range(NS.size):
        f.write("{ns:d} {ni:d}\n".format(ns=NS[i], ni=NI[i]))

    for track in TL:
        f.write("{tl:4.1f}\n".format(tl=track))

    f.close()
    return 0
