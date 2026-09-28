"""Simulate simultaneous transits in multiplanet TESS systems."""

import os, sys
import tdpy

from tqdm import tqdm

import numpy as np

import ephesos


def main():

    typecomp = "psys"
    typedata = "toyy"
    typeinst = "tess"

    # typepoplstar = 'toii'
    typepoplstar = "tessnomi2min"

    if typedata == "toyy" and typepoplstar == "toii":
        raise Exception("")

    # get time axis
    listtsec = np.arange(1, 2)
    numbtsec = listtsec.size
    indxtsec = np.arange(numbtsec)
    pathtess = os.environ["TESS_DATA_PATH"] + "/data/lcur/"
    listtime = [[] for o in indxtsec]
    for o in indxtsec:
        pathtemp = pathtess + "sector-%02d/" % listtsec[o]
        listpath = os.listdir(pathtemp)
        path = pathtemp + listpath[0]
        arrylcur, _, _, _, _, _ = ephesos.read_tesskplr_file(
            path, typeinst="tess", strgtype="SAP_FLUX", boolmaskqual=False
        )
        listtime[o] = arrylcur[:, 0]
    time = np.concatenate(listtime)

    # get the target list
    if typepoplstar == "toii":
        typepoplstar = "toii"
        dictpoplcomp = ephesos.retr_dicttoii()
        dictpoplcomp["incl"] = np.zeros_like(dictpoplcomp["epoc"])
        dictpoplstar = ephesos.retr_dicthostplan("toii")
        dictpoplstar["incl"] = []
        for k in range(len(dictpoplstar["epoc"])):
            dictpoplstar["incl"].append(np.zeros_like(dictpoplstar["epoc"][k]))
    if typepoplstar == "tessnomi2min":
        numbsyst = 100
        timeepoc = np.mean(time)
        # get a synthetic population
        dictpopl, _, _ = ephesos.retr_dictpoplstarcomp(
            typepoplstar, typecomp, numbsyst=numbsyst, timeepoc=timeepoc
        )
        dictpoplcomp = dictpopl["comptran"]
        dictpoplstar = dictpopl["starcomp"]

        print("Number of components: %d" % len(dictpoplcomp["epoc"]))
        print("Number of stars: %d" % len(dictpoplstar["epoc"]))
    numbcomp = len(dictpoplcomp["epoc"])
    indxcomp = np.arange(numbcomp)

    print("Number of planets: %d" % numbcomp)

    print("Finding simultaneous transits")
    ## find stars with simultaneous transits
    indxcomppair = []
    for k in tqdm(range(numbcomp)):
        for l in indxcomp:
            if k >= l or dictpoplcomp["idenstar"][k] != dictpoplcomp["idenstar"][l]:
                continue

            duratotl = dictpoplcomp["duratran"]
            if not np.isfinite(duratotl[k]) or not np.isfinite(duratotl[l]):
                continue

            epoc = dictpoplcomp["epoc"]
            peri = dictpoplcomp["peri"]

            indxtimetranfrst = ephesos.retr_indxtimetran(time, epoc[k], peri[k], duratotl[k])
            indxtimetranseco = ephesos.retr_indxtimetran(time, epoc[l], peri[l], duratotl[l])
            indxtimetran = np.intersect1d(indxtimetranfrst, indxtimetranseco)
            if indxtimetran.size > 0:
                indxcomppair.append([k, l])
    numbsyst = 3
    numbtarg = len(indxcomppair)
    indxtarg = np.arange(numbtarg)

    if typedata == "toyy":
        # calculate photometric precision for the star population
        dictpoplcomp["nois"] = ephesos.retr_noistess(dictpoplcomp["tmag"])

    if typedata == "toyy" or typedata == "mock":
        for k in indxtarg:
            # generate light curve
            rflx = ephesos.eval_modl(
                time,
                dictpoplstar["radistar"][k],
                dictpoplstar["peri"][k],
                dictpoplstar["epoc"][k],
                inclcomp=dictpoplstar["incl"][k],
                massstar=dictpoplstar["massstar"][k],
                radicomp=dictpoplstar["radicomp"][k],
                masscomp=dictpoplstar["masscomp"][k],  # pathanim=pathmoon, strgextn=strgextn, \
            )


tdpy.dispatch_cli(globals())
