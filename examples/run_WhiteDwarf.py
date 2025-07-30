import numpy as np
import os, sys

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import nicomedia
import ephesos
import pergamon
from tdpy import summgene
import tdpy


'''
Run ephesos to produce light curves of planets transiting white dwarfs
'''

# path of the folder for visuals
pathbase = os.environ['EPHESOS_DATA_PATH'] + '/'
pathpopl = pathbase + 'WDs/'

# fix the seed
np.random.seed(2)

# type of the population of systems
## TESS 2-min target list during the nominal mission
typepoplsyst = 'SyntheticPopulation'

# compound typesyst
typesyst = 'PlanetarySystem'

print('typesyst')
print(typesyst)

# number of systems
numbsyst = 3

minmtime = 0.
maxmtime = 2.

boolcalcdistcomp = False
# number of systems to visualize via ephesos
numbsystvisu = 2
minmpericomp = 0.4
minmnumbcompstar = 1
maxmnumbcompstar = 1
typesamporbtcomp = 'peri'

print('minmnumbcompstar')
print(minmnumbcompstar)
print('maxmnumbcompstar')
print(maxmnumbcompstar)
# get dictionaries for stars, companions, and moons
dictnico = nicomedia.retr_dictpoplstarcomp( \
                                            typesyst, \
                                            
                                            typepoplsyst, \
                                            typestar='wdwf', \
                                            typesamporbtcomp=typesamporbtcomp, \
                                            minmnumbcompstar=minmnumbcompstar, \
                                            maxmnumbcompstar=maxmnumbcompstar, \
                                            #minmradicomp=10., \
                                            minmmasscomp=10., \
                                            minmpericomp=minmpericomp, \
                                            maxmpericomp=2., \
                                            maxmcosicomp=0.1, \
                                            numbsyst=numbsyst, \
                                          )

# parse nicomedia output
dictpoplstar = dictnico['dictpopl']['star']
dictpoplcomp = dictnico['dictpopl']['comp']
indxcompstar = dictnico['dictindx']['comp']['star']


strgpoplstartotl = 'star_%s_All' % typepoplsyst
strgpoplcomptotl = 'compstar_%s_All' % typepoplsyst
strgpoplcomptran = 'compstar_%s_tran' % typepoplsyst

# indices of the systems
indxsyst = np.arange(numbsyst)



# visualize individual systems
pathvisu = pathpopl
os.system('mkdir -p %s' % pathvisu)

dictefesinpt = dict()
dictefesinpt['typelmdk'] = 'quad'
#dictefesinpt['typesyst'] = typesyst
#dictefesinpt['typenorm'] = 'edgeleft'
dictefesinpt['lablunittime'] = 'days'
dictefesinpt['booltqdm'] = False
dictefesinpt['typeverb'] = 0
#dictefesinpt['typelang'] = typelang
#dictefesinpt['typefileplot'] = typefileplot

#dictefesinpt['booldiag'] = False

dictefesinpt['boolcalcdistcomp'] = boolcalcdistcomp

#dictefesinpt['boolintp'] = boolintp
#dictefesinpt['boolwritover'] = boolwritover
#dictefesinpt['strgextn'] = strgextn
#dictefesinpt['strgtitl'] = strgtitl
#dictefesinpt['typecoor'] = typecoor

## cadence of simulation

print(dictpoplcomp.keys())
print(dictpoplcomp[strgpoplcomptotl].keys())

# minimum period in the ensemble
minmperi = 1e100
minmduratran = 1e100
for k in range(numbsystvisu):
    minmperi = min(np.amin(dictpoplcomp[strgpoplcomptotl]['pericomp'][0][k]), minmperi)
    minmduratran = min(np.amin(dictpoplcomp[strgpoplcomptotl]['duratrantotl'][0][k]), minmduratran)

print('minmduratran')
print(minmduratran)

# cadence
cade = 0.1 * minmduratran / 24.

if cade > 0.1 * minmduratran / 24.:
    print('')
    print('')
    print('')
    print('minmperi')
    print(minmperi)
    print('1e-3 * minmperi * 24. * 3600 [seconds]')
    print(1e-3 * 24 * 3600 * minmperi)
    facttime, lablunittime = tdpy.retr_timeunitdays(cade)
    raise Exception('Simulation cadence (%g %s) is too long, which will undersample the motion of the innermost companion and diminish animation quality.' % \
                                                                                                                                        (cade * facttime, lablunittime))

### duration of simulation
#if typesyst == 'PlanetarySystemWithPhaseCurve':
#    durasimu = dicttemp['pericomp']
#    if namebatc == 'longbase':
#        durasimu *= 3.
#else:
#    durasimu = 6. / 24. # days

#duratrantotl = nicomedia.retr_duratrantotl(dicttemp['pericomp'], dicttemp['rsmacomp'], dicttemp['cosicomp']) / 24. # [days]
#if typesyst == 'PlanetarySystemWithPhaseCurve':
#    # minimum time
#    minmtime = -0.25 * durasimu
#    # maximum time
#    maxmtime = 0.75 * durasimu
#else:
#    # minimum time
#    minmtime = -0.5 * 1.5 * duratrantotl
#    # maximum time
#    maxmtime = 0.5 * 1.5 * duratrantotl
# time axis
time = np.arange(minmtime, maxmtime, cade)

listnamevarb = ['peri', 'epocmtra', 'rsma', 'cosi', 'rrat']

from tqdm import tqdm

if boolcalcdistcomp:
    listnamefeatmult = ['rateppcr', 'numbppcr', 'minmcompdist']
    for name in listnamefeatmult:
        dictpoplstar[strgpoplstartotl][name] = np.empty(dictpoplstar[strgpoplstartotl]['radistar'].size)

print('Running ephesos on each system...')
for k in tqdm(range(numbsyst)):
    
    if k < numbsystvisu:
        dictefesinpt['boolmakeanimorbt'] = True
        dictefesinpt['boolmakeanimsbrt'] = True
        dictefesinpt['pathvisu'] = pathvisu
    else:
        dictefesinpt['boolmakeanimorbt'] = False
        dictefesinpt['boolmakeanimsbrt'] = False
        dictefesinpt['pathvisu'] = None
    
    if indxcompstar[k].size == 0:
        print('')
        print('')
        print('')
        raise Exception('indxcompstar[k].size == 0')
    
    if not np.isfinite(dictpoplcomp[strgpoplcomptotl]['rratcomp'][0]).all():
        print('')
        print('')
        print('')
        print('dictpoplcomp[strgpoplcomptotl][rratcomp][0]')
        summgene(dictpoplcomp[strgpoplcomptotl]['rratcomp'][0])
        raise Exception('not np.isfinite(dictpoplcomp[strgpoplcomptotl][rratcomp]).all()')

    for namevarb in listnamevarb:
        dictefesinpt['%scomp' % namevarb] = dictpoplcomp[strgpoplcomptotl]['%scomp' % namevarb][0][indxcompstar[k]]
    
    dictefesinpt['strgextn'] = '%s_Target%04d' % (typesyst, k)
    
    # generate light curve
    dictefesoutp = ephesos.eval_modl(time, typesyst, **dictefesinpt)

    if boolcalcdistcomp:
        for name in listnamefeatmult:
            if not np.isscalar(dictefesoutp[name]):
                print('name')
                print(name)
                print( dictefesoutp[name].shape)
                raise Exception('')
    
            dictpoplstar[strgpoplstartotl][name][k] = dictefesoutp[name]

