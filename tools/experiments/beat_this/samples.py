"""Reproducible generated pulse/drum samples, no external/user audio."""
import hashlib
import numpy as np


def generate(case):
    rate=48000;duration=18;signal=np.zeros(rate*duration,dtype=np.float64)
    period=.75 if case=='compound_6_8' else .5
    offset=.25 if case=='pickup_4_4' else .5
    group=2 if case=='compound_6_8' else 4
    pickup=3 if case=='pickup_4_4' else 0
    beats=np.arange(offset,duration-.5,period)
    downbeats=beats[(np.arange(len(beats))+pickup)%group==0]
    rng=np.random.default_rng(20260920)
    for i,t in enumerate(beats):
        length=int(.12*rate);x=np.arange(length)/rate
        strong=(i+pickup)%group==0
        drum=np.sin(2*np.pi*(100 if strong else 220)*x)*np.exp(-x*45)
        drum+=rng.normal(0,.08,length)*np.exp(-x*70)
        start=round(t*rate);signal[start:start+length]+=(.6 if strong else .35)*drum
        # A short pitched voice every pulse, with a repeating harmonic cycle.
        note=261.63*2**([0,4,7,4][i%4]/12)
        length=int(.28*rate);x=np.arange(length)/rate
        tone=.1*np.sin(2*np.pi*note*x)*np.exp(-x*12)
        signal[start:start+length]+=tone
    pcm=(np.clip(signal,-.95,.95)*32767).astype('<i2')
    stereo=np.repeat(pcm[:,None],2,axis=1).tobytes()
    return stereo,rate,duration*rate,beats,downbeats
