import numpy as np
import json
import astropy.units as u
import os
import subprocess
import h5py
from multiprocessing import Pool

def convert_wavn(wavn):
    wav = (1/wavn)*u.cm.to(u.um)
    return wav

def extract_fluxes(smartoutput):

    fout = open(smartoutput, 'r')
    lines = fout.readlines()

    wavelength = []
    d = {}

    for i, l in enumerate(lines):

        hold = l.split()
        if 'wng0' in hold:
            curr_wav = convert_wavn(float(hold[len(hold)-1]))
            wavelength.append(curr_wav)
            d[curr_wav] = {}

        elif 'Flup:' in hold:
            flup = []
            for add in range(1,15):
                curr = lines[i+add].split('\n')[0]
                curr = curr.split()
                for val in curr:
                    flup.append(float(val))

            for neg in range(len(flup)):
                if flup[neg] < 0:
                    flup[neg] = 0
            
            d[curr_wav]['Flup'] = flup
        
        elif 'rfldir:' in hold:
            rfldir = []
            for add in range(1,15):
                curr = lines[i+add].split('\n')[0]
                curr = curr.split()
                for val in curr:
                    rfldir.append(float(val))
            
            d[curr_wav]['rfldir'] = rfldir

        elif 'rfldn:' in hold:
            rfldn = []
            for add in range(1,15):
                curr = lines[i+add].split('\n')[0]
                curr = curr.split()
                for val in curr:
                    rfldn.append(float(val))

            for neg in range(len(rfldn)):
                if rfldn[neg] < 0:
                    #rfldn[neg] = rfldn[neg]*-1
                    li = neg-1
                    while rfldn[li] < 0:
                        li = li-1
                    hi = neg+1
                    while rfldn[hi] < 0:
                        hi = hi + 1

                    rfldn[neg] = np.average([rfldn[li], rfldn[hi]])
            
            d[curr_wav]['rfldn'] = rfldn

        elif 'dtauc:' in hold:
            dtauc = []
            for add in range(1,15):
                curr = lines[i+add].split('\n')[0]
                curr = curr.split()
                for val in curr:
                    dtauc.append(float(val))
            
            d[curr_wav]['dtauc'] = dtauc
        
        '''
        elif 'ssalb:' in hold:
            ssalb = []
            for add in range(1,15):
                curr = lines[i+add].split('\n')[0]
                curr = curr.split()
                for val in curr:
                    ssalb.append(float(val))
            
            d[curr_wav]['ssalb'] = ssalb
        '''

    d['Wavelength'] = wavelength

    for i in d.keys():
        if i != 'Wavelength':
            for k in d[i].keys():
                if k in ['dtauc']:
                    del d[i][k][49:70]
                else:   
                    del d[i][k][50:70]
    return d

def create_initial_h5_file(smartoutput, water_multiplier=1, outputname='DISORT_Training_Data.h5'):

    # Return the data required
    smartout = extract_fluxes(smartoutput)

    # Read in H2O VMR
    h2o_vmr_baseline = np.array([5.075e-08,
    5.734551020408172e-08,
    8.451469387755104e-08,
    1.408999999999997e-07,
    2.436510204081634e-07,
    4.182285714285707e-07,
    6.923795918367353e-07,
    1.0878571428571416e-06,
    1.6125306122448994e-06,
    2.2459183673469373e-06,
    2.9286326530612235e-06,
    3.550612244897958e-06,
    4.051551020408163e-06,
    4.447836734693877e-06,
    4.765714285714285e-06,
    5.027102040816327e-06,
    5.247938775510204e-06,
    5.436755102040816e-06,
    5.599673469387756e-06,
    5.737897959183674e-06,
    5.851612244897959e-06,
    5.942428571428572e-06,
    6.008510204081632e-06,
    6.049448979591837e-06,
    6.0615306122448975e-06,
    6.03869387755102e-06,
    5.977163265306122e-06,
    5.870102040816327e-06,
    5.718e-06,
    5.524448979591837e-06,
    5.295632653061225e-06,
    5.040632653061225e-06,
    4.765816326530612e-06,
    4.480551020408163e-06,
    4.193204081632653e-06,
    3.9114285714285715e-06,
    3.64634693877551e-06,
    3.403142857142857e-06,
    3.184673469387755e-06,
    2.994326530612245e-06,
    2.8556734693877554e-06,
    2.806448979591837e-06,
    6.35557142857143e-06,
    1.6490408163265315e-05,
    5.1510204081632635e-05,
    0.00023100408163265307,
    0.0009479673469387759,
    0.0029970408163265305,
    0.007079,
    0.01547])
    h2o_vmr = h2o_vmr_baseline * water_multiplier
    
    # initialize training input array for wavelength, tau values 
    tr_input = np.zeros((len(smartout['Wavelength']), 50))

    # Initialize training output array for flup, rfldir, rfldn, h2o vmr
    tr_output = np.zeros((len(smartout['Wavelength']), 50*4))

    # Set training inputs and outputs
    wav = np.sort(smartout['Wavelength'])
    for i in range(len(tr_input)):
        tr_input[i][0] = wav[i]
        tr_input[i,1:] = smartout[wav[i]]['dtauc']

        tr_output[i,0:50] = smartout[wav[i]]['Flup']
        tr_output[i,50:100] = smartout[wav[i]]['rfldir']
        tr_output[i,100:150] = smartout[wav[i]]['rfldn']
        tr_output[i,150:200] = h2o_vmr

    with h5py.File('/gscratch/vsm/gialluca/PostDocPropose/'+outputname, 'w') as hf:
        hf.create_dataset("IN", data=tr_input, maxshape=(None, 50), chunks=True)
        hf.create_dataset("OUT", data=tr_output, maxshape=(None, 50*4), chunks=True)

def append_to_h5_file(smartoutput, water_multiplier=1, outputname='DISORT_Training_Data.h5'):

    # Return the data required
    smartout = extract_fluxes(smartoutput)

    # Read in H2O VMR
    h2o_vmr_baseline = np.array([5.075e-08,
    5.734551020408172e-08,
    8.451469387755104e-08,
    1.408999999999997e-07,
    2.436510204081634e-07,
    4.182285714285707e-07,
    6.923795918367353e-07,
    1.0878571428571416e-06,
    1.6125306122448994e-06,
    2.2459183673469373e-06,
    2.9286326530612235e-06,
    3.550612244897958e-06,
    4.051551020408163e-06,
    4.447836734693877e-06,
    4.765714285714285e-06,
    5.027102040816327e-06,
    5.247938775510204e-06,
    5.436755102040816e-06,
    5.599673469387756e-06,
    5.737897959183674e-06,
    5.851612244897959e-06,
    5.942428571428572e-06,
    6.008510204081632e-06,
    6.049448979591837e-06,
    6.0615306122448975e-06,
    6.03869387755102e-06,
    5.977163265306122e-06,
    5.870102040816327e-06,
    5.718e-06,
    5.524448979591837e-06,
    5.295632653061225e-06,
    5.040632653061225e-06,
    4.765816326530612e-06,
    4.480551020408163e-06,
    4.193204081632653e-06,
    3.9114285714285715e-06,
    3.64634693877551e-06,
    3.403142857142857e-06,
    3.184673469387755e-06,
    2.994326530612245e-06,
    2.8556734693877554e-06,
    2.806448979591837e-06,
    6.35557142857143e-06,
    1.6490408163265315e-05,
    5.1510204081632635e-05,
    0.00023100408163265307,
    0.0009479673469387759,
    0.0029970408163265305,
    0.007079,
    0.01547])
    h2o_vmr = h2o_vmr_baseline * water_multiplier
    
    # initialize training input array for wavelength, tau values 
    tr_input = np.zeros((len(smartout['Wavelength']), 50))

    # Initialize training output array for flup, rfldir, rfldn, h2o vmr
    tr_output = np.zeros((len(smartout['Wavelength']), 50*4))

    # Set training inputs and outputs
    wav = np.sort(smartout['Wavelength'])
    for i in range(len(tr_input)):
        tr_input[i][0] = wav[i]
        tr_input[i,1:] = smartout[wav[i]]['dtauc']

        tr_output[i,0:50] = smartout[wav[i]]['Flup']
        tr_output[i,50:100] = smartout[wav[i]]['rfldir']
        tr_output[i,100:150] = smartout[wav[i]]['rfldn']
        tr_output[i,150:200] = h2o_vmr

    # Resize and append to h5 file
    with h5py.File('/gscratch/vsm/gialluca/PostDocPropose/'+outputname, 'a') as hf:
        oldinsize = hf['IN'].shape[0]
        newinsize = oldinsize + tr_input.shape[0]
        hf['IN'].resize(newinsize, axis=0)
        hf['IN'][oldinsize:newinsize] = tr_input

        oldoutsize = hf['OUT'].shape[0]
        newoutsize = oldoutsize + tr_output.shape[0]
        hf['OUT'].resize(newoutsize, axis=0)
        hf['OUT'][oldoutsize:newoutsize] = tr_output

def run_smart_1instance(runscript, identifier):
        subprocess.run('/gscratch/vsm/gialluca/PostDocPropose/smart/smart_spectra < '+runscript+' > /gscratch/vsm/gialluca/PostDocPropose/outputs/'+identifier+'_output.run', shell=True)

def run_lblabc_1instance(runscript, identifier):
        f = open('/gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/'+identifier+'_lblabcout_H2O.run', 'w')
        subprocess.run('/gscratch/vsm/gialluca/VPLModelingTools_Dev/lblabc/lblabc < '+runscript, shell=True, stdout=f)
        f.close()

def make_lblabc_runscript(identifier, water_multiplier=1):

    # Read in the template
    temp = open('/gscratch/vsm/gialluca/PostDocPropose/defaults/RunLBLABC_H2O_Baseline.script', 'r')

    # Open new file to write to
    runscr = open('/gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/RunLBLABC_H2O_'+identifier+'.script', 'w')

    lines = temp.readlines()
    for l in lines:
        hold = l.split()

        # Multiply the rmix factor by the water factor
        if 'rmix' in hold and 'scaling' in hold:
            runscr.write('100000,'+str(float(water_multiplier))+'                              p, rmix scaling factors\n')
        
        # Rename script from default
        elif len(hold[0].split('lblabcfiles/')) > 1:
            runscr.write('/gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/H2O_'+identifier+'.abs\n')

        # Preserve all other settings
        else:
            runscr.write(l)

    temp.close()
    runscr.close()

def make_smart_runscript(identifier, water_multiplier=1):
    
    # Read in the template
    temp = open('/gscratch/vsm/gialluca/PostDocPropose/defaults/runsmart_FluxTesting.script', 'r')

    # Open new file to write to
    runscr = open('/gscratch/vsm/gialluca/PostDocPropose/outputs/runsmart_'+identifier+'.script', 'w')

    lines = temp.readlines()
    h2oblok = False
    for l in lines:
        hold = l.split()
        if h2oblok == True:

            # Replace the abs file being used
            if len(hold[0].split('H2O_FeatTest')) > 1:
                runscr.write('/gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/H2O_'+identifier+'.abs\n')
            
            # Replace the rmix scaling factor
            elif 'rMix' in hold and 'Scaling' in hold:
                runscr.write('100000.,'+str(float(water_multiplier))+'			P and rMix Scaling Factors\n')
                h2oblok = False
            
            else:
                runscr.write(l)

        elif len(hold) >= 3:
            if hold[0] == '1' and hold[1] == 'Gas' and hold[2] == 'Code':
                h2oblok = True
            else:
                runscr.write(l)

        elif len(hold[0].split('SMART')) > 1:
            runscr.write('/gscratch/vsm/gialluca/PostDocPropose/spectra/Earth_'+identifier+'_SMART\n')

        else:
            runscr.write(l)

    temp.close()
    runscr.close()

def run_one_model(inputs):

    # For parallelization with pool
    # Inputs need id and water multiplier
    identifier, water_multiplier = inputs

    # Create lblabc runscript
    make_lblabc_runscript(identifier, water_multiplier=water_multiplier)

    # Run LBLABC
    run_lblabc_1instance('/gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/RunLBLABC_H2O_'+identifier+'.script', identifier)

    # Make SMART runscript
    make_smart_runscript(identifier, water_multiplier=water_multiplier)

    # Run SMART
    run_smart_1instance('/gscratch/vsm/gialluca/PostDocPropose/outputs/runsmart_'+identifier+'.script', identifier)

    subprocess.run('rm /gscratch/vsm/gialluca/PostDocPropose/lblabcfiles/H2O_'+identifier+'.abs', shell=True)

    return ['/gscratch/vsm/gialluca/PostDocPropose/outputs/'+identifier+'_output.run', water_multiplier]



# Create Training Data

inputs = [['xp5', 0.5],
          ['x1', 1],
          ['x1p5', 1.5],
          ['x2', 2],
          ['x2p5', 2.5],
          ['x3', 3],
          ['x3p5', 3.5],
          ['x4', 4],
          ['x4p5', 4.5],
          ['x5', 5],
          ['x5.5', 5.5],
          ['x6', 6]]

with Pool() as p:
    models = p.map(run_one_model, inputs)

for i,m in enumerate(models):
    if i == 0:
        create_initial_h5_file(m[0], water_multiplier=m[1])
    else:
        append_to_h5_file(m[0], m[1])


# Create 3 testing cases 

inputs = [['xp7', 0.7],
          ['x3p2', 3.2],
          ['x5p8', 5.8]]

with Pool() as p:
    models = p.map(run_one_model, inputs)

create_initial_h5_file(models[0][0], water_multiplier=models[0][1], outputname='DISORT_Testing_Data.h5')
append_to_h5_file(models[1][0], water_multiplier=models[1][1], outputname='DISORT_Testing_Data.h5')
append_to_h5_file(models[2][0], water_multiplier=models[2][1], outputname='DISORT_Testing_Data.h5')
