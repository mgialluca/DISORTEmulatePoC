import numpy as np
import matplotlib.pyplot as plt 
import h5py
import pandas
from astropy.io import ascii
from degrade_spec_demo import *

pt = ascii.read('../FluxPrintOut_Dev/PT_profile_Baseline.pt')
p = list(pt['Press'])

disortdat = h5py.File('../DISORT_Testing_Data.h5', 'r')



def plot_test_fluxes(testind, watermultiplier):

    emulation = np.load('../Predictions_Test'+str(testind)+'.npy')

    # Get the inds in the h5 file where the profile changes:
    inds = []
    arr = np.array(disortdat['IN'])
    for i in range(len(arr)-1):
        if arr[i][0] > arr[i+1][0]:
            inds.append(i+1)
    
    if testind == 1:
        lowb = None
        upb = inds[0]
    elif testind == 2:
        lowb = inds[0]
        upb = inds[1]
    elif testind == 3:
        lowb = inds[1]
        upb = None

    # make pandas dfs for plotting, true answers
    wavelength = [w[0] for w in disortdat['IN'][lowb:upb]]
    flup_dat = pandas.DataFrame(index=p, columns=wavelength, data=disortdat['OUT'][lowb:upb,:50].transpose())
    rfldir_dat = pandas.DataFrame(index=p, columns=wavelength, data=disortdat['OUT'][lowb:upb,50:100].transpose())
    rfldn_dat = pandas.DataFrame(index=p, columns=wavelength, data=disortdat['OUT'][lowb:upb,100:150].transpose())

    # Predictions from emulator    
    flup_pred = pandas.DataFrame(index=p, columns=wavelength, data=emulation[:,:50].transpose())
    rfldir_pred = pandas.DataFrame(index=p, columns=wavelength, data=emulation[:,50:100].transpose())
    rfldn_pred = pandas.DataFrame(index=p, columns=wavelength, data=emulation[:,100:150].transpose())

    # Plotting
    fig = plt.figure(figsize=(18,18))
    up = fig.add_subplot(331)
    fdir = fig.add_subplot(332)
    fdn = fig.add_subplot(333)
    uppred = fig.add_subplot(334)
    fdirpred = fig.add_subplot(335)
    fdnpred = fig.add_subplot(336)
    updif = fig.add_subplot(337)
    fdirdif = fig.add_subplot(338)
    fdndif = fig.add_subplot(339)

    # True data from disort plotting
    im1 = up.imshow(flup_dat.values, aspect='auto', 
                extent=[flup_dat.columns.min(), flup_dat.columns.max(),
                        flup_dat.index.min(), flup_dat.index.max()],
                origin='lower')
    cbar1 = fig.colorbar(im1, ax=up)
    cbar1.set_label("Flup True Val")

    im2 = fdir.imshow(rfldir_dat.values, aspect='auto', 
                extent=[rfldir_dat.columns.min(), rfldir_dat.columns.max(),
                        rfldir_dat.index.min(), rfldir_dat.index.max()],
                origin='lower')
    cbar2 = fig.colorbar(im2, ax=fdir)
    cbar2.set_label("rfldir True Val")

    im3 = fdn.imshow(rfldn_dat.values, aspect='auto', 
                extent=[rfldn_dat.columns.min(), rfldn_dat.columns.max(),
                        rfldn_dat.index.min(), rfldn_dat.index.max()],
                origin='lower')
    cbar3 = fig.colorbar(im3, ax=fdn)
    cbar3.set_label("rfldn True Val")

    # Emulated data from NN plotting
    im4 = uppred.imshow(flup_pred.values, aspect='auto', 
                extent=[flup_pred.columns.min(), flup_pred.columns.max(),
                        flup_pred.index.min(), flup_pred.index.max()],
                origin='lower')
    cbar4 = fig.colorbar(im4, ax=uppred)
    cbar4.set_label("Flup NN Predict")

    im5 = fdirpred.imshow(rfldir_pred.values, aspect='auto', 
                extent=[rfldir_pred.columns.min(), rfldir_pred.columns.max(),
                        rfldir_pred.index.min(), rfldir_pred.index.max()],
                origin='lower')
    cbar5 = fig.colorbar(im5, ax=fdirpred)
    cbar5.set_label("rfldir NN Predict")

    im6 = fdnpred.imshow(rfldn_pred.values, aspect='auto', 
                extent=[rfldn_pred.columns.min(), rfldn_pred.columns.max(),
                        rfldn_pred.index.min(), rfldn_pred.index.max()],
                origin='lower')
    cbar6 = fig.colorbar(im6, ax=fdnpred)
    cbar6.set_label("rfldn NN Predict")

    # Percent difference in prediction 
    #flup_diff = np.absolute(((flup_dat - flup_pred)/flup_dat)*100)
    flup_diff = flup_dat - flup_pred
    im7 = updif.imshow(flup_diff.values, aspect='auto', 
                extent=[flup_diff.columns.min(), flup_diff.columns.max(),
                        flup_diff.index.min(), flup_diff.index.max()],
                origin='lower')
    cbar7 = fig.colorbar(im7, ax=updif)
    cbar7.set_label("True - Predicted")

    #rfldir_diff = np.absolute(((rfldir_dat - rfldir_pred)/rfldir_dat)*100)
    rfldir_diff = rfldir_dat - rfldir_pred
    im8 = fdirdif.imshow(rfldir_diff.values, aspect='auto', 
                extent=[rfldir_diff.columns.min(), rfldir_diff.columns.max(),
                        rfldir_diff.index.min(), rfldir_diff.index.max()],
                origin='lower')
    cbar8 = fig.colorbar(im8, ax=fdirdif)
    cbar8.set_label("True - Predicted")

    #rfldn_diff = np.absolute(((rfldn_dat - rfldn_pred)/rfldn_dat)*100)
    rfldn_diff = rfldn_dat - rfldn_pred
    im9 = fdndif.imshow(rfldn_diff.values, aspect='auto', 
                extent=[rfldn_diff.columns.min(), rfldn_diff.columns.max(),
                        rfldn_diff.index.min(), rfldn_diff.index.max()],
                origin='lower')
    cbar9 = fig.colorbar(im9, ax=fdndif)
    cbar9.set_label("True - Predicted")

    subs=[up, fdir, fdn, uppred, fdirpred, fdnpred, updif, fdirdif, fdndif]
    for ax in subs:
        ax.tick_params(length=6, width=1, labelsize=14)
        ax.tick_params(which='minor', length=4)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.set_yscale('log')
        ax.set_ylim([0.1, 1])
        ax.invert_yaxis()

    for ax in [updif, fdirdif, fdndif]:
        ax.set_xlabel('Wavelength [um]', size=16)

    for ax in [up, uppred, updif]:
        ax.set_ylabel('Pressure [bar]', size=16)

    fig.suptitle('Emulation Test for '+str(watermultiplier)+'x H2O VMR', fontweight='bold', size=18)
    plt.tight_layout()
    plt.savefig('Emulation'+str(testind)+'.png')
    plt.show()

    return [flup_dat, rfldir_dat, rfldn_dat, flup_pred, rfldir_pred, rfldn_pred, flup_diff, rfldir_diff, rfldn_diff]

def plot_test_waterVMR(testind, watermultiplier):

    emulation = np.load('../Predictions_Test'+str(testind)+'.npy')

    # Get the inds in the h5 file where the profile changes:
    inds = []
    arr = np.array(disortdat['IN'])
    for i in range(len(arr)-1):
        if arr[i][0] > arr[i+1][0]:
            inds.append(i+1)
    
    if testind == 1:
        lowb = None
        upb = inds[0]
    elif testind == 2:
        lowb = inds[0]
        upb = inds[1]
    elif testind == 3:
        lowb = inds[1]
        upb = None

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

    h2o_vmr = h2o_vmr_baseline * watermultiplier

    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(111)

    for e in emulation[:,150:]:
        ax1.plot(e, p, color='xkcd:black', alpha=0.2)

    q16, q5, q84 = np.quantile(emulation[:,150:], [0.16, 0.5, 0.84], axis=0)

    ax1.plot(q16, p, color='xkcd:blue', linestyle='--', linewidth=1, label='16th/84th percentile')
    ax1.plot(q5, p, color='xkcd:blue', linestyle='-', linewidth=1, label='50th percentile')
    ax1.plot(q84, p, color='xkcd:blue', linestyle='--', linewidth=1)

    ax1.plot(h2o_vmr, p, color='xkcd:red', linewidth=2, label='True VMR')

    subs = [ax1]
    for ax in subs:
        ax.tick_params(length=6, width=1, labelsize=14)
        ax.tick_params(which='minor', length=4)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.set_yscale('log')
        #ax.set_ylim([0.1, 1])
        ax.invert_yaxis()
        ax.legend(fontsize=16)

    fig.suptitle('Emulation Test for '+str(watermultiplier)+'x H2O VMR', fontweight='bold', size=18)
    plt.tight_layout()
    plt.savefig('WaterVMREmulation'+str(testind)+'.png')
    plt.show()


def plot_TOAFlup(resol='high'):

    em1 = np.load('../Predictions_Test1.npy')
    em2 = np.load('../Predictions_Test2.npy')
    em3 = np.load('../Predictions_Test3.npy')

    # Get the inds in the h5 file where the profile changes:
    inds = []
    arr = np.array(disortdat['IN'])
    for i in range(len(arr)-1):
        if arr[i][0] > arr[i+1][0]:
            inds.append(i+1)

    wav1 = np.array([w[0] for w in disortdat['IN'][None:inds[0]]])
    wav2 = np.array([w[0] for w in disortdat['IN'][inds[0]:inds[1]]])
    wav3 = np.array([w[0] for w in disortdat['IN'][inds[1]:None]])

    tru1 = disortdat['OUT'][None:inds[0],0]
    tru2 = disortdat['OUT'][inds[0]:inds[1],0]
    tru3 = disortdat['OUT'][inds[1]:None,0]

    #pr1 = em1[:,0]

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    if resol == 'high':
        mask = tru1 > 0
        ax1.plot(wav1[mask], tru1[mask], color='xkcd:red', label='0.7x H2O')
        mask = em1[:,0] > 0
        ax1.plot(wav1[mask], em1[:,0][mask], color='xkcd:dark red', linestyle='--', label='Emulation')
        
        mask = tru2 > 0
        ax2.plot(wav2[mask], tru2[mask], color='xkcd:orange', label='3.2x H2O')
        mask = em2[:,0] > 0
        ax2.plot(wav2[mask], em2[:,0][mask], color='xkcd:dark orange', linestyle='--', label='Emulation')
        
        mask = tru3 > 0
        ax3.plot(wav3[mask], tru3[mask], color='xkcd:blue', label='5.8x H2O')
        mask = em3[:,0] > 0
        ax3.plot(wav3[mask], em3[:,0][mask], color='xkcd:dark blue', linestyle='--', label='Emulation')
        
    
    elif resol == 'low':
        newwl = np.linspace(max(wav1[0], wav2[0], wav3[0]), min(wav1[len(wav1)-1], wav2[len(wav2)-1], wav3[len(wav3)-1]), 500)
        
        # can have as many or as few short/long cutoffs as the user desires; three are used here
        lams  = [max(wav1[0], wav2[0], wav3[0])]   # short wavelength cutoff (um)
        laml  = [min(wav1[len(wav1)-1], wav2[len(wav2)-1], wav3[len(wav3)-1])]   # long wavelength cutoff (um)
        res   = [5000]      # spectral resolving power (lam/dlam)

        # made-up hi-res grid and associated hi-res spectrum
        #lam_hr = np.arange(0.1,2.0,0.001)          # replace w/hi-res wavelength grid
        #F_hr   = np.random.normal(0,1,len(lam_hr)) # replace w/hi-res spectrally-dependent data

        # set lo-res grid
        lam_lr,dlam_lr = gen_spec_grid(lams,laml,np.float_(res),Nres=0)

        
        mask = em1[:,0] > 0
        #fl1 = instrument_non_uniform(newwl, wav1[mask], em1[:,0][mask], bw)
        # generate instrument response function
        kern = kernel(lam_lr,wav1[mask])
        # degrade spectrum
        fl1 = kernel_convol(kern,em1[:,0][mask])

        mask = tru1 > 0
        #tru1 = instrument_non_uniform(newwl, wav1[mask], tru1[mask], bw)
        # generate instrument response function
        kern = kernel(lam_lr,wav1[mask])
        # degrade spectrum
        tru1 = kernel_convol(kern,tru1[mask])

        mask = em2[:,0] > 0
        #fl2 = instrument_non_uniform(newwl, wav2[mask], em2[:,0][mask], bw)
        kern = kernel(lam_lr,wav2[mask])
        # degrade spectrum
        fl2 = kernel_convol(kern,em2[:,0][mask])

        mask = tru2 > 0
        #tru2 = instrument_non_uniform(newwl, wav2[mask], tru2[mask], bw)
        # generate instrument response function
        kern = kernel(lam_lr,wav2[mask])
        # degrade spectrum
        tru2 = kernel_convol(kern,tru2[mask])

        mask = em3[:,0] > 0
        #fl3 = instrument_non_uniform(newwl, wav3[mask], em3[:,0][mask], bw)
        # generate instrument response function
        kern = kernel(lam_lr,wav3[mask])
        # degrade spectrum
        fl3 = kernel_convol(kern,em3[:,0][mask])

        mask = tru3 > 0
        #tru3 = instrument_non_uniform(newwl, wav3[mask], tru3[mask], bw)
        # generate instrument response function
        kern = kernel(lam_lr,wav3[mask])
        # degrade spectrum
        tru3 = kernel_convol(kern,tru3[mask])

        ax1.plot(lam_lr, tru1, color='xkcd:dark red', label=r'0.7x H$_{2}$O')
        ax1.plot(lam_lr, fl1, color='xkcd:grey', linestyle=':', label='Emulation')

        ax2.plot(lam_lr, tru2, color='xkcd:dark orange', label=r'3.2x H$_{2}$O')
        ax2.plot(lam_lr, fl2, color='xkcd:grey', linestyle=':', label='Emulation')

        ax3.plot(lam_lr, tru3, color='xkcd:dark blue', label=r'5.8x H$_{2}$O')
        ax3.plot(lam_lr, fl3, color='xkcd:grey', linestyle=':', label='Emulation')


    subs=[ax1, ax2, ax3]
    for ax in subs:
        ax.tick_params(length=6, width=1, labelsize=14)
        ax.tick_params(which='minor', length=4)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        #ax.set_yscale('log')
        #ax.set_ylim([0.1, 1])
        #ax.invert_yaxis()
        ax.legend(fontsize=20, frameon=False)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    ax3.set_xlabel(r'Wavelength [$\mu$m]', size=20)
    fig.supylabel('Reflectance', size=20)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.078, bottom=0.09, right=0.993, top=0.99, wspace=0.2, hspace=0.075)
    plt.savefig('SpectraCompare_linear_'+resol+'res.png')
    plt.show()

    '''
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.plot(wav1, tru1-em1[:,0], color='xkcd:red', label='0.7x H2O')

    ax2.plot(wav2, tru2-em2[:,0], color='xkcd:orange', label='3.2x H2O')
    
    ax3.plot(wav3, tru3-em3[:,0], color='xkcd:blue', label='5.8x H2O')

    subs=[ax1, ax2, ax3]
    for ax in subs:
        ax.tick_params(length=6, width=1, labelsize=14)
        ax.tick_params(which='minor', length=4)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.set_yscale('log')
        #ax.set_ylim([0.1, 1])
        #ax.invert_yaxis()
        ax.legend(fontsize=16)


    ax3.set_xlabel('Wavelength [um]', size=18)
    fig.supylabel('Normalized Flux True - Predicted', size=18)

    plt.tight_layout()
    plt.savefig('DifferenceSpectraCompare.png')
    plt.show()
    '''
