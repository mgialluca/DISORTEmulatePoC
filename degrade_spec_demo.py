# example suite of routines for degrading spectral data 
# via gaussian convolution where the spectral smoothing 
# length can be wavelength-dependent. core example begins on 
# line indicated with "begin example".
#
# author: Tyler D. Robinson (tdrobin@arizona.edu)
#
import numpy    as     np
from   scipy    import interpolate
import matplotlib.pyplot as plt
from   astropy.io        import ascii
from   astropy.table     import Table, Column, MaskedColumn
#
#
# spectral grid routine, general
#
# inputs:
#
#      x_min  - minimum spectral cutoff
#      x_max  - maximumum spectral cutoff
#         dx  - if set, adopts fixed spacing of width dx
#        res  - if set, uses fixed or spectrally varying resolving power
#       lamr  - if set, uses spectrally varying resolving power and
#               lamr must have same size as res
#
# outputs:
#
#          x  - center of spectral gridpoints
#         Dx  - spectral element width
#
# notes:
#
#   in case of spectrally-varying resolving power, we use the 
#   derivative of the resolving power at x_i to find the resolving 
#   power at x_i+1.  this sets up a quadratic equation that relates 
#   the current spectral element to the next.
#
def spectral_grid(x_min,x_max,res = -1,dx = -1,lamr = -1):

  # constant resolution case
  if ( np.any(dx != -1) ): 
    x     = np.arange(x_min,x_max,dx)
    if (max(x) + dx == x_max):
      x = np.concatenate((x,[x_max]))
    Dx    = np.zeros(len(x))
    Dx[:] = dx
  # scenarios with constant or non-constant resolving power
  if ( np.any( res != -1) ):
    if ( np.any( lamr == -1) ): # constant resolving power
      x,Dx = spectral_grid_fixed_res(x_min,x_max,res)
    else: #spectrally-varying resolving power
      # function for interpolating spectral resolution
      res_interp = interpolate.interp1d(lamr,res,fill_value="extrapolate")

      # numerical derivative and interpolation function
      drdx = np.gradient(res,lamr)
      drdx_interp = interpolate.interp1d(lamr,drdx,fill_value="extrapolate")

      # initialize
      x  = [x_min]
      Dx = [x_min/res_interp(x_min)]
      i  = 0

      # loop until x_max is reached
      while (x[i] < x_max):
        resi =  res_interp(x[i])
        resp = drdx_interp(x[i])
        a = 2*resp
        b = 2*resi - 1 - 4*x[i]*resp - resp/resi*x[i]
        c = 2*resp*x[i]**2 + resp/resi*x[i]**2 - 2*x[i]*resi - x[i]
        if (a != 0 and resp*x[i]/resi > 1.e-6 ):
          xi1 = (-b + np.sqrt(b*b - 4*a*c))/2/a
        else:
          xi1 = (1+2*resi)/(2*resi-1)*x[i]
        Dxi = 2*(xi1 - x[i] - xi1/2/res_interp(xi1))
        x  = np.concatenate((x,[xi1]))
        i  = i+1
      if (max(x) > x_max):
        x  = x[0:-1]

      Dx = x/res_interp(x)

  return np.squeeze(x),np.squeeze(Dx)
#
#
# spectral grid routine, fixed resolving power
#
# inputs:
#
#     res     - spectral resolving power (x/dx)
#     x_min   - minimum spectral cutoff
#     x_max   - maximumum spectral cutoff
#
# outputs:
#
#         x   - center of spectral gridpoints
#        Dx   - spectral element width
#
def spectral_grid_fixed_res(x_min,x_max,res):
#
  x    = [x_min]
  fac  = (1 + 2*res)/(2*res - 1)
  i    = 0
  while (x[i]*fac < x_max):
    x = np.concatenate((x,[x[i]*fac]))
    i  = i + 1
  Dx = x/res
#
  return np.squeeze(x),np.squeeze(Dx)
#
#
# spectral grid routine, pieces together multiple wavelength
# regions with differing resolutions
#
# inputs:
#
#     res     - spectral resolving power (x/dx)
#     x_min   - minimum spectral cutoff
#     x_max   - maximumum spectral cutoff
#
# outputs:
#
#         x   - center of spectral gridpoints
#        Dx   - spectral element width
#
def gen_spec_grid(x_min,x_max,res,Nres=0):
  if ( len(x_min) == 1 ):
    x_min = x_min[0]
    x_max = x_max[0]
    res = res[0]
    x_sml = x_min/1e3
    x_low = max(x_sml,x_min - x_min/res*Nres)
    x_hgh = x_max + x_max/res*Nres
    x,Dx  = spectral_grid(x_low,x_hgh,res=res)
  else:
    x_sml = x_min[0]/1e3
    x_low = max(x_sml,x_min[0] - x_min[0]/res[0]*Nres)
    x_hgh = x_max[0] + x_max[0]/res[0]*Nres
    x,Dx  = spectral_grid(x_low,x_hgh,res=res[0])
    for i in range(1,len(x_min)):
      x_sml  = x_min[i]/1e3
      x_low  = max(x_sml,x_min[i] - x_min[i]/res[i]*Nres)
      x_hgh  = x_max[i] + x_max[i]/res[i]*Nres
      xi,Dxi = spectral_grid(x_low,x_hgh,res=res[i])
      x      = np.concatenate((x,xi)) 
      Dx     = np.concatenate((Dx,Dxi))
    Dx = [Dxs for _,Dxs in sorted(zip(x,Dx))]
    x  = np.sort(x)
  return np.squeeze(x),np.squeeze(Dx)
#
#
# set weights (kernel) for spectral convolution
#
# inputs:
#
#          x  - low-resolution spectral grid
#       x_hr  - high-resolution spectral grid (same units as x)
#
# outputs:
#
#       kern  - array describing wavelength-dependent kernels for
#               convolution (len(x) x len(x_hr))
#
# options:
#
#         Dx  - widths of low-resolution gridpoints (len(x))
#       mode  - vector (len(x)) of integers indicating if 
#               x_i is a spectroscopic point (1) or photometric 
#               point.  if not set, assumes all are spectroscopic 
#               and applies gaussian lineshape.
#
# notes:
#
#   designed to pair with kernel_convol function.  heavily modified 
#   and sped-up from a version originated by Mike Line.
#
def kernel(x,x_hr,Dx = -1,mode = -1):

  # number of points in lo-res grid
  Nx= len(x)

  # compute widths if not provided
  if ( np.any( Dx == -1) ):
    dx  = np.zeros(Nx)
    xm  = 0.5*(x[1:] + x[:-1])
    dx1 = xm[1:] - xm[:-1]
    dx[1:-1]    = dx1[:]
    res_interp  = interpolate.interp1d(x[1:-1],x[1:-1]/dx1,fill_value="extrapolate")
    dx[0]    = x[0]/res_interp(x[0])
    dx[Nx-1] = x[Nx-1]/res_interp(x[Nx-1])
  else:
    dx    = np.zeros(Nx)
    dx[:] = Dx

  # initialize output array
  kern = np.zeros([Nx,len(x_hr)])

  # loop over lo-res grid and compute convolution kernel
  fac = (2*(2*np.log(2))**0.5) # ~= 2.355

  # case where mode is not specified
  if ( np.any( mode == -1) ):
    for i in range(Nx):

      # FWHM = 2.355 * standard deviation of a gaussian
      sigma=dx[i]/fac

      # kernel
      kern[i,:]=np.exp(-(x_hr[:]-x[i])**2/(2*sigma**2))
      kern[i,:]=kern[i,:]/np.sum(kern[i,:])

  # case where mode is specified
  else:
    for i in range(Nx):
      if (mode[i] == 1): # spectroscopic point
        # FWHM = 2.355 * standard deviation of a gaussian
        sigma=dx[i]/fac

        # kernel
        kern[i,:] = np.exp(-(x_hr[:]-x[i])**2/(2*sigma**2))
        sumk      = np.sum(kern[i,:])
        if (sumk != 0):
          kern[i,:] = kern[i,:]/np.sum(kern[i,:])
        else:
          kern[i,:] = 0

      elif (mode[i] == 0): # photometric point
        j         = np.squeeze(np.where(np.logical_and(x_hr >= x[i]-Dx[i]/2, x_hr <= x[i]+Dx[i]/2)))
        if ( len(j) > 0 ):
          kern[i,j] = 1
          # edge handling
          jmin      = j[0]
          jmax      = j[-1]
          if (jmin == 0):
            Dxmin = abs(x_hr[jmin+1]-x_hr[jmin])
          else:
            Dxmin = abs( 0.5*(x_hr[jmin]+x_hr[jmin+1]) -  0.5*(x_hr[jmin]+x_hr[jmin-1]) )
          if (jmax == len(x_hr)-1):
            Dxmax = abs(x_hr[jmax]-x_hr[jmax-1])
          else:
            Dxmax = abs( 0.5*(x_hr[jmax]+x_hr[jmax+1]) -  0.5*(x_hr[jmax]+x_hr[jmax-1]) )
          xb = (x[i]-Dx[i]/2) - (x_hr[jmin]-Dxmin/2)
          xa = (x_hr[jmax]-Dxmax/2) - (x[i]+Dx[i]/2)
          if (xb >= 0):
            fb = 1 - xb/Dxmin
          else:
            fb = 1
          if (xa >= 0):
            fa = 1 - xa/Dxmax
          else:
            fa = 1
          kern[i,jmin] = fb
          kern[i,jmax] = fa
          kern[i,:] = kern[i,:]/np.sum(kern[i,:]) #re-normalize

  return kern

#
#
# convolve spectrum with general kernel
#
# inputs:
#
#      kern   - kernel matrix from kernel (len(low-res) x len(high-res))
#   spec_hr   - hi-res spectrum (len(hi-res))
#
# outputs:
#
#   spec_lr   - degraded spectrum (len(low-res))
#
def kernel_convol(kern,spec_hr):

  spec_lr    = np.zeros(kern.shape[0])
  conv       = np.multiply(kern,spec_hr)
  spec_lr[:] = np.sum(conv,axis=1)

  return spec_lr

### begin example ###

'''
# example min and max wavelength (um) and resolving power (lambda/Dlambda)
# can have as many or as few short/long cutoffs as the user desires; three are used here
lams  = 0.2,0.4,1.0   # short wavelength cutoff (um)
laml  = 0.4,1.0,1.8   # long wavelength cutoff (um)
res   = 7,140,40      # spectral resolving power (lam/dlam)

# made-up hi-res grid and associated hi-res spectrum
lam_hr = np.arange(0.1,2.0,0.001)          # replace w/hi-res wavelength grid
F_hr   = np.random.normal(0,1,len(lam_hr)) # replace w/hi-res spectrally-dependent data

# set lo-res grid
lam_lr,dlam_lr = gen_spec_grid(lams,laml,np.float_(res),Nres=0)

# generate instrument response function
kern = kernel(lam_lr,lam_hr)

# degrade spectrum
F_lr = kernel_convol(kern,F_hr)

# plot example
plt.plot(lam_hr,F_hr)
plt.plot(lam_lr,F_lr)
plt.show()
'''