import numpy as np
import re
import scipy.io
import scipy

def param(folder):
    """
    Opens the lesgo_param.out file in folder/output/. Only certain parameters
    are saved in the dictionary that is returned. But, this function does extra
    work so that adding additional parameter-value pairs should be
    straightforward.
    """
    f = open(folder + "/output/lesgo_param.out")

    # Create objects
    lp = {}
    param = []
    value = []

    # Read lines
    for line in f:
        # The line is separated into a list of peters and values
        # by a ':'
        l = line[:-1].split(":")
        if len(l) == 2:
            # Split the parameters
            p = l[0].split(",")
            for i in range(0, len(p)):
                p[i] = p[i].strip(' ')

            # Split the corresponding values
            v = re.sub(' +',' ',l[1]).strip(' ').split(' ')
            for i in range(0, len(v)):
                v[i] = v[i].strip(' ')

            # after this point, we don't need anything
            if p[0] == "tavg_calc":
                break

            # now add to param and val list
            for i in range(0, len(p)):
                if (len(p) == len(v)):
                    param.append(p[i])
                    value.append(v[i])

    f.close()

    # go through the parameter value pairs to find the ones we want to keep
    for i in range(0, len(param)):
        p = param[i]
        v = value[i]

        # Ints
        if p == 'nproc' or p == 'nx' or p == 'ny' or p == 'nz' or p == 'nz_tot':
            lp[p] = int(v)
        # Floats
        if p == 'z_i' or p == 'L_x' or p == 'L_y' or p == 'L_z' or p == 'dx'   \
            or p == 'dy' or p == 'dz' or p == 'u_star':
            lp[p] = float(v)
        # Endianness
        if p == 'write_endian':
            if v == 'LITTLE_ENDIAN':
                lp[p] = '<f8'
            elif v == 'BIG_ENDIAN':
                lp[p] = '>f8'
            else:
                lp[p] = '=f8'

    # Create the mesh
    lp['x'] = np.arange(0.0, lp['L_x']-0.1*lp['dx'], lp['dx'])
    lp['y'] = np.arange(0.0, lp['L_y']-0.01*lp['dy'], lp['dy'])
    lp['z_w'] = np.arange(0.0, lp['L_z']+0.51*lp['dz'], lp['dz'])
    lp['z_uv'] = np.arange(0.5*lp['dz'], lp['L_z'], lp['dz'])

    # Wavenumbers
    lp['kx'] = np.fft.fftfreq(lp['nx'])*lp['nx']*2.0*np.pi/lp['L_x']
    lp['ky'] = np.fft.rfftfreq(lp['ny'])*lp['ny']*2.0*np.pi/lp['L_y']

    return lp

def ddx(lp,u):
    """
    Calculates spectral x derivatives
    """

    # Check size of variable
    assert np.size(u,0) == lp['nx'], "variable is not on a valid grid"
    assert np.size(u,1) == lp['ny'], "variable is not on a valid grid"

    # Create arrays
    dudx_hat = np.zeros((lp['nx'],lp['ny']//2+1),dtype=np.complex)
    dudx = np.zeros(np.shape(u))

    # Iterate through each z plane
    for k in range(0, np.size(u,2)):
        u_hat = np.fft.rfft2(u[:,:,k])
        for i in range(0, np.size(lp['kx'])):
            dudx_hat[i,:] = u_hat[i,:]*lp['kx'][i]*np.complex(0,1)
        dudx_hat[:,-1] = np.complex(0,0)
        dudx_hat[lp['nx']//2,:] = np.complex(0,0)
        dudx[:,:,k] = np.fft.irfft2(dudx_hat)

    return dudx

def ddy(lp,u):
    """
    Calculates spectral y derivatives
    """

    # Check size of variable
    assert np.size(u,0) == lp['nx'], "variable is not on a valid grid"
    assert np.size(u,1) == lp['ny'], "variable is not on a valid grid"

    # Create arrays
    dudy_hat = np.zeros((lp['nx'],lp['ny']//2+1),dtype=np.complex)
    dudy = np.zeros(np.shape(u))

    # Iterate through each z plane
    for k in range(0, np.size(u,2)):
        u_hat = np.fft.rfft2(u[:,:,k])
        for j in range(0, np.size(lp['ky'])):
            dudy_hat[:,j] = u_hat[:,j]*lp['ky'][j]*np.complex(0,1)
        dudy_hat[:,-1] = np.complex(0,0)
        dudy_hat[lp['nx']//2,:] = np.complex(0,0)
        dudy[:,:,k] = np.fft.irfft2(dudy_hat)

    return dudy

def ddxy(lp,u):
    """
    Calculates spectral x,y derivatives
    """

    # Check size of variable
    assert np.size(u,0) == lp['nx'], "variable is not on a valid grid"
    assert np.size(u,1) == lp['ny'], "variable is not on a valid grid"

    # Create arrays
    dudx_hat = np.zeros((lp['nx'],lp['ny']//2+1),dtype=np.complex)
    dudx = np.zeros(np.shape(u))
    dudy_hat = np.zeros((lp['nx'],lp['ny']//2+1),dtype=np.complex)
    dudy = np.zeros(np.shape(u))

    # Iterate through each z plane
    for k in range(0, np.size(u,2)):
        u_hat = np.fft.rfft2(u[:,:,k])
        for i in range(0, np.size(lp['kx'])):
            dudx_hat[i,:] = u_hat[i,:]*lp['kx'][i]*np.complex(0,1)
        dudx_hat[:,-1] = np.complex(0,0)
        dudx_hat[lp['nx']//2,:] = np.complex(0,0)
        dudx[:,:,k] = np.fft.irfft2(dudx_hat)
        for j in range(0, np.size(lp['ky'])):
            dudy_hat[:,j] = u_hat[:,j]*lp['ky'][j]*np.complex(0,1)
        dudy_hat[:,-1] = np.complex(0,0)
        dudy_hat[lp['nx']//2,:] = np.complex(0,0)
        dudy[:,:,k] = np.fft.irfft2(dudy_hat)

    return dudx, dudy

def ddz(lp,u):
    """
    Calculates finite differece z derivatives
    """

    # w grid -> uv grid
    if (np.size(u,2) == lp['nz_tot']):
        dudz = (u[:,:,1:] - u[:,:,0:-1])/lp['dz']
    # uv grid -> w grid
    elif (np.size(u,2) == lp['nz_tot']-1):
        dudz = np.zeros((np.size(u,0),np.size(u,1),lp['nz_tot']))
        dudz[:,:,1:-1] = (u[:,:,1:] - u[:,:,0:-1])/lp['dz']
    # invalid grid
    else:
        assert False, "variable is not on a valid grid"

    return dudz

def interp_uv_to_w(lp,uv):
    """
    Interpolates from uv grid to w grid
    """
    assert np.size(uv,2) == lp['nz_tot']-1, "variable is not on a valid grid"
    w = np.zeros((np.size(uv,0),np.size(uv,1),lp['nz_tot']))
    w[:,:,1:-1] = 0.5*(uv[:,:,1:] + uv[:,:,:-1])

    return w


def interp_w_to_uv(lp,w):
    """
    Interpolates from uv grid to w grid
    """
    assert np.size(w,2) == lp['nz_tot'], "variable is not on a valid grid"
    uv = 0.5*(w[:,:,1:] + w[:,:,:-1])

    return uv

def turbine(folder, lp, N=-1):
    """
    Reads the turbine files in folder/turbine and input_turbines/param.dat. lp
    is the lesgio parameters dictionary. Everything is rescaled to be
    dimensional where necessary

    N: Number of turbines
    Nt: Number of time points

    The param.dat file has the following data:
        x: x location of turbine
        y: y location of turbine
        z: z location of turbine
        D: Diameter of turbine

    Each turbine file has the following data:
        t: current time (dimensional
        uc: disk center u velocity
        vc: disk center v velocity
        wc: disk center w velocity
        u_d: instantaneous disk-averaged velocity
        u_d_T: current time- and disk-averaged velocity
        theta1: yaw angle
        theta2: tilt angle
        Ct_prime: local thrust coefficient
        Cp_prime: local power coefficient
        omega: rotational speed
    """

    # Create empty dictionary
    t = {}

    # Read the input_param.dat file to get the number of turbines
    A = np.loadtxt(folder + "/input_turbines/param.dat", delimiter=',')
    if (N < 1):
        t['N'] = np.size(A,0)
        # Save the turbine locations and diameter
        if (np.ndim(A) == 1):
            A = np.array(A,ndmin=2)
        print(A)
        t['x'] = A[:,0]
        t['y'] = A[:,1]
        t['z'] = A[:,2]
        t['D'] = A[:,3]
    else:
        t['N'] = N
    print(t['N'])

    # Read the first file to get the number of time steps
    A = np.loadtxt(folder + "/turbine/turbine_1.dat")
    t['t'] = A[:,0]
    t['Nt'] = np.size(t['t'])

    # Preallocate
    t['uc'] = np.zeros((t['Nt'],t['N']))*lp['u_star']
    t['vc'] = np.zeros((t['Nt'],t['N']))
    t['wc'] = np.zeros((t['Nt'],t['N']))
    t['u_d'] = np.zeros((t['Nt'],t['N']))
    t['u_d_T'] = np.zeros((t['Nt'],t['N']))
    t['theta1'] = np.zeros((t['Nt'],t['N']))
    t['theta2'] = np.zeros((t['Nt'],t['N']))
    t['Ct_prime'] = np.zeros((t['Nt'],t['N']))
    if (np.size(A,1) > 9):
        t['Cp_prime'] = np.zeros((t['Nt'],t['N']))
        t['omega'] = np.zeros((t['Nt'],t['N']))

    # Read every turbine file
    for i in range(0, t['N']):
        A = np.loadtxt(folder + "/turbine/turbine_" + str(i+1) + ".dat")
        t['uc'][:,i] = A[:,1]*lp['u_star']
        t['vc'][:,i] = A[:,2]*lp['u_star']
        t['wc'][:,i] = A[:,3]*lp['u_star']
        t['u_d'][:,i] = A[:,4]*lp['u_star']
        t['u_d_T'][:,i] = A[:,5]*lp['u_star']
        t['theta1'][:,i] = A[:,6]
        t['theta2'][:,i] = A[:,7]
        t['Ct_prime'][:,i] = A[:,8]
        if (np.size(A,1) > 9):
            t['Cp_prime'][:,i] = A[:,9]
            t['omega'][:,i] = A[:,10]

    # Return values
    return t

def vel_inst(folder, lp, step):
    """
    Returns u, v, and w on uv-grid for time step step in folder/output.
    """
    # Preallocate
    u = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    v = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    w = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/vel.%i.c%i.bin' % (step, i)
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Au = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        u[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Au[:,:,0:-1]

        Av = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        v[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Av[:,:,0:-1]

        Aw = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        w[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[:,:,0:-1]

    return u, v, w

def restart(folder, lp):
    """
    Returns u, v, and w on uv-grid for time step step in folder/output.
    """
    # Preallocate
    u = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    v = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    w = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        NN = (lp['nx']+2)*lp['ny']*lp['nz']
        file = folder + '/vel.out.c%i' % i
        A = scipy.io.FortranFile(file).read_reals()

        Au = np.reshape(A[0:NN],(lp['nx']+2,lp['ny'],lp['nz']),order='F')
        u[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Au[0:lp['nx'],:,0:-1]

        Av = np.reshape(A[NN:2*NN],(lp['nx']+2,lp['ny'],lp['nz']),order='F')
        v[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Av[0:lp['nx'],:,0:-1]

        Aw = np.reshape(A[2*NN:3*NN],(lp['nx']+2,lp['ny'],lp['nz']),order='F')
        w[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[0:lp['nx'],:,0:-1]

    return u, v, w

def theta_inst(folder, lp, step):
    """
    Returns u, v, and w on uv-grid for time step step in folder/output.
    """
    # Preallocate
    theta = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/theta.%i.c%i.bin' % (step, i)
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        At = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        theta[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = At[:,:,0:-1]

    return theta

def pressure_inst(folder, lp, step):
    """
    Returns pressure on uv-grid for time step step in folder/output.
    """
    # Preallocate
    p = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/pres.%i.c%i.bin' % (step, i)
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))
        Ap = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        p[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ap[:,:,0:-1]

    return p

def vel_zplane(folder, lp, step, zl):
    """
    Returns u, v, and w at z-plane zl at time step step in folder/output.
    """
    # Preallocate
    u = np.zeros((lp['nx'],lp['ny']))
    v = np.zeros((lp['nx'],lp['ny']))
    w = np.zeros((lp['nx'],lp['ny']))

    N = lp['nx']*lp['ny']
    file = folder + '/output/vel.z-%0.5f.%i.c4.bin' % (zl, step)
    A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

    u = np.reshape(A[0:N],(lp['nx'],lp['ny']),order='F')
    v = np.reshape(A[N:2*N],(lp['nx'],lp['ny']),order='F')
    w = np.reshape(A[2*N:3*N],(lp['nx'],lp['ny']),order='F')

    return u, v, w

def vort_inst(folder, lp, step):
    """
    Returns vorticity on w-grid for time step step in folder/output.
    """
    # Preallocate
    wx = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    wy = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    wz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/vort.%i.c%i.bin' % (step, i)
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Au = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        wx[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)+1] = Au[:,:,0:]

        Av = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        wy[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)+1] = Av[:,:,0:]

        Aw = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        wz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)+1] = Aw[:,:,0:]

    return wx, wy, wz

def veluv_avg(folder, lp):
    """
    Returns u, v, and w for uv-averaged data in folder/output.
    """
    # Preallocate
    u = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    v = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    w = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/veluv_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Au = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        u[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Au[:,:,:-1]

        Av = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        v[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Av[:,:,:-1]

        Aw = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        w[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[:,:,:-1]

    return u, v, w

def force_avg(folder, lp):
    """
    Returns force for uv-averaged data in folder/output.
    """
    # Preallocate
    fx = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    fy = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    fz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/force_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Au = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        fx[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Au[:,:,:-1]

        Av = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        fy[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Av[:,:,:-1]

        Aw = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        fz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[:,:,:-1]

    return fx, fy, fz

def velw_avg(folder, lp):
    """
    Returns w for w-averaged data in folder/output.
    """
    # Preallocate
    w = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/velw_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))
        Aw = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        w[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[:,:,:-1]

    return w

def pressure_avg(folder, lp):
    """
    Returns pressure for uv-averaged data in folder/output.
    """
    # Preallocate
    p = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/pres_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Ap = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        p[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ap[:,:,:-1]

    return p

def vort_avg(folder, lp):
    """
    Returns pressure for w-averaged data in folder/output.
    """
    # Preallocate
    vortx = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    vorty = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    vortz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/vort_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Ax = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        vortx[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ax[:,:,:-1]

        Ay = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        vorty[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ay[:,:,:-1]

        Az = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        vortz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Az[:,:,:-1]

    return vortx, vorty, vortz

def tau_avg(folder, lp):
    """
    Returns u, v, and w for uv-averaged data in folder/output.
    """
    # Preallocate
    txx = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    txy = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    tyy = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    txz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    tyz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']))
    tzz = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/tau_avg.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Axx = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        txx[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axx[:,:,:-1]

        Axy = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        txy[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axy[:,:,:-1]

        Ayy = np.reshape(A[2*N:3*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        tyy[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ayy[:,:,:-1]

        Axz = np.reshape(A[3*N:4*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        txz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axz[:,:,:-1]

        Ayz = np.reshape(A[4*N:5*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        tyz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ayz[:,:,:-1]

        Azz = np.reshape(A[5*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        tzz[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Azz[:,:,:-1]

    return txx, txy, tyy, txz, tyz, tzz

def rs_avg(folder, lp):
    """
    Returns u, v, and w for uv-averaged data in folder/output.
    """
    # Preallocate
    u2 = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    v2 = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    w2 = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    uw = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    vw = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))
    uv = np.zeros((lp['nx'],lp['ny'],lp['nz_tot']-1))

    # Read the data
    for i in range(0, lp['nproc']):
        N = lp['nx']*lp['ny']*lp['nz']
        file = folder + '/output/rs.c%i.bin' % i
        A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

        Axx = np.reshape(A[0:N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        u2[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axx[:,:,:-1]

        Axy = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        v2[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axy[:,:,:-1]

        Ayy = np.reshape(A[2*N:3*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        w2[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ayy[:,:,:-1]

        Axz = np.reshape(A[3*N:4*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        uw[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Axz[:,:,:-1]

        Ayz = np.reshape(A[4*N:5*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        vw[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Ayz[:,:,:-1]

        Azz = np.reshape(A[5*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        uv[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Azz[:,:,:-1]

    return u2, v2, w2, uw, vw, uv
