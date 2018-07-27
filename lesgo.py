import numpy as np
import re

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
    lp['x'] = np.arange(0.0, lp['L_x'], lp['dx'])
    lp['y'] = np.arange(0.0, lp['L_y'], lp['dy'])
    lp['z_w'] = np.arange(0.0, lp['L_z']+0.5*lp['dz'], lp['dz'])
    lp['z_uv'] = np.arange(0.5*lp['dz'], lp['L_z'], lp['dz'])

    return lp

def turbine(folder, lp, N):
    """
    Reads the turbine files in folder/turbine with the parameters lp, and N is
    turbines. Everything is rescaled to be dimensional where necessary

    Each file has the following data:
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

    # Read the first file to get the number of time steps
    t = {}
    A = np.loadtxt(folder + "/turbine/turbine_1.dat")
    t['t'] = A[:,0]

    # Preallocate
    Nt = np.size(t['t'])
    t['uc'] = np.zeros((Nt,N))*lp['u_star']
    t['vc'] = np.zeros((Nt,N))
    t['wc'] = np.zeros((Nt,N))
    t['u_d'] = np.zeros((Nt,N))
    t['u_d_T'] = np.zeros((Nt,N))
    t['theta1'] = np.zeros((Nt,N))
    t['theta2'] = np.zeros((Nt,N))
    t['Ct_prime'] = np.zeros((Nt,N))
    if (np.size(A,1) > 9):
        t['Cp_prime'] = np.zeros((Nt,N))
        t['omega'] = np.zeros((Nt,N))

    # Read every turbine file
    for i in range(0, N):
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
    file = folder + '/output/vel.z-%0.5f.%i.c3.bin' % (zl, step)
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

def vel_zplane(folder, lp, step, zl):
    """
    Returns u, v, and w at z-plane zl at time step step in folder/output.
    """
    # Preallocate
    u = np.zeros((lp['nx'],lp['ny']))
    v = np.zeros((lp['nx'],lp['ny']))
    w = np.zeros((lp['nx'],lp['ny']))

    N = lp['nx']*lp['ny']
    file = folder + '/output/vel.z-%0.5f.%i.c3.bin' % (zl, step)
    A = np.fromfile(file, dtype=np.dtype(lp['write_endian']))

    u = np.reshape(A[0:N],(lp['nx'],lp['ny']),order='F')
    v = np.reshape(A[N:2*N],(lp['nx'],lp['ny']),order='F')
    w = np.reshape(A[2*N:3*N],(lp['nx'],lp['ny']),order='F')

    return u, v, w
