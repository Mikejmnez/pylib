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
                param.append(p[i])
                value.append(v[i])
    
    f.close()
    
    # now go through the parameter value pairs to find the ones we want to keep
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
    
    # Return values
    return t
    
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
        u[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Au[:,:,1:]
        
        Av = np.reshape(A[N:2*N],(lp['nx'],lp['ny'],lp['nz']),order='F')
        v[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Av[:,:,1:]
        
        Aw = np.reshape(A[2*N:],(lp['nx'],lp['ny'],lp['nz']),order='F')
        w[:,:,i*(lp['nz']-1):(lp['nz']-1)*(i+1)] = Aw[:,:,1:]

    return u, v, w
