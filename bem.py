import numpy as np
import scipy.interpolate as sci
import os
import warnings

class Blade:
    """
    A class to hold the properties of the blade. Chord, twist, and airfoil
    type id number are stored as interpolation functions. Each interpolant is a
    function of radius (m).
    """
    def __init__(self, r, c, twist, aft_id):
        # Check inputs
        assert np.size(r) == np.size(c), "c must be the same size as r"
        assert np.size(r) == np.size(twist), "twist must be the same size as r"
        assert np.size(r) == np.size(aft_id), "aft_id must be the same size as r"
        # chord length (m)
        self.c = sci.interp1d(r,c,kind="linear", \
            bounds_error=False, fill_value=(c[0],c[-1]))
        # twist angle (radians)
        self.twist = sci.interp1d(r,twist,kind="linear", \
            bounds_error=False, fill_value=(twist[0],twist[-1]))
        # id number of airfoil type
        self.aft_id = sci.interp1d(r,aft_id, kind="nearest", \
            bounds_error=False, fill_value=(aft_id[0],aft_id[-1]))

class Airfoil:
    """
    A class to hold the properties of an airfoil. Cl and Cd are stored as
    interpolation functions, which depend on the angle of attack (radians).
    """
    def __init__(self, name, AoA, cl, cd):
        # Check inputs
        assert np.size(AoA) == np.size(cl), "cl must be the same size as AoA"
        assert np.size(AoA) == np.size(cd), "cd must be the same size as AoA"
        # Airfoil name
        self.name = name
        # lift coefficient
        self.cl = sci.interp1d(AoA,cl,kind="linear", \
            bounds_error=False, fill_value=(cl[0],cl[-1]))
        # drag coefficient
        self.cd = sci.interp1d(AoA,cd,kind="linear", \
            bounds_error=False, fill_value=(cd[0],cd[-1]))

def read_airfoil_data(folder):
    """
    Reads the airfoil data files from 'folder'.

    Each file must be a delimited ASCII file with columns listing angle of
    attack (in degrees), cl coefficient,and cd coefficient. There should be no
    header on the first line.
    """
    types = os.listdir(folder)

    N = np.size(types)
    aft = np.empty(N, dtype=np.object)
    for i in range(0, N):
        A = np.loadtxt(folder + "/" + types[i])
        aft[i] = Airfoil(types[i], np.deg2rad(A[:,0]), A[:,1], A[:,2])

    return aft

def read_blade_data(file, aft):
    """
    Reads the blade data files from 'file' using the airfoil data contained in
    'aft'.

    The blade file must be a delimited ASCII file whose columns list
    radius(m), chord(m), twist(deg), airfoiltype. The first line is skipped and
    can be used for a header.

    'aft' must be an array of airfoil types constructed using read_airfoil_data
    """
    # Read blade data from file and create blade object
    A = np.loadtxt(file, skiprows=1, usecols=(0,1,2))
    Nbld = np.size(A,0)
    r = A[:,0]
    c = A[:,1]
    twist = np.deg2rad(A[:,2])
    airfoiltype = np.loadtxt(file, skiprows=1, usecols=3, \
        dtype=np.dtype('S10')).astype(str)

    # from aft generate array of the names of the airfoil types
    Naft = np.size(aft)
    aft_name = np.ndarray(Naft, dtype=np.object)
    for i in range(0, Naft):
        aft_name[i] = aft[i].name

    # identify which airfoil type index each segment is
    aft_id = np.zeros(Nbld, dtype=int)
    ind = np.linspace(0, Naft-1, Naft)
    for i in range(0, Nbld):
        val_list = ind[aft_name == airfoiltype[i]]
        assert np.size(val_list) == 1, "airfoiltype must match only one airfoil"
        aft_id[i] = int(val_list[0])

    # return blade object
    return Blade(r, c, twist, aft_id)

def bem(aft, bld, Rinner, Router, N, U, B, TSR, pitch):
    """
    Uses blade element momentum theory (BEM) to calculate the aerodynamic
    properties of a wind turbine.

    Parameters:
        aft: airfoil type data array created using read_airfoil_data
        bld: blade data created using read_blade_data
        Rinner: inner radius of blade (m)
        Router: outer radius of blade (m)
        N: number of annulus segments to use
        U: freestream velocity (m/s)
        B: number of blades
        TSR: tip speed ratio (omega*R/U)
        pitch: blade pitch (radians)

    Returns:
        Ct: thrust coefficient
        Cp: power coefficient
        r: array giving center of each annulus segment
        dL: array of differential lift for each annulus segment
        dD: array of differential drag for each annulus segment
        dT: array of differential thrust for each annulus segment
        dQ: array of differential power for each annulus segment

    Reference: Burton et al. Wind Energy Handbook. (2011)
    """
    # discretize the blade into annuli
    dr = (Router - Rinner)/N
    r = np.linspace(Rinner + 0.5*dr, Router - 0.5*dr, N)
    c = bld.c(r)
    twist = bld.twist(r)
    aft_id = [int(i) for i in bld.aft_id(r)]

    # preliminary variables
    omega = TSR*U/Router        # rotational speed
    sigma_r = B*c/(2*np.pi*r)   # chord solidity (3.56)
    tol = 1E-8                  # error tolerance
    maxIter = 5000              # maximum number of iterations

    # preallocate variables
    a = np.zeros(N)
    ap = np.zeros(N)
    Cl = np.zeros(N)
    Cd = np.zeros(N)
    error = 1E6
    iteration = 0
    a_prev = a
    ap_prev = ap

    while error > tol and iteration < maxIter:
        # set new guess and reset previous values
        a = 0.5 * (a + a_prev)
        ap = 0.5 * (ap + ap_prev)
        a_prev = a
        ap_prev = ap

        # calculate angle of attack (alpha)
        phi = np.arctan((1-a)*U/((1+ap)*omega*r))
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        alpha = phi - twist + pitch

        # read lift and drag coefficients from table
        for i in range(0, N):
            Cl[i] =  aft[aft_id[i]].cl(alpha[i])
            Cd[i] =  aft[aft_id[i]].cd(alpha[i])

        # normal and tangential coefficients
        Cn = Cl*cos_phi + Cd*sin_phi                            # 3.53
        Ct = Cl*sin_phi - Cd*cos_phi                            # 3.53

        # calculate new induction factors
        a = sigma_r * Cn / (sigma_r * Cn + 4*sin_phi**2)        # 3.54
        ap = sigma_r * Ct / (sigma_r * Ct + 4*sin_phi*cos_phi)  # 3.54

        # calculate error and advance iteration
        error = np.sum((a - a_prev)**2 + (ap - ap_prev)**2)
        iteration = iteration + 1

    # warn user if a solution was not found
    if iteration == maxIter:
        print("Warning: maximum number of iterations reached. Error = %f"%error)

    # calculate differential lif, drag, thrust, and torque
    W = np.sqrt(U**2*(1-a)**2 + r**2*omega**2*(1+ap)**2);
    dL = 0.5*W**2*c*Cl*dr;
    dD = 0.5*W**2*c*Cd*dr;
    dT = B*(dL*cos_phi + dD*sin_phi);
    dQ = B*(dL*sin_phi - dD*cos_phi)*r;

    # total thrust, torque, and power
    T = np.sum(dT)
    Q = np.sum(dQ)
    P = Q*omega

    # power and thrust coefficients
    A = np.pi*Router**2
    Ct = T/(0.5*A*U**2)
    Cp = P/(0.5*A*U**3)

    return Ct, Cp, r, dL, dD, dT, dQ

# An example script using BEM
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Rinner = 3.0/2
    Router = 126.0/2
    print(__file__[:-6])
    blade_file = __file__[:-6] + "/NREL5MW"
    airfoil_folder = __file__[:-6] + "/airfoilProperties"
    aft = read_airfoil_data(airfoil_folder)
    bld = read_blade_data(blade_file, aft)

    # Preliminary variables
    U = 8                   # U_infty
    B = 3                   # number of blades
    TSR = 7                 # tip speed ratio
    pitch = 0.0
    N = 128
    Ct, Cp, r, dL, dD, dT, dQ = bem(aft, bld, Rinner, Router, N, U, B, TSR, pitch)
    plt.plot(r,dL,r,dD,r,dT,r,dQ)
    plt.legend(['$dL$','$dD$','$dT$','$dQ$'])
    plt.xlabel('$r$ (m)')
    plt.title(r'NREL5MW turbine at $\lambda = 7$, $\beta = 0$')
    plt.show()

    print(r'Thrust and power coeffcients for NREL5MW turbine at $\lambda = 7$, $\beta = 0$')
    print("Ct = " + str(Ct))
    print("Cp = " + str(Cp))
