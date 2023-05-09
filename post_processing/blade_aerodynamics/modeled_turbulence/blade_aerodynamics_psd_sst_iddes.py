# state file generated using paraview version 5.9.0
import matplotlib as mpl 
import netCDF4 as nc
import numpy as np
#import mpi4py
#from mpi4py import MPI
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq 
import scipy.signal

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
from vtk.numpy_interface import dataset_adapter as dsa
import sys, os, glob, pickle
import mpi4py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

# For PSD:
mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['axes.titlesize'] = 10#18
mpl.rcParams['axes.labelsize'] = 10#18
mpl.rcParams['xtick.labelsize'] = 10#18
mpl.rcParams['ytick.labelsize'] = 10#18
mpl.rcParams['legend.fontsize'] = 7.0 
mpl.rcParams['figure.figsize'] = (6.328, 5.328)

rho = 1.246
dr = 0.1 
eps = 1.E-4
Fs = 0 

####################################################################
# Purpose: Plot of power spectral density(PSD), AOA, Urel and 
# Strouhal number (St) for SST and IDDES turbulence models.
####################################################################
# User specified parameters:
turbine_R = 5.0 # Turbine radius
omega = 7.529 # Rotational velocity

# radial locations where sectional forces need to be computed
rLoc = np.array([1.51, 2.343, 3.172, 4.023])
rbyRv = np.array([30, 47, 63, 80])
rbyRs = np.array([0.3, 0.47, 0.63, 0.80])

# chord length and sectional pitch angles computed at those locations
chord_len = np.array([0.711, 0.627, 0.543, 0.457])
sec_pitch_angle = np.array([19.0, 10.0, 6.0, 4.0])
Nrev_avg = 2.0
time_per_rev = 0.833
sim_time = Nrev_avg*time_per_rev

T_sst = 0.0005795  # sample period / spacing
T_iddes = 0.0002894 # sample period / spacing
sf = 0 
#####################################################################

# Pressure forces at r/R = 0.3, 0.47, 0.63, 0.80
def get_pressure_force(exo_file, uinf, curR, steps):

    # get the material library
    materialLibrary1 = GetMaterialLibrary()
    
    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [678, 539]
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.CenterOfRotation = [32.489392161369324, -0.4534595012664795, -9.000301361083984e-05]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [74.6759438795902, 3.018277656055136, 39.4111720801928]
    renderView1.CameraFocalPoint = [-12.053870140992457, -4.119141870291524, -41.61302382052542]
    renderView1.CameraViewUp = [0.6753062818870328, 0.09575269439292922, -0.7312953214402549]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 30.774361257105042
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1
    
    SetActiveView(None)
    
    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------
    
    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize((1079, 539))
    
    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------
    
    # create a new 'Exodus reader'
    bladesexo = ExodusIIReader(registrationName='blades.exo', FileName=[exo_file])
    bladesexo.PointVariables = ['pressure', 'pressure_force_', 'viscous_force_', 'mesh_displacement_', 'tau_wall']
    bladesexo.ElementBlocks = ['blade1_hex8_quad4']

    warpByVector1 = WarpByVector(Input=bladesexo)
    warpByVector1.Vectors = ['POINTS', 'mesh_displacement_']

    extractSurface1 = ExtractSurface(Input=warpByVector1)
    generateSurfaceNormals1 = GenerateSurfaceNormals(Input=extractSurface1)
    calculator1 = Calculator(Input=generateSurfaceNormals1)
    calculator1.ResultArrayName = 'pressureForce'
    calculator1.Function = 'Normals*pressure'

    calculator2 = Calculator(Input=calculator1)
    calculator2.ResultArrayName = 'viscousForce'
    calculator2.Function = 'viscous_force_ / mag(viscous_force_) * tau_wall'
  
    tsteps = bladesexo.TimestepValues
    nsteps = len(tsteps)
    
    renderView1.ViewTime = tsteps[-1]
    bshow = Show(calculator2, renderView1)

    # ******* Compute Pressure Forces *******
    tstep_range = np.array_split( range(steps), size)[rank]
    curT_vec = tsteps[-steps:]

    Fp = np.zeros((steps, 3))
    Fp_flatten = np.zeros(steps*3)
    g_Fp_flatten = np.zeros(steps*3)

    Fpx = np.zeros(steps)
    Fpy = np.zeros(steps)
    Fpz = np.zeros(steps)
 
    Fpx_flatten = np.zeros(steps)

    for it in tstep_range:

        renderView1.ViewTime = curT_vec[it]
        Render()
            
        # create a new 'Clip'
        clip1 = Clip(registrationName='Clip1', Input=calculator2)
        clip1.ClipType = 'Cylinder'                             
        clip1.Invert = 0                                        
        # init the 'Plane' selected for 'ClipType'              
        clip1.ClipType.Center = [0.0, 0.0, 0.0]                 
        clip1.ClipType.Axis = [1.0, 0.0, 0.0]                   
        clip1.ClipType.Radius = curR-dr*0.5                     
                                                                    
        clip2 = Clip(registrationName='Clip2', Input=clip1)     
        clip2.ClipType = 'Cylinder'                             
        clip2.Invert = 1                                        
        # init the 'Plane' selected for 'ClipType'              
        clip2.ClipType.Center = [0.0, 0.0, 0.0]                 
        clip2.ClipType.Axis = [1.0, 0.0, 0.0]                   
        clip2.ClipType.Radius = curR+dr*0.5                     
            
        # create a new 'Integrate Variables'
        integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=clip2)
        integrateVariables1.DivideCellDataByVolume = 1
            
        vtk_iv = servermanager.Fetch(integrateVariables1)
        numpy_iv = dsa.WrapDataObject(vtk_iv)
        pforce_calc = numpy_iv.PointData.GetArray('pressureForce')[0]
        pforce_nalu = numpy_iv.PointData.GetArray('pressure_force_')[0]
        tforce = numpy_iv.PointData.GetArray('viscousForce')[0]
        pforce = pforce_calc + tforce

        Fpx[it] = pforce[0]
        Fpy[it] = pforce[1]
        Fpz[it] = pforce[2]

        Delete(clip1)
        Delete(clip2)
        Delete(integrateVariables1)
        del clip1
        del clip2
        del integrateVariables1
        del vtk_iv
        del numpy_iv 

    comm.Reduce([Fpx, MPI.DOUBLE], [Fpx, MPI.DOUBLE], op = MPI.SUM, root = 0) 
    comm.Reduce([Fpy, MPI.DOUBLE], [Fpy, MPI.DOUBLE], op = MPI.SUM, root = 0) 
    comm.Reduce([Fpz, MPI.DOUBLE], [Fpz, MPI.DOUBLE], op = MPI.SUM, root = 0) 

    comm.Bcast([Fpx, MPI.DOUBLE], root = 0)
    comm.Bcast([Fpy, MPI.DOUBLE], root = 0)
    comm.Bcast([Fpz, MPI.DOUBLE], root = 0)
  
    return Fpx, Fpy, Fpz

# C_th vs. time, C_th_avg at r/R = 0.3, 0.47, 0.63, 0.80
# C_tq vs. time, C_tq_avg at r/R = 0.3, 0.47, 0.63, 0.80
# C_norm, C_tang at r/R = 0.3, 0.47, 0.63, 0.80
def get_forces(exo_file, uinf, T):

    steps = int(sim_time/T) # number of sample points

    C_th = np.zeros( (steps, np.size(rLoc) ) )
    C_tq = np.zeros( (steps, np.size(rLoc) ) )

    C_th_avg = np.zeros(np.size(rLoc))
    C_tq_avg = np.zeros(np.size(rLoc))

    # ************ Dynamic Pressure **************** 
    Pdyn = np.zeros(4)

    for i, c_len in enumerate(chord_len):
        Pdyn[i] = 0.5*rho*chord_len[i]*dr*(pow(uinf, 2.0) + pow(omega*rLoc[i], 2.0))

    theta = np.radians(sec_pitch_angle)
 
    # *********** get_pressure_force ***************
    for iR, curR in enumerate(rLoc):
        Fp = get_pressure_force(exo_file, uinf, curR, steps) 
        F_th = Fp[0]
        F_tq = Fp[1]

        # *********** Force Coefficients **************
        C_th_rbyR = np.divide(F_th,Pdyn[iR])
        C_tq_rbyR = np.divide(F_tq,Pdyn[iR])

        C_th[:,iR] = C_th_rbyR
        C_tq[:,iR] = C_tq_rbyR

        C_th_avg_curR = np.divide(np.mean(F_th),Pdyn[iR])
        C_tq_avg_curR = np.divide(np.mean(F_tq),Pdyn[iR])

        C_th_avg[iR] = C_th_avg_curR
        C_tq_avg[iR] = C_tq_avg_curR

    C_norm_avg = np.cos(theta)*C_th_avg + np.sin(theta)*C_tq_avg
    C_tang_avg = -np.sin(theta)*C_th_avg + np.cos(theta)*C_tq_avg

    return(C_th, C_th_avg, C_tq, C_tq_avg, C_norm_avg, C_tang_avg)

# Get angle of attack (and angle of relative wind)
# Based on the paper: 
# Guntur, S and Sorenson, N. N., An evaluation of several methods of
# determining the local angle of attack on wind turbine blade,
# Journal of Physics: Conference Series 555
# (2014).
# Based on book:
# Manwell, J. F. and McGowan, J. G. and Rogers, A. L.,
# Wind Energy Explained - Theory, Design and Applications, 2nd Edition,
# John Wiley & Sons Ltd., 2009.
def get_aoa_urel(C_th_avg, C_tq_avg, Uinf):

    # ****************** Angle of attack ******************
    B = 2.0 # Number of Blades

    a_axi = np.zeros(np.size(rLoc))
    a_tang = np.zeros(np.size(rLoc))
    phi = np.zeros(np.size(rLoc))
    alpha = np.zeros(np.size(rLoc))

    # ******* Sectional pitch angle *********
    theta = np.radians(sec_pitch_angle)
 
    for iR, curR in enumerate(rLoc):
      a_axi[iR] = 0.0
      a_tang[iR] = 0.0

      a_axi_old = 1.0E3
      a_tang_old = 1.0E3

      i = 0
      while (np.abs(a_axi[iR] - a_axi_old) > eps and np.abs(a_tang[iR] - a_tang_old) > eps):

        a_axi_old = a_axi[iR]
        a_tang_old = a_tang[iR]

        Vn = (1.0 - a_axi[iR])*uinf
        Vt = (1.0 + a_tang[iR])*(curR*omega)

        # Angle of relative wind:
        phi[iR] = np.arctan(Vn/Vt)

        # Correction Factor:
        F = (2.0/np.pi)*np.arccos(np.exp( -1.0*(B/2.0)*(1.0 - rLoc[iR]/turbine_R) / ((rLoc[iR]/turbine_R)*np.sin(phi[iR])) ))

        a_axi[iR] = 1.0 / (1.0 + (8.0*np.pi*rLoc[iR]*F*np.power(np.sin(phi[iR]), 2.0))/(chord_len[iR]*B*C_th_avg[iR]))
        a_tang[iR] = 1.0 / (1.0 + (8.0*np.pi*rLoc[iR]*F*np.sin(phi[iR])*np.cos(phi[iR]))/(chord_len[iR]*B*C_tq_avg[iR])) 

        i = i + 1
           
    alpha = phi - theta # angle of attack

    Urel = np.divide(Uinf*(1.0 - a_axi), np.sin(phi))

    return (alpha, Urel)

# Get power spectral density
def get_psd(C_th, T):
    N = int(sim_time/T) # number of sample points
    half_steps = N//2 - Fs
 
    # *************** Shedding Frequency ********************* 
    C_th_sampling_freq = np.zeros((half_steps, np.size(rLoc)))
    C_th_psd = np.zeros((half_steps, np.size(rLoc)))

    for iR, curR in enumerate(rLoc):
        C_th_curR = C_th[:,iR] 
        sampling_freq = fftfreq(N, T)[Fs:N//2]        
        C_th_curR_fft = fft(np.array(C_th_curR) - np.average(C_th_curR))
        energy = 2.0/N * np.abs(C_th_curR_fft[Fs:N//2])  
        C_th_sampling_freq[:, iR] = sampling_freq
        C_th_psd[:, iR] = energy
    return (C_th_sampling_freq, C_th_psd)

# Get power spectral density from experiments
def get_psd_exp(uinf):
    Texp = 0.00192 # sample spacing
    Nexp = 4400 #15625 # number of sample points 
    Fs_exp = 0
    half_steps = Nexp//2 - Fs_exp
 
    C_th_exp_sampling_freq = np.zeros((half_steps, np.size(rLoc)))
    C_th_exp_psd = np.zeros((half_steps, np.size(rLoc)))
 
    for iR, curR in enumerate(rLoc):
        cth_data = np.loadtxt("/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/" + "cth" + str(int(rbyRv[iR])) + 
                               "_for_u_" + str(int(uinf)) + "_time.dat", usecols=1, dtype=float)
        sampling_freq = fftfreq(Nexp, Texp)[Fs_exp:Nexp//2]       
        cth_data_fft = fft(np.array(cth_data[-Nexp:]) - np.average(cth_data[-Nexp:])) 
        energy = 2.0/Nexp * np.abs(cth_data_fft[Fs_exp:Nexp//2])  
        C_th_exp_sampling_freq[:, iR] = sampling_freq
        C_th_exp_psd[:, iR] = energy
    return (C_th_exp_sampling_freq, C_th_exp_psd)

# Primary shedding frequency
def get_shedding_freq(C_th, T):
    N = int(sim_time/T)  # number of sample points
 
    # *************** Shedding Frequency ********************* 
    C_th_max_freq_list = np.zeros(np.size(rLoc))
    for iR, curR in enumerate(rLoc):
        C_th_curR = C_th[:,iR] 
        sampling_freq = fftfreq(N, T)[sf:N//2]        
        C_th_curR_fft = fft(np.array(C_th_curR) - np.average(C_th_curR))                    
        energy = 2.0/N * np.abs(C_th_curR_fft[sf:N//2]) 
        max_freq = sampling_freq[energy.argmax()] 
        C_th_max_freq_list[iR] = max_freq
    
    return C_th_max_freq_list

# Strouhal Number
def get_St(C_th, alpha, T):
    N = int(sim_time/T) # number of sample points
 
    # *************** Strouhal Number ***********************
    chord_lengthn = np.multiply(chord_len,np.sin(alpha))
    
    C_th_St_list = np.zeros(np.size(rLoc))
    C_th_St_by_sin_list = np.zeros(np.size(rLoc))
 
    for iR, curR in enumerate(rLoc):
        C_th_curR = C_th[:,iR] 
        sampling_freq = fftfreq(N, T)[Fs:N//2]        
        C_th_curR_fft = fft(np.array(C_th_curR) - np.average(C_th_curR))                    
        energy = 2.0/N * np.abs(C_th_curR_fft[Fs:N//2]) 
        max_freq = sampling_freq[energy.argmax()] 
        st = max_freq*chord_lengthn[iR]/uinf
        st_by_sin = (max_freq)*chord_len[iR]/uinf
        C_th_St_list[iR] = st
        C_th_St_by_sin_list[iR] = st_by_sin
        
    return (C_th_St_list, C_th_St_by_sin_list)

# Get thurst, LSSTQ, power vs. time
def get_thrust_lsstq_power_vs_time(case_dir):

    force_file = case_dir+"/forces.dat"
    rot_time = 2.0 * np.pi / omega
    data = pd.read_csv(force_file, sep="\s+")

    self.thrust = data['Fpx']+data['Fvx']
    self.torque = -data['Mtx']
    self.power = self.torque * omega
    self.rotations = data['Time']/rot_time

    return(thrust, torque, power)

###############################################
# ***************** Plots *********************
 
def plot_psd_comp(C_th_sst, T_sst, C_th_iddes, T_iddes, uinf):
    C_th_samp_freq_psd_sst = get_psd(C_th_sst, T_sst)
    C_th_samp_freq_psd_iddes = get_psd(C_th_iddes, T_iddes)
    C_th_exp_samp_freq_psd  = get_psd_exp(uinf) 

    # CFD
    sampling_freq_sst = C_th_samp_freq_psd_sst[0]
    psd_sst = C_th_samp_freq_psd_sst[1]
    sampling_freq_iddes = C_th_samp_freq_psd_iddes[0] 
    psd_iddes = C_th_samp_freq_psd_iddes[1]

    # EXP
    sampling_freq_exp = C_th_exp_samp_freq_psd[0]
    psd_exp = C_th_exp_samp_freq_psd[1]

    with PdfPages('nrelvi_psd_plots_u_' + str(int(uinf)) + '.pdf') as pfpgs:
         plt.figure()  
         fig, axs = plt.subplots(2, 2, squeeze=False)
         k = 0
         for i in range(0, 2):
             for j in range (0, 2):
                 ax = axs[i, j]
                 sst, = ax.loglog(sampling_freq_sst[:,k], psd_sst[:,k], label="SST", linewidth=1)
                 iddes, = ax.loglog(sampling_freq_iddes[:,k], psd_iddes[:,k], label="IDDES", color="red", linewidth=1)
                 exp, = ax.loglog(sampling_freq_exp[:,k], psd_exp[:,k], label="EXP", color="black", linewidth=1)
#                 ax.loglog(sampling_freq_exp[:,0], pow(sampling_freq_exp[:,0], (-5.0/3.0)), color="black")
                 ax.set_title('r/R = {}'.format(rbyRs[k]))
                 ax.set_xlabel('Frequency (Hz)')
                 ax.set_ylabel('PSD ($C_{thrust}$)')
                 ax.legend(handles=[sst, iddes, exp])
                 ax.set_xlim(0.1, 1000.0)
                 ax.set_ylim(1.0e-10, 10.0)
                 k = k + 1
         plt.tight_layout()
         pfpgs.savefig()    
         plt.close(fig)

##############################################
if __name__=="__main__":

    exo_file_sst = sys.argv[1]
    exo_file_iddes = sys.argv[2]
    uinf = float(sys.argv[3])

    ###############################
    #    Force coefficients
    ###############################
    force_coeff_sst = get_forces(exo_file_sst, uinf, T_sst)
    force_coeff_iddes = get_forces(exo_file_iddes, uinf, T_iddes)

    C_th_sst = force_coeff_sst[0]
    C_th_avg_sst = force_coeff_sst[1]
    C_tq_sst = force_coeff_sst[2]
    C_tq_avg_sst = force_coeff_sst[3]

    C_th_iddes = force_coeff_iddes[0]
    C_th_avg_iddes = force_coeff_iddes[1]
    C_tq_iddes = force_coeff_iddes[2]
    C_tq_avg_iddes = force_coeff_iddes[3]

    
    ###############################
    #    Primary freq. and PSD
    ###############################
    C_th_max_freq_list_sst = get_shedding_freq(C_th_sst, T_sst)
    C_th_max_freq_list_iddes = get_shedding_freq(C_th_iddes, T_iddes)

    plot_psd_comp(C_th_sst, T_sst, C_th_iddes, T_iddes, uinf)    

    ############################## 
    #        AOA, Urel and St
    ##############################

    # AOA, Urel for SST
    alpha_urel_sst = get_aoa_urel(C_th_avg_sst, C_tq_avg_sst, uinf)
    alpha_sst = alpha_urel_sst[0]
    alpha_deg_sst = alpha_sst*(180.0/np.pi)
    Urel_sst = alpha_urel_sst[1]

    # AOA, Urel for IDDES
    alpha_urel_iddes = get_aoa_urel(C_th_avg_iddes, C_tq_avg_iddes, uinf)
    alpha_iddes = alpha_urel_iddes[0]
    alpha_deg_iddes = alpha_iddes*(180.0/np.pi)
    Urel_iddes = alpha_urel_iddes[1]

    # St for SST
    C_th_St_sst = get_St(C_th_sst, alpha_sst, T_sst)
    C_th_St_list_sst = C_th_St_sst[0]
    C_th_St_by_sin_list_sst = C_th_St_sst[1]

    # St for IDDES
    C_th_St_iddes = get_St(C_th_iddes, alpha_iddes, T_iddes)
    C_th_St_list_iddes = C_th_St_iddes[0]
    C_th_St_by_sin_list_iddes = C_th_St_iddes[1]


    ###############################
    #     Table of Results
    ###############################

    # Ueff and AOA:
    if (rank == 0):
       print("       \t SST:             \t\t IDDES:\n")    
       print("Radius \t Ueff[m/s] \t AOA[deg.] \t Ueff[m/s] \t AOA[deg.]\n")
       for iR, curR in enumerate(rLoc):
           print("%5.4f \t %5.4f \t %5.4f \t %5.4f \t %5.4f \t" % (curR, Urel_sst[iR], alpha_deg_sst[iR], 
                                                                   Urel_iddes[iR], alpha_deg_iddes[iR]))
    # Freq. and St:
    if (rank == 0):
       print("\n")
       print("       \t SST:            \t\t IDDES:\n")    
       print("Radius \t f[Hz] \t \t St \t\t  f[Hz] \t St\n")
       for iR, curR in enumerate(rLoc):
           print("%5.4f \t %5.4f \t %5.4f \t %5.4f \t %5.4f \t" % (curR, C_th_max_freq_list_sst[iR], C_th_St_list_sst[iR],
                                                                   C_th_max_freq_list_iddes[iR], C_th_St_list_iddes[iR]))
