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
import pandas as pd

mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 14.0
mpl.rcParams['figure.figsize'] = (6.328, 5.328)


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
from vtk.numpy_interface import dataset_adapter as dsa
import sys, os, glob, pickle
import mpi4py
from mpi4py import MPI

####################################################################
# User specified parameters:
rho = 1.246
turbine_R = 5.029 
omega = 7.529
dr = 0.1

wspeed = np.array([7, 12, 15, 20])
rLoc = np.array([1.51, 2.343, 3.172, 4.023, 4.780])
rbyR_exp = np.array([0.3, 0.47, 0.63, 0.80, 0.95])
rbyR = rbyR_exp
chord_len = np.array([0.711, 0.627, 0.543, 0.457, 0.381])
sec_pitch_angle = np.array([19.0, 10.0, 6.0, 4.0, 3.0])

NIt_per_rev_sst = 1440
NIt_per_rev_iddes = 2880
Nrev = 5
#####################################################################

# Pressure forces at r/R = 0.3, 0.47, 0.63, 0.80
def get_pressure_force(exo_file, uinf, iR, curR, Nrev, NIt_per_rev, steps):

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
    bladesexo.ElementBlocks = ['blade2_hex8_quad4']

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
    renderView1.ViewTime = tsteps[-1]
    bshow = Show(calculator2, renderView1)

    # ******* Compute Pressure Forces *******
    tstep_range = range(steps)
    curT_vec = tsteps[-steps:]
 
    Fp = np.zeros((Nrev+1, 3))
    Fpx = np.zeros(Nrev+1)
    Fpy = np.zeros(Nrev+1)
    Fpz = np.zeros(Nrev+1)

    Fpx_sum = np.zeros(Nrev+1)
    Fpy_sum = np.zeros(Nrev+1)
    Fpz_sum = np.zeros(Nrev+1)
 
    it = 0
    k = 0
    while it in tstep_range:

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
        pforce = pforce_calc 

        Fpx[k] = pforce[0]
        Fpy[k] = pforce[1]
        Fpz[k] = pforce[2]

        Delete(clip1)
        Delete(clip2)
        Delete(integrateVariables1)
        del clip1
        del clip2
        del integrateVariables1
        del vtk_iv
        del numpy_iv 

        it = it + NIt_per_rev
        k = k + 1

    Fp[:,0]  = Fpx
    Fp[:,1]  = Fpy
    Fp[:,2]  = Fpz
  
    return Fp

# C_th vs. time, C_th_avg at r/R = 0.3, 0.47, 0.63, 0.80
# C_tq vs. time, C_tq_avg at r/R = 0.3, 0.47, 0.63, 0.80
# C_norm, C_tang at r/R = 0.3, 0.47, 0.63, 0.80
def get_forces(exo_file, uinf, Nrev, NIt_per_rev):

    steps = Nrev*NIt_per_rev + 1; #Number of time steps

    C_th = np.zeros( (Nrev+1, np.size(rLoc) ) )
    C_tq = np.zeros( (Nrev+1, np.size(rLoc) ) )

    C_th_avg = np.zeros(np.size(rLoc))
    C_tq_avg = np.zeros(np.size(rLoc))

    # ************ Dynamic Pressure **************** 
    Pdyn = np.zeros(5)

    for i, c_len in enumerate(chord_len):
        Pdyn[i] = 0.5*rho*c_len*dr*(pow(uinf, 2.0) + pow(omega*rLoc[i], 2.0))

    # *********** get_pressure_force ***************
    for iR, curR in enumerate(rLoc):
        Fp = get_pressure_force(exo_file, uinf, iR, curR, Nrev, NIt_per_rev, steps)
        F_th = np.abs(Fp[:, 0])
        F_tq = np.abs(Fp[:, 1])

        # *********** Force Coefficients **************
        C_th_rbyR = np.divide(F_th,Pdyn[iR])
        C_tq_rbyR = np.divide(F_tq,Pdyn[iR])

        C_th[:,iR] = C_th_rbyR
        C_tq[:,iR] = C_tq_rbyR

        C_th_avg_curR = np.divide(np.mean(F_th),Pdyn[iR])
        C_tq_avg_curR = np.divide(np.mean(F_tq),Pdyn[iR])

        C_th_avg[iR] = C_th_avg_curR
        C_tq_avg[iR] = C_tq_avg_curR

    sec_angle_rad = np.radians(sec_pitch_angle)

    C_norm_avg = np.cos(sec_angle_rad)*C_th_avg + np.sin(sec_angle_rad)*C_tq_avg
    C_tang_avg = -np.sin(sec_angle_rad)*C_th_avg + np.cos(sec_angle_rad)*C_tq_avg

    return(C_th, C_th_avg, C_tq, C_tq_avg, C_norm_avg, C_tang_avg)

###############################################
# ***************** Plots *********************
def compare_cth_exp_data(C_th_mt, C_th_avg_mt, C_th_lti, C_th_avg_lti, C_th_hti, C_th_avg_hti, uinf):
    cth_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/cth_avg_u_' +
                    str(int(uinf)) + '.txt', 'r')
    cth_exp_data = cth_exp.readlines()

    cth_avg_exp_data = []
    cth_min_exp_data = []
    cth_max_exp_data = []

    for i, data in enumerate(cth_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               cth_avg_exp_data.append(float(j))
        if (i == 1):
           min_data_string = data.split()
           for j in min_data_string:
               cth_min_exp_data.append(float(j))
        if (i == 2):
           max_data_string = data.split()
           for j in max_data_string:
               cth_max_exp_data.append(float(j))

    cth_min_mt = []
    cth_max_mt = []

    cth_min_lti = []
    cth_max_lti = []

    cth_min_hti = []
    cth_max_hti = []

    for iR, curR in enumerate(rLoc):
        cth_min_mt.append(C_th_avg_mt[iR] - np.std(C_th_mt[:,iR]))
        cth_max_mt.append(C_th_avg_mt[iR] + np.std(C_th_mt[:,iR]))
        cth_min_lti.append(C_th_avg_lti[iR] - np.std(C_th_lti[:,iR]))
        cth_max_lti.append(C_th_avg_lti[iR] + np.std(C_th_lti[:,iR]))
        cth_min_hti.append(C_th_avg_hti[iR] - np.std(C_th_hti[:,iR]))
        cth_max_hti.append(C_th_avg_hti[iR] + np.std(C_th_hti[:,iR]))

    with PdfPages('C_th_comp_st_' + str(uinf) + '.pdf') as pfpgs:
         plt.figure()
         ms = 200
         plt.plot(rbyR, cth_avg_exp_data, marker='o', color='black', label='EXP')
         plt.scatter(rbyR, cth_min_exp_data, marker='_', color='black', s=ms)
         plt.scatter(rbyR, cth_max_exp_data, marker='_', color='black', s=ms)

         plt.plot(rbyR, C_th_avg_mt, marker='v', color='green', label='MT (0.5%)')
         plt.scatter(rbyR, cth_min_mt, marker='_', color='green')
         plt.scatter(rbyR, cth_max_mt, marker='_', color='green')

         plt.plot(rbyR, C_th_avg_lti, marker='x', color='blue', label='ST (0.5%)')
         plt.scatter(rbyR, cth_min_lti, marker='_', color='blue')
         plt.scatter(rbyR, cth_max_lti, marker='_', color='blue')

         plt.plot(rbyR, C_th_avg_hti, marker='*', color='red', label='ST (6%)')
         plt.scatter(rbyR, cth_min_hti, marker='_', color='red')
         plt.scatter(rbyR, cth_max_hti, marker='_', color='red')

         for i, curR in enumerate(rbyR_exp):
             plt.vlines(rbyR_exp[i], cth_min_exp_data[i], cth_max_exp_data[i], linestyle='dashed', color='black')
         for i, curR in enumerate(rbyR):
             plt.vlines(rbyR[i], cth_min_mt[i], cth_max_mt[i], linestyle='dashed', color='green')
             plt.vlines(rbyR[i], cth_min_lti[i], cth_max_lti[i], linestyle='dashed', color='blue')
             plt.vlines(rbyR[i], cth_min_hti[i], cth_max_hti[i], linestyle='dashed', color='red')

         plt.xlim([0.2,1.0])
         plt.ylim([0.0,4.0])
         plt.legend(loc='upper right')
         plt.xlabel('r/R')
         plt.ylabel('$C_{thrust}$')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

def compare_ctq_exp_data(C_tq_mt, C_tq_avg_mt, C_tq_lti, C_tq_avg_lti, C_tq_hti, C_tq_avg_hti, uinf):
    ctq_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/ctq_avg_u_' + 
                    str(int(uinf)) + '.txt', 'r')
    ctq_exp_data = ctq_exp.readlines()

    ctq_avg_exp_data = []
    ctq_min_exp_data = []
    ctq_max_exp_data = []
    for i, data in enumerate(ctq_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               ctq_avg_exp_data.append(float(j))           
        if (i == 1):
           min_data_string = data.split()
           for j in min_data_string:
               ctq_min_exp_data.append(float(j))           
        if (i == 2):
           max_data_string = data.split()
           for j in max_data_string:
               ctq_max_exp_data.append(float(j))           

    ctq_min_mt = []
    ctq_max_mt = []

    ctq_min_lti = []
    ctq_max_lti = []

    ctq_min_hti = []
    ctq_max_hti = []

    for iR, curR in enumerate(rLoc):
        ctq_min_mt.append(C_tq_avg_mt[iR] - np.std(C_tq_mt[:,iR]))
        ctq_max_mt.append(C_tq_avg_mt[iR] + np.std(C_tq_mt[:,iR]))
        ctq_min_lti.append(C_tq_avg_lti[iR] - np.std(C_tq_lti[:,iR]))
        ctq_max_lti.append(C_tq_avg_lti[iR] + np.std(C_tq_lti[:,iR]))
        ctq_min_hti.append(C_tq_avg_hti[iR] - np.std(C_tq_hti[:,iR]))
        ctq_max_hti.append(C_tq_avg_hti[iR] + np.std(C_tq_hti[:,iR]))

    with PdfPages('C_tq_comp_st_' + str(uinf) + '.pdf') as pfpgs:
         plt.figure()
         ms=200
         plt.plot(rbyR, ctq_avg_exp_data, marker='o', color='black', label='EXP')
         plt.scatter(rbyR, ctq_min_exp_data, marker='_', color='black', s=ms)
         plt.scatter(rbyR, ctq_max_exp_data, marker='_', color='black', s=ms)

         plt.plot(rbyR, C_tq_avg_mt, marker='v', color='green', label='MT (0.5%)')
         plt.scatter(rbyR, ctq_min_mt, marker='_', color='green')
         plt.scatter(rbyR, ctq_max_mt, marker='_', color='green')

         plt.plot(rbyR, C_tq_avg_lti, marker='x', color='blue', label='ST (0.5%)')
         plt.scatter(rbyR, ctq_min_lti, marker='_', color='blue')
         plt.scatter(rbyR, ctq_max_lti, marker='_', color='blue')

         plt.plot(rbyR, C_tq_avg_hti, marker='*', color='red', label='ST (6%)')
         plt.scatter(rbyR, ctq_min_hti, marker='_', color='red')
         plt.scatter(rbyR, ctq_max_hti, marker='_', color='red')

         for i, curR in enumerate(rbyR_exp):
             plt.vlines(rbyR_exp[i], ctq_min_exp_data[i], ctq_max_exp_data[i], linestyle='dashed', color='black')
         for i, curR in enumerate(rbyR):
             plt.vlines(rbyR[i], ctq_min_mt[i], ctq_max_mt[i], linestyle='dashed', color='green')
             plt.vlines(rbyR[i], ctq_min_lti[i], ctq_max_lti[i], linestyle='dashed', color='blue')
             plt.vlines(rbyR[i], ctq_min_hti[i], ctq_max_hti[i], linestyle='dashed', color='red')

         plt.xlim([0.2,1.0])
         plt.ylim([-0.1,1.5])
         plt.legend(loc='upper right')
         plt.xlabel('r/R')
         plt.ylabel('$C_{torque}$')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

def compare_cn_exp_data(C_n_avg_sst, C_n_avg_iddes, uinf):

    cn_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/cn_avg_u_' + 
                    str(int(uinf)) + '.txt', 'r')
    cn_exp_data = cn_exp.readlines()

    cn_avg_exp_data = []
    cn_min_exp_data = []
    cn_max_exp_data = []
    for i, data in enumerate(cn_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               cn_avg_exp_data.append(float(j))           
        if (i == 1):
           min_data_string = data.split()
           for j in min_data_string:
               cn_min_exp_data.append(float(j))           
        if (i == 2):
           max_data_string = data.split()
           for j in max_data_string:
               cn_max_exp_data.append(float(j))           

    with PdfPages('C_n_comp_' + str(uinf) + '.pdf') as pfpgs:
         plt.figure()
         plt.scatter(rbyR, cn_avg_exp_data, marker='o', color='black', label='Exp - Avg')
         plt.scatter(rbyR, cn_min_exp_data, marker='o', color='blue', label='Exp - Min.')
         plt.scatter(rbyR, cn_max_exp_data, marker='o', color='red', label='Exp - Max.')
         plt.scatter(rbyR, -C_n_avg_sst, marker='v', color='black', label='SST')
         plt.scatter(rbyR, C_n_avg_iddes, marker='x', color='black', label='IDDES')
         plt.vlines(rbyR[0], cn_min_exp_data[0], cn_max_exp_data[0])
         plt.vlines(rbyR[1], cn_min_exp_data[1], cn_max_exp_data[1])
         plt.vlines(rbyR[2], cn_min_exp_data[2], cn_max_exp_data[2])
         plt.vlines(rbyR[3], cn_min_exp_data[3], cn_max_exp_data[3])
         plt.legend(loc='upper right')
         plt.xlabel('r/R')
         plt.ylabel('$C_{normal}$') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()


def compare_ct_exp_data(C_t_avg_sst, C_t_avg_iddes, uinf):

    ct_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/ct_avg_u_' + 
                    str(int(uinf)) + '.txt', 'r')
    ct_exp_data = ct_exp.readlines()

    ct_avg_exp_data = []
    ct_min_exp_data = []
    ct_max_exp_data = []
    for i, data in enumerate(ct_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               ct_avg_exp_data.append(float(j))           
        if (i == 1):
           min_data_string = data.split()
           for j in min_data_string:
               ct_min_exp_data.append(float(j))           
        if (i == 2):
           max_data_string = data.split()
           for j in max_data_string:
               ct_max_exp_data.append(float(j))           

    with PdfPages('C_t_comp_' + str(uinf) + '.pdf') as pfpgs:
         plt.figure()
         plt.scatter(rbyR, ct_avg_exp_data, marker='o', color='black', label='Exp - Avg')
         plt.scatter(rbyR, ct_min_exp_data, marker='o', color='blue', label='Exp - Min.')
         plt.scatter(rbyR, ct_max_exp_data, marker='o', color='red', label='Exp - Max.')
         plt.scatter(rbyR, -C_t_avg_sst, marker='v', color='black', label='SST')
         plt.scatter(rbyR, C_t_avg_iddes, marker='x', color='black', label='IDDES')
         plt.vlines(rbyR[0], ct_min_exp_data[0], ct_max_exp_data[0])
         plt.vlines(rbyR[1], ct_min_exp_data[1], ct_max_exp_data[1])
         plt.vlines(rbyR[2], ct_min_exp_data[2], ct_max_exp_data[2])
         plt.vlines(rbyR[3], ct_min_exp_data[3], ct_max_exp_data[3])
         plt.legend(loc='upper right')
         plt.xlabel('r/R')
         plt.ylabel('$C_{tangential}$') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

##############################################
if __name__=="__main__":

    exo_file_mt = sys.argv[1]
    exo_file_hti = sys.argv[2]
    exo_file_lti = sys.argv[3]
    uinf = float(sys.argv[4])

    # Force coefficients for modeled turbulence (MT),
    # synthetic turbulence for low (LTI) and high (HTI) turbulent intensities
    # Returns: C_th vs. time, C_th_avg, C_tq vs. time, C_tq_avg,
    #          C_norm_avg, C_tang_avg at each radial station
    force_coeff_mt = get_forces(exo_file_mt, uinf, Nrev, NIt_per_rev_iddes)
    force_coeff_lti = get_forces(exo_file_lti, uinf, Nrev, NIt_per_rev_iddes)
    force_coeff_hti = get_forces(exo_file_hti, uinf, Nrev, NIt_per_rev_iddes)

    # MT:
    C_th_mt = force_coeff_mt[0]
    C_th_avg_mt = force_coeff_mt[1]
    C_tq_mt = force_coeff_mt[2]
    C_tq_avg_mt = force_coeff_mt[3]

    C_n_avg_mt = force_coeff_mt[4]
    C_t_avg_mt = force_coeff_mt[5]

    # LTI
    C_th_lti = force_coeff_lti[0]
    C_th_avg_lti = force_coeff_lti[1]
    C_tq_lti = force_coeff_lti[2]
    C_tq_avg_lti = force_coeff_lti[3]

    C_n_avg_lti = force_coeff_lti[4]
    C_t_avg_lti = force_coeff_lti[5]

    # HTI:
    C_th_hti = force_coeff_hti[0]
    C_th_avg_hti = force_coeff_hti[1]
    C_tq_hti = force_coeff_hti[2]
    C_tq_avg_hti = force_coeff_hti[3]

    C_n_avg_hti = force_coeff_hti[4]
    C_t_avg_hti = force_coeff_hti[5]

    print(C_th_avg_mt, C_th_avg_lti, C_th_avg_hti)
    print(C_tq_avg_mt, C_tq_avg_lti, C_tq_avg_hti)

    # Plots:
    compare_cth_exp_data(C_th_mt, C_th_avg_mt, C_th_lti, C_th_avg_lti, C_th_hti, C_th_avg_hti, uinf)
    compare_ctq_exp_data(C_tq_mt, C_tq_avg_mt, C_tq_lti, C_tq_avg_lti, C_tq_hti, C_tq_avg_hti, uinf)
