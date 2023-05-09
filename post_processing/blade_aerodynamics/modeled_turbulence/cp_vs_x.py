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

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
from vtk.numpy_interface import dataset_adapter as dsa
import sys, os, glob, pickle

rho = 1.246
turbine_R = 5.029 
omega = 7.529
dr = 0.1 

rLoc = np.array([2.375, 3.0125, 3.6125, 4.25])
rLocp = np.array([30, 47, 63, 80])
chord_len = np.array([0.625, 0.56, 0.498, 0.434])
sec_pitch_angle = np.array([9.0, 6.0, 5.0, 4.0]) # In degrees

# Normalzed Cp vs. x at r/R = 0.3, 0.47, 0.63, 0.80
# The results are averaged over the last Nrev ('N' revolutions)
def get_cp_vs_x(exo_file, uinf, iR, curR, Nrev, NIt_per_rev):

    steps = Nrev*NIt_per_rev + 1; #Number of time steps

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

    # Merge blocks
    mergeblocks = MergeBlocks(Input=bladesexo)

    # Warp by vector
    warpByVector1 = WarpByVector(Input=mergeblocks)
    warpByVector1.Vectors = ['POINTS', 'mesh_displacement_']

    # Extract surface
    extractSurface1 = ExtractSurface(Input=warpByVector1)

    # Array of y-cord
    calculator1 = Calculator(Input=extractSurface1)
    calculator1.ResultArrayName = 'y_cord'
    calculator1.Function = 'coordsY'

    tsteps = bladesexo.TimestepValues    
    renderView1.ViewTime = tsteps[-1]
    bshow = Show(calculator1, renderView1)

    tstep_range = range(steps)
    curT_vec = tsteps[-steps:]

    Cp_mean = np.zeros(1000)
    ycord_mean = np.zeros(1000)
    max_y_points = np.zeros(1)

    j = 0
    it = 0
    while it in tstep_range: 

        renderView1.ViewTime = curT_vec[it]
        Render()
            
        # Slice
        slice1 = Slice(registrationName='Slice1', Input=calculator1)
        slice1.SliceType = 'Plane'
        slice1.HyperTreeGridSlicer = 'Plane'
        slice1.SliceOffsetValues = [0.0]
        slice1.SliceType.Origin = [0.0, 0.0, curR]
        slice1.HyperTreeGridSlicer.Origin = [0.0, 0.0, curR]
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]

        # Transform
        transform1 = Transform(registrationName='Transform1', Input=slice1)
        transform1.Transform = 'Transform'
        transform1.Transform.Rotate = [0.0, 0.0, -sec_pitch_angle[iR]]

        vtk_iv = servermanager.Fetch(transform1)
        numpy_iv = dsa.WrapDataObject(vtk_iv)
        pressure = numpy_iv.PointData.GetArray('pressure')
        ycord_time = numpy_iv.PointData.GetArray('y_cord')

        # Normalization
        ycord_time += abs(np.min(ycord_time))
        ycord_time = ycord_time / chord_len[iR]
 
        if (np.size(ycord_time) > max_y_points[0]):
           max_y_points[0] = np.size(ycord_time)

        ycord_time = np.pad(ycord_time, (0, 1000-np.size(ycord_time)), 'constant')

        Cp_time = pressure / (0.5*rho*(pow(uinf,2.0) + pow(curR*omega,2.0)))
        Cp_time = np.pad(Cp_time, (0, 1000-np.size(Cp_time)), 'constant')

        Cp_mean += Cp_time
        ycord_mean += ycord_time

        Delete(slice1)
        Delete(transform1)
        
        del slice1
        del transform1

        del vtk_iv
        del numpy_iv 

        it = it + NIt_per_rev
        j = j + 1

    Cp_mean /= j
    ycord_mean /= j
   
    N = 1000-max_y_points[0].astype(int)
    ycord_mean = ycord_mean[:-N]
    Cp_mean = Cp_mean[:-N]   

    return (ycord_mean, Cp_mean)
 

def plot_cp_vs_x(uinf, r_by_R, ycord_mean, Cp_mean):

    xbyc = []
    cp_avg_xp_span_exp = []
    cp_stdp_xp_span_exp = []
    cp_stdm_xp_span_exp = []

    cp_xp_span_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/Cp_u_' + 
                           str(int(uinf)) + '_' + str(r_by_R) + 'p_span.dat', 'r')
    cp_xp_span_exp_data = cp_xp_span_exp.readlines()

    for i, data in enumerate(cp_xp_span_exp_data):
        cp_data = data.split()
        xbyc.append(float(cp_data[0]))
        cp_avg_xp_span_exp.append(float(cp_data[1]))
        cp_stdp_xp_span_exp.append(float(cp_data[2]))
        cp_stdm_xp_span_exp.append(float(cp_data[3]))
        cp_xp_span_exp.close()    

    fig = plt.figure()

    with PdfPages('cp_vs_x_' + str(uinf) + '__' + str(r_by_R) + '.pdf') as pfpgs: 
         plt.scatter(ycord_mean,Cp_mean, label='Hybrid Solver')
         plt.scatter(xbyc, cp_avg_xp_span_exp, label='Experiment')
         plt.gca().invert_yaxis()
         plt.title('r/R = ' + str(r_by_R))
         plt.xlabel('x/c')
         plt.ylabel('$C_p$')
         plt.legend(loc=0)
         plt.tight_layout()
         pfpgs.savefig()    
         plt.close(fig)


##############################################

if __name__=="__main__":

    exo_file = sys.argv[1]
    uinf = float(sys.argv[2])

    Nrev = 25
    NIt_per_rev = 1440
    curR = 2.375

    for iR, curR in enumerate(rLoc):
        cp_vs_x = get_cp_vs_x(exo_file, uinf, iR, curR, Nrev, NIt_per_rev)
        plot_cp_vs_x(uinf, rLocp[iR], cp_vs_x[0], cp_vs_x[1])
        print(iR, curR, rLocp[iR])
