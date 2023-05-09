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
#import mpi4py
#from mpi4py import MPI

#########################################################################################
# Script compares thrust and torque coefficients for modeled turbulent intensity of 0.5%, 
# and synthetic turbulence intensities of 0.5% and 6.0%.
#########################################################################################

##################################################################################################################
# User specified parameters:
rho = 1.246
dr = 0.1

turbine_R = 5.029 
omega = 7.529

wspeed = np.array([7, 12, 15, 20])
rLoc = np.array([1.51, 2.343, 3.172, 4.023])
rbyR = np.array([0.3, 0.47, 0.63, 0.80])
chord_len = np.array([0.711, 0.627, 0.543, 0.457])
sec_pitch_angle = np.array([19.0, 10.0, 6.0, 4.0])

# Note: User has to specify the initial and final iteration values in compare_torque_exp_data()
# and compare_thrust_exp_data() functions.
########################################################################################################################
#                                 Thrust and Torque vs. Wind Speed
########################################################################################################################
class CasePostprocess:
    """ Read and post-process information from an Exawind solver run
    """

    def __init__(self, case_dir):
        force_file = case_dir+"/forces.dat"
        rot_time = 2.0 * np.pi / omega
        data = pd.read_csv(force_file, sep="\s+") 
        self.thrust = data['Fpx']+data['Fvx']
        self.torque = -data['Mtx']
        self.time = data['Time']

def compare_torque_exp_data():

    tq_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/torque.txt', 'r')
    tq_exp_data = tq_exp.readlines()

    # Modeled Turbulence
    NInit = 57600
    NFinal = 86400
    Nsteps = 28800 #13821
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7_fine_mesh_far_wake/iddes_30r', iddes_case_dir+'/u_12_fine_mesh_far_wake/iddes_30r', iddes_case_dir+'/u_15_fine_mesh_far_wake/iddes_30r', iddes_case_dir+'/u_20_new/iddes_10r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    tq_iddes_mt = []
    tq_iddes_smin_mt = []
    tq_iddes_smax_mt = []
    std_mt = []
    for i,c in enumerate(cases_iddes):
        if (i == 3):
           avg = np.average(c.torque.iloc[-Nsteps:])
           std = np.std(c.torque.iloc[-Nsteps:])
           std_mt.append(std)
        else:   
           avg = np.average(c.torque.iloc[NInit:NFinal])
           std = np.std(c.torque.iloc[NInit:NFinal])
           std_mt.append(std)
        tq_iddes_mt.append(avg)
        tq_iddes_smin_mt.append(avg-std)
        tq_iddes_smax_mt.append(avg+std)

    # High TI case:
    NInit = [1000, 1000, 1000, 1000]
    NFinal = [57600, 57600, 57600, 57600]
    Nsteps = np.subtract(NFinal, NInit)
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7_st_cm_imp/iddes_20r', iddes_case_dir+'/u_12_st_cm_imp/iddes_20r', iddes_case_dir+'/u_15_st_cm_imp/iddes_20r', iddes_case_dir+'/u_20_st_cm_imp/iddes_20r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    tq_iddes_hti = []
    tq_iddes_smin_hti = []
    tq_iddes_smax_hti = []
    std_hti = []
    for i,c in enumerate(cases_iddes):
        NI = NInit[i]
        NF = NFinal[i]
        avg = np.average(c.torque.iloc[NI:NF])
        print(avg)
        std = np.std(c.torque.iloc[NI:NF])
        std_hti.append(std)
        tq_iddes_hti.append(avg)
        tq_iddes_smin_hti.append(avg-std)
        tq_iddes_smax_hti.append(avg+std) 

    # Low TI case:
    NInit = [1000, 1000, 1000, 1000]
    NFinal = [57600, 57600, 57600, 57600]
    Nsteps = np.subtract(NFinal, NInit)
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_12_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_15_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_20_st_cm_low_ti_imp/iddes_20r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    tq_iddes_lti = []
    tq_iddes_smin_lti = []
    tq_iddes_smax_lti = []
    std_lti = []
    for i,c in enumerate(cases_iddes):
        NI = NInit[i]
        NF = NFinal[i]
        avg = np.average(c.torque.iloc[NI:NF])
        print(avg)
        std = np.std(c.torque.iloc[NI:NF])
        std_lti.append(std)
        tq_iddes_lti.append(avg)
        tq_iddes_smin_lti.append(avg-std)
        tq_iddes_smax_lti.append(avg+std) 
        
    tq_avg_exp_data = []
    tq_stdmin_exp_data = []
    tq_stdmax_exp_data = []
    for i, data in enumerate(tq_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               tq_avg_exp_data.append(float(j))           
        if (i == 1):
           stdmin_data_string = data.split()
           for j in stdmin_data_string:
               tq_stdmin_exp_data.append(float(j))           
        if (i == 2):
           stdmax_data_string = data.split()
           for j in stdmax_data_string:
               tq_stdmax_exp_data.append(float(j))           

    std_exp = np.subtract(tq_avg_exp_data, tq_stdmin_exp_data) 

    with PdfPages('torque_std_st_imp.pdf') as pfpgs:
         plt.figure()
         ms=200
         plt.plot(wspeed, std_exp, marker='o', color='black', label='Experiment')
         plt.plot(wspeed, std_mt, marker='o', color='green', label='IDDES (MT - TI=0.5%)')
         plt.plot(wspeed, std_lti, marker='o', color='blue', label='IDDES (ST - TI=0.5%)')
         plt.plot(wspeed, std_hti, marker='o', color='red', label='IDDES (ST - TI=6%)')

         plt.legend(loc='lower right')
         plt.minorticks_on()
         plt.xticks([7, 12, 15, 20])
         plt.xlabel('Wind Speed (m/s)')
         plt.ylabel('$\sigma_{torque}$ (N)') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

    with PdfPages('torque_comp_st_imp.pdf') as pfpgs:
         plt.figure()
         ms = 200
         plt.scatter(wspeed, tq_avg_exp_data, marker='o', color='black', label='Experiment')
         plt.scatter(wspeed, tq_stdmin_exp_data, marker='_', color='black', s=ms)
         plt.scatter(wspeed, tq_stdmax_exp_data, marker='_', color='black', s=ms)
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], tq_stdmin_exp_data[i], tq_stdmax_exp_data[i], color='black')
 
         print(tq_avg_exp_data)

         plt.plot(wspeed, tq_iddes_mt, marker='o', color='green', label='IDDES (MT - TI=0.5%)')
         plt.scatter(wspeed, tq_iddes_smin_mt, marker='_', color='green')
         plt.scatter(wspeed, tq_iddes_smax_mt, marker='_', color='green') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], tq_iddes_smin_mt[i], tq_iddes_smax_mt[i], linestyle='dashed', color='green')

         plt.plot(wspeed, tq_iddes_lti, marker='o', color='blue', label='IDDES (ST - TI=0.5%)')
         plt.scatter(wspeed, tq_iddes_smin_lti, marker='_', color='blue')
         plt.scatter(wspeed, tq_iddes_smax_lti, marker='_', color='blue') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], tq_iddes_smin_lti[i], tq_iddes_smax_lti[i], linestyle='dashed', color='blue')

         plt.plot(wspeed, tq_iddes_hti, marker='o', color='red', label='IDDES (ST - TI=6%)')
         plt.scatter(wspeed, tq_iddes_smin_hti, marker='_', color='red')
         plt.scatter(wspeed, tq_iddes_smax_hti, marker='_', color='red') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], tq_iddes_smin_hti[i], tq_iddes_smax_hti[i], linestyle='dashed', color='red')

         plt.legend(loc='lower right')
         plt.minorticks_on()
         plt.ylim([600,1700])
         plt.xticks([7, 12, 15, 20])
         plt.xlabel('Wind Speed (m/s)')
         plt.ylabel('Low-Speed Shaft Torque (N-m)') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

def compare_thrust_exp_data():

    th_exp = open('/projects/hfm/sbidadi/nrel_phase_vi/NREL_Phase_6_Exp_Data/Sequence_S/extracted_data/thrust.txt', 'r')
    th_exp_data = th_exp.readlines()

    # MT case:
    NInit = 57600
    NFinal = 72000 
    Nsteps = 28800 
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7/iddes_30r', iddes_case_dir+'/u_12/iddes_30r', iddes_case_dir+'/u_15/iddes_30r', iddes_case_dir+'/u_20_new/iddes_10r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    th_iddes_mt = []
    th_iddes_smin_mt = []
    th_iddes_smax_mt = []
    std_mt = []
    for i,c in enumerate(cases_iddes):
        if (i == 3):
           avg = np.average(c.thrust.iloc[-Nsteps:])
           std = np.std(c.thrust.iloc[-Nsteps:])
           std_mt.append(std)
        else: 
           avg = np.average(c.thrust.iloc[NInit:NFinal])
           std = np.std(c.thrust.iloc[NInit:NFinal])
           std_mt.append(std)
        th_iddes_mt.append(avg)
        th_iddes_smin_mt.append(avg-std)
        th_iddes_smax_mt.append(avg+std)
 
    # High TI case:
    NInit = [1000, 1000, 1000, 1000]
    NFinal = [57600, 57600, 57600, 57600]
    Nsteps = np.subtract(NFinal, NInit)
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7_st_cm_imp/iddes_20r', iddes_case_dir+'/u_12_st_cm_imp/iddes_20r', iddes_case_dir+'/u_15_st_cm_imp/iddes_20r', iddes_case_dir+'/u_20_st_cm_imp/iddes_20r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    th_iddes_hti = []
    th_iddes_smin_hti = []
    th_iddes_smax_hti = []
    std_hti = []
    for i,c in enumerate(cases_iddes):
        avg = np.average(c.thrust.iloc[NInit[i]:NFinal[i]])
        std = np.std(c.thrust.iloc[NInit[i]:NFinal[i]])
        th_iddes_hti.append(avg)
        th_iddes_smin_hti.append(avg-std)
        th_iddes_smax_hti.append(avg+std)
        std_hti.append(std)

    # Low TI case:
    NInit = [1000, 1000, 1000, 1000]
    NFinal = [57600, 57600, 57600, 57600]
    Nsteps = np.subtract(NFinal, NInit)
    iddes_case_dir = '/scratch/sbidadi/nrel_vi/147s/iddes'
    case_dir_iddes = [iddes_case_dir+'/u_7_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_12_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_15_st_cm_low_ti_imp/iddes_20r', iddes_case_dir+'/u_20_st_cm_low_ti_imp/iddes_20r'] 
    cases_iddes = [CasePostprocess(c) for c in case_dir_iddes]  

    th_iddes_lti = []
    th_iddes_smin_lti = []
    th_iddes_smax_lti = []
    std_lti = []
    for i,c in enumerate(cases_iddes):
        avg = np.average(c.thrust.iloc[NInit[i]:NFinal[i]])
        std = np.std(c.thrust.iloc[NInit[i]:NFinal[i]])
        th_iddes_lti.append(avg)
        th_iddes_smin_lti.append(avg-std)
        th_iddes_smax_lti.append(avg+std)
        std_lti.append(std) 

    th_avg_exp_data = []
    th_stdmin_exp_data = []
    th_stdmax_exp_data = []
    for i, data in enumerate(th_exp_data):
        if (i == 0):
           avg_data_string = data.split()
           for j in avg_data_string:
               th_avg_exp_data.append(float(j))           
        if (i == 1):
           stdmin_data_string = data.split()
           for j in stdmin_data_string:
               th_stdmin_exp_data.append(float(j))           
        if (i == 2):
           stdmax_data_string = data.split()
           for j in stdmax_data_string:
               th_stdmax_exp_data.append(float(j))           

    std_exp = np.subtract(th_avg_exp_data, th_stdmin_exp_data)

    with PdfPages('thrust_std_st_imp.pdf') as pfpgs:
         plt.figure()
         ms=200
         plt.plot(wspeed, std_exp, marker='o', color='black', label='Experiment')
         plt.plot(wspeed, std_mt, marker='o', color='green', label='IDDES (MT - TI=0.5%)')
         plt.plot(wspeed, std_lti, marker='o', color='blue', label='IDDES (ST - TI=0.5%)')
         plt.plot(wspeed, std_hti, marker='o', color='red', label='IDDES (ST - TI=6%)')

         plt.legend(loc='upper left')
         plt.minorticks_on()
         plt.xticks([7, 12, 15, 20])
         plt.xlabel('Wind Speed (m/s)')
         plt.ylabel('$\sigma_{thrust}$ (N)') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

    with PdfPages('thrust_comp_st_imp.pdf') as pfpgs:
         plt.figure()
         ms=200
         plt.scatter(wspeed, th_avg_exp_data, marker='o', color='black', label='Experiment')
         plt.scatter(wspeed, th_stdmin_exp_data, marker='_', color='black', s=ms)
         plt.scatter(wspeed, th_stdmax_exp_data, marker='_', color='black', s=ms)
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], th_stdmin_exp_data[i], th_stdmax_exp_data[i], color='black')
 
         print(th_avg_exp_data)

         print(th_iddes_mt)
         plt.plot(wspeed, th_iddes_mt, marker='o', color='green', label='IDDES (MT - TI=0.5%)')
         plt.scatter(wspeed, th_iddes_smin_mt, marker='_', color='green')
         plt.scatter(wspeed, th_iddes_smax_mt, marker='_', color='green') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], th_iddes_smin_mt[i], th_iddes_smax_mt[i], linestyle='dashed', color='green') 

         print(th_iddes_lti) 
         plt.plot(wspeed, th_iddes_lti, marker='o', color='blue', label='IDDES (ST - TI=0.5%)')
         plt.scatter(wspeed, th_iddes_smin_lti, marker='_', color='blue')
         plt.scatter(wspeed, th_iddes_smax_lti, marker='_', color='blue') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], th_iddes_smin_lti[i], th_iddes_smax_lti[i], linestyle='dashed', color='blue') 

         print(th_iddes_hti) 
         plt.plot(wspeed, th_iddes_hti, marker='o', color='red', label='IDDES (ST - TI=6%)')
         plt.scatter(wspeed, th_iddes_smin_hti, marker='_', color='red')
         plt.scatter(wspeed, th_iddes_smax_hti, marker='_', color='red') 
         for i, ws in enumerate(wspeed):
             plt.vlines(wspeed[i], th_iddes_smin_hti[i], th_iddes_smax_hti[i], linestyle='dashed', color='red') 

         plt.legend(loc='upper left')
         plt.minorticks_on()
         plt.ylim([1000,5000])
         plt.xticks([7, 12, 15, 20])
         plt.xlabel('Wind Speed (m/s)')
         plt.ylabel('Thrust (N)') 
         plt.tight_layout()
         pfpgs.savefig()
         plt.close()

##############################################

if __name__=="__main__":

    compare_thrust_exp_data()
    compare_torque_exp_data()

