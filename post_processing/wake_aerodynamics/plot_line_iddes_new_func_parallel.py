import matplotlib.pyplot as plt
import netCDF4 as nc
from mpi4py import MPI
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['lines.linewidth'] = 2

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

u_infty = 7.0
d = 10.058
NTS = 28000
NP = 192

tstep_range = np.array_split( range(NTS), size)[rank]

sl = nc.Dataset('/projects/hfm/sbidadi/nrel_phase_vi/nrel_phase_vi_output/ti_calculations/u_7/iddes/sampling_line00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

###########################################################################################
#                                   Line Plots
###########################################################################################

def get_mean_velocity_line(velocity):
    """Returns average velocity on a line for a given velocity component"""

    vel_mean = np.zeros(NP)
    g_vel_mean = np.zeros(NP)

    vel = sl["line1"][velocity]

    for i in tstep_range: # loop over time steps
        vel_temp_tsi = vel[i,:]
     
        for j in range(np.size(vel_mean)):
            vel_mean[j] = vel_mean[j] + vel_temp_tsi[j]

    comm.Reduce([vel_mean, MPI.DOUBLE], [g_vel_mean, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0): 
        g_vel_mean /= NTS

    comm.Bcast([g_vel_mean, MPI.DOUBLE], root = 0)

    return g_vel_mean / u_infty

def get_variance_line(velocity):
    """Returns velocity variance on a line for a given velocity component"""

    vel_var = np.zeros(NP)
    g_vel_var = np.zeros(NP)

    vel = sl["line1"][velocity]  
    vel2 = vel
   
    vel_mean = get_mean_velocity_line(velocity) * u_infty

    for i in tstep_range: # loop over time steps

        vel_temp_tsi = vel[i,:]
        vel_var += (vel_temp_tsi - vel_mean) * (vel_temp_tsi - vel_mean) 

    comm.Reduce(  [vel_var, MPI.DOUBLE], [g_vel_var, MPI.DOUBLE], op = MPI.SUM, root = 0)
 
    if (rank == 0): 
        g_vel_var /= NTS
        print(g_vel_var)

    comm.Bcast([g_vel_var, MPI.DOUBLE], root = 0)

    return g_vel_var

    
def get_mean_tke_line():
    """Returns mean tke on a line"""

    tke_res_m = np.zeros(NP)
    tke_sgs_m = np.zeros(NP) 
    tke_sgs = sl['line1']['tke']

    g_tke_res_m = np.zeros(NP)
    g_tke_sgs_m = np.zeros(NP)

    uvar = get_variance_line("velocityx")
    vvar = get_variance_line("velocityy")
    wvar = get_variance_line("velocityz")

    # resolved TKE
    for i in range(np.size(uvar)):
        tke_res_m[i] = 0.5*(uvar[i] + vvar[i] + wvar[i])

    g_tke_res_m = tke_res_m

    # sgs TKE
    for i in tstep_range: # loop over time steps

        tke_sgs_temp_tsi = tke_sgs[i,:]
     
        for j in range(np.size(tke_sgs_temp_tsi)):
            tke_sgs_m[j] = tke_sgs_m[j] + tke_sgs_temp_tsi[j]

    comm.Reduce(  [tke_sgs_m, MPI.DOUBLE], [g_tke_sgs_m, MPI.DOUBLE], op = MPI.SUM, root = 0)
 
    if (rank == 0):
       g_tke_sgs_m /= NTS

    comm.Bcast([g_tke_sgs_m, MPI.DOUBLE], root = 0)
   
    return (np.array(g_tke_res_m) + np.array(g_tke_sgs_m))

def get_turb_intensity_line():
    """Returns turbulent intensity on a line"""

    turb_intensity = np.zeros(NP) 

    tke = get_mean_tke_line()

    for i in range(np.size(tke)):
        turb_intensity[i] = ((np.sqrt(2.0 * tke[i] / 3.0))/u_infty)*100.0

    return turb_intensity

def plot_contours_line():

    nrows = 3
    ncols = 1

    X = sl['line1']['coordinates'][:-1,0]
    X = X / d

    with PdfPages('mean_velocity_line.pdf') as pfpgs: 
         fig = plt.figure() 
         ux_mean = get_mean_velocity_line('velocityx')
         plt.plot(X, ux_mean[:-1])
         plt.title('mean velocityx along the centerline at z=0')
         plt.ylabel('$u_{mean}$/U')
         plt.xlabel('x/d')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

    with PdfPages('ti_line.pdf') as pfpgs: 
         fig = plt.figure() 
         ti = get_turb_intensity_line() 
         plt.plot(X, ti[:-1])
         plt.title('turbulent intensity along the centerline at z=0')
         plt.ylabel('TI(%)')
         plt.xlabel('x/d')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

if __name__=="__main__":

     plot_contours_line()
