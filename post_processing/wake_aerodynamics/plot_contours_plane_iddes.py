import matplotlib as mpl
import netCDF4 as nc
import mpi4py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
#from mpi4py import MPI

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
plt.style.use('classic')

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

u_infty = 20.0
d = 10.058
NTS = 28800
NPS = 7
Nyz = 88

tstep_range = np.array_split( range(NTS), size)[rank]

sp = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_20/iddes_30r/post_processing/sampling_plane49060.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

###########################################################################################
#                                   Plane Plots
###########################################################################################

def get_mean_velocity(velocity, yz_plane):
    """Returns average velocity on a yz plane for a given velocity component"""

    vel_mean = np.zeros((Nyz,Nyz))
    g_vel_mean = np.zeros((Nyz,Nyz))
    g_vel_mean_flatten = np.zeros(Nyz*Nyz)

    vel = sp["p_yz"][velocity]

    for i in tstep_range: # loop over time steps

        vel_temp_tsi = vel[i,:].reshape(NPS, Nyz, Nyz)
        vel_mean += vel_temp_tsi[yz_plane, :, :]
   
    vel_mean_flatten = vel_mean.reshape(Nyz*Nyz)

    comm.Reduce([vel_mean_flatten, MPI.DOUBLE], [g_vel_mean_flatten, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0): 
        g_vel_mean_flatten /= NTS 

    comm.Bcast([g_vel_mean_flatten, MPI.DOUBLE], root = 0)
        
    vel_mean = g_vel_mean_flatten.reshape(Nyz, Nyz) / u_infty

    return vel_mean

def get_variance(velocity, yz_plane):
    """Returns velocity variance on a yz plane for a given velocity component"""

    vel_var = np.zeros((Nyz,Nyz))
    vel_var_flatten = np.zeros((Nyz,Nyz))
    g_vel_var_flatten = np.zeros((Nyz,Nyz))

    vel = sp["p_yz"][velocity]

    vel_mean = get_mean_velocity(velocity, yz_plane) * u_infty

    for i in tstep_range: # loop over time steps
        vel_temp_tsi = vel[i,:].reshape(NPS, Nyz, Nyz)
        vel_var += (vel_temp_tsi[yz_plane,:,:] - vel_mean) * (vel_temp_tsi[yz_plane,:,:] - vel_mean) 

    vel_var_flatten = vel_var.reshape(Nyz*Nyz)

    comm.Reduce([vel_var_flatten, MPI.DOUBLE], [g_vel_var_flatten, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0): 
        g_vel_var_flatten /= NTS

    comm.Bcast([g_vel_var_flatten, MPI.DOUBLE], root = 0)

    vel_var = g_vel_var_flatten.reshape(Nyz,Nyz)
  
    return vel_var

def get_tke(yz_plane):
    """Returns velocity variance on a yz plane for a given velocity component"""

    tke_res_m = np.zeros((Nyz,Nyz))
    tke_sgs_m = np.zeros((Nyz,Nyz))

    tke_res_flatten_m = np.zeros(Nyz*Nyz)
    tke_sgs_flatten_m = np.zeros(Nyz*Nyz)

    g_tke_res_flatten_m = np.zeros(Nyz*Nyz)
    g_tke_sgs_flatten_m = np.zeros(Nyz*Nyz)

    g_tke_res_m = np.zeros((Nyz,Nyz))
    g_tke_sgs_m = np.zeros((Nyz,Nyz))

    uvar = get_variance('velocityx', yz_plane)
    vvar = get_variance('velocityy', yz_plane)
    wvar = get_variance('velocityz', yz_plane)

    # resolved TKE
    for i in range(np.size(uvar,axis=0)):
        for j in range(np.size(uvar,axis=0)):
            tke_res_m[i,j] = 0.5*(uvar[i,j] + vvar[i,j] + wvar[i,j])

    g_tke_res_m = tke_res_m

    # sgs TKE
    tke_sgs = sp['p_yz']['tke']

    for i in tstep_range: # loop over time steps

        tke_sgs_tsi = tke_sgs[i,:].reshape(NPS,Nyz,Nyz)

        for j in range(np.size(tke_sgs_tsi,axis=1)):
            for k in range(np.size(tke_sgs_tsi,axis=2)):
                tke_sgs_m[j,k] = tke_sgs_m[j,k] + tke_sgs_tsi[yz_plane,j,k]

    tke_sgs_flatten_m = tke_sgs_m.reshape(Nyz*Nyz)

    comm.Reduce(  [tke_sgs_flatten_m, MPI.DOUBLE], [g_tke_sgs_flatten_m, MPI.DOUBLE], op = MPI.SUM, root = 0)
 
    if (rank == 0): 
       g_tke_sgs_flatten_m /= NTS 

    comm.Bcast([g_tke_sgs_flatten_m, MPI.DOUBLE], root = 0)

    g_tke_sgs_m = g_tke_sgs_flatten_m.reshape(Nyz,Nyz)

    return (np.array(g_tke_res_m) + np.array(g_tke_sgs_m))
 
def get_turb_intensity(yz_plane):
    """Returns turbulent intensity on a yz plane"""

    turb_intensity = np.zeros((88,88))

    tke = get_tke(yz_plane)

    for i in range(np.size(turb_intensity,axis=0)):
        for j in range(np.size(turb_intensity,axis=0)):
               turb_intensity[i,j] = ((np.sqrt(2.0 * tke[i,j] / 3.0))/u_infty)*100.0

    return turb_intensity

def plot_velocity_deficit():

    nrows = 1 
    ncols = 4 

    X = [1, 2, 3, 5, 7, 10, 14] 

    Y,Z = np.meshgrid(np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))

    Y = Y/d 
    Z = Z/d 
    with PdfPages('velocity_deficit_z0_iddes.pdf') as pfpgs:
         velocities = ['velocityx']
         plt.rcParams["figure.figsize"] = [30, 7]
         fig, axs = plt.subplots(nrows, ncols, squeeze=False)
         for iplane in range(4):
             ax = axs[0, iplane]
             uavg = get_mean_velocity('velocityx', iplane)
             uavg_at_z0 = uavg[:,44]
             u_def = 1.0 - uavg_at_z0
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             if (rank == 0): 
                print(JM)
             ax.plot(u_def, Y[0,:], linewidth=2)
             ax.axvline(JM, linewidth=2, color='red')
             ax.legend(['x/d = {}'.format(X[iplane])])
             ax.set_xlim([0, 0.6])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('y/d')

         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

    with PdfPages('velocity_deficit_y0_iddes.pdf') as pfpgs:
         velocities = ['velocityx']
         plt.rcParams["figure.figsize"] = [30, 7]
         fig, axs = plt.subplots(nrows, ncols, squeeze=False)
         for iplane in range(4):
             ax = axs[0, iplane]
             uavg = get_mean_velocity('velocityx', iplane)
             uavg_at_y0 = uavg[44,:]
             u_def = 1.0 - uavg_at_y0
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             if (rank == 0):
                print(JM)
             ax.plot(u_def, Y[0,:], linewidth=2)
             ax.axvline(JM, linewidth=2, color='red')
             ax.legend(['x/d = {}'.format(X[iplane])])
             ax.set_xlim([0, 0.6])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('z/d')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

def plot_contours():

    nrows = 3
    ncols = 1

    X = [1, 2, 3, 5, 7, 10, 14]

    Y,Z = np.meshgrid( np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))
    Y = Y / d
    Z = Z / d

    with PdfPages('velocity_deficit_contour.pdf') as pfpgs:
         velocities = ['velocityx']
         plt.figure()
         plt.rcParams["figure.figsize"] = [30, 5]
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]
             uavg = get_mean_velocity('velocityx', iplane)  
             u_def = 1.0 - uavg
             u_def[u_def<0] = 0.0 
             levels = np.linspace(0.0, 0.2, 6)
             pcm = ax.contourf(Y, Z, u_def, levels=levels) 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set(xlabel='y/d', ylabel='z/d')
         fig.colorbar(pcm,ax=axs)
         pfpgs.savefig()
         plt.close(fig)

         i = 0
         plt.figure()
         plt.rcParams["figure.figsize"] = [20, 5]
         fig, axs = plt.subplots(1, 3, squeeze=False) 
         for iplane in range(4,7):
             ax = axs[0, i]
             uavg = get_mean_velocity('velocityx', iplane)  
             u_def = 1.0 - uavg
             u_def[u_def<0] = 0.0
             levels = np.linspace(-0.01, 0.2, 6)
             pcm = ax.contourf(Y, Z, u_def, levels=levels) 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set(xlabel='y/d', ylabel='z/d')
             i = i + 1
         fig.colorbar(pcm,ax=axs)
         pfpgs.savefig()
         plt.close(fig)

    with PdfPages('turbulent_intensity.pdf') as pfpgs:
         plt.rcParams["figure.figsize"] = [30, 5]
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]
             turb_intensity = get_turb_intensity(iplane) 
             levels = np.linspace(0.0, 12, 20)
             pcm = ax.contourf(Y, Z, turb_intensity, levels=levels) 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set(xlabel='y/d', ylabel='z/d')
         fig.colorbar(pcm,ax=axs)
         pfpgs.savefig()
         plt.close(fig)

         i = 0
         plt.figure()
         plt.rcParams["figure.figsize"] = [20, 5]
         fig, axs = plt.subplots(1, 3, squeeze=False) 
         for iplane in range(4,7):
             ax = axs[0, i]
             turb_intensity = get_turb_intensity(iplane) 
             levels = np.linspace(0.0, 12, 20)
             pcm = ax.contourf(Y, Z, turb_intensity, levels=levels) 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set(xlabel='y/d', ylabel='z/d')
             i = i + 1
         fig.colorbar(pcm,ax=axs)
         pfpgs.savefig()
         plt.close(fig)

if __name__=="__main__":

     plot_contours()
