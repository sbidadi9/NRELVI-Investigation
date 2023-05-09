import matplotlib as mpl
import netCDF4 as nc
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['axes.titlesize'] = 34
mpl.rcParams['axes.labelsize'] = 34
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 15.0
#mpl.rcParams['figure.figsize'] = (6.328, 6.328)
mpl.rcParams["figure.figsize"] = [20, 7]
#plt.style.use('classic')

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

rho = 1.246
d = 10.058 

NPS_IDDES = 7
NPS_SST = 7
Nyz = 88

xmin = 0.0
xmax = 0.5

ymin = -1.2
ymax = 1.2

# SST
NInit_SST = 36000
NFinal_SST = 43200

# IDDES
#u_infty = 7.0
#NInit_IDDES = 23022 #53360 
#NFinal_IDDES = 28782 #62000 #73489 #62160
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_7_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_7_fine_mesh_far_wake/iddes_10r/post_processing/sampling_plane138340.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

#u_infty = 12.0
#NInit_IDDES = 57600
#NFinal_IDDES = 72100
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_12_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_12_fine_mesh_far_wake/iddes_30r/post_processing/sampling_plane00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

#u_infty = 15.0
#NInit_IDDES = 57600
#NFinal_IDDES = 72100
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_15_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_15_fine_mesh_far_wake/iddes_30r/post_processing/sampling_plane00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

u_infty = 20.0
NInit_IDDES = 0
NFinal_IDDES = 28800
sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_20_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_20_new/iddes_10r/post_processing/sampling_plane138340.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

NTS_IDDES = NFinal_IDDES - NInit_IDDES
NTS_SST = NFinal_SST - NInit_SST

###########################################################################################
#                                   Plane Plots
###########################################################################################

def get_mean_velocity(velocity, yz_plane, turb_model):
    """Returns average velocity on a yz plane for a given velocity component"""

    vel_mean = np.zeros((Nyz,Nyz))
    g_vel_mean = np.zeros((Nyz,Nyz))
    g_vel_mean_flatten = np.zeros(Nyz*Nyz)

    if (turb_model == 'SST'):
       vel = sp_sst["p_yz"][velocity]
       tstep_range = np.array_split( range(NInit_SST, NFinal_SST, 1), size)[rank]
    elif (turb_model == 'IDDES'):
       vel = sp_iddes["p_yz"][velocity]
       tstep_range = np.array_split( range(NInit_IDDES, NFinal_IDDES, 1), size)[rank]

    for i in tstep_range: # loop over time steps
        if (turb_model == 'SST'):
           vel_temp_tsi = vel[i,:].reshape(NPS_SST, Nyz, Nyz)
        elif (turb_model == 'IDDES'):
           vel_temp_tsi = vel[i,:].reshape(NPS_IDDES, Nyz, Nyz)

        vel_mean += vel_temp_tsi[yz_plane, :, :]
   
    vel_mean_flatten = vel_mean.reshape(Nyz*Nyz)

    comm.Reduce([vel_mean_flatten, MPI.DOUBLE], [g_vel_mean_flatten, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0):
        if (turb_model == 'SST'): 
           g_vel_mean_flatten /= NTS_SST 
        elif (turb_model == 'IDDES'):
           g_vel_mean_flatten /= NTS_IDDES 
 
    comm.Bcast([g_vel_mean_flatten, MPI.DOUBLE], root = 0)
        
    vel_mean = g_vel_mean_flatten.reshape(Nyz, Nyz) / u_infty

    return vel_mean

def plot_velocity_deficit():

    nrows = 1
    ncols = 5 
 
    Xtot = [1, 2, 3, 5, 7, 10, 14] 
    X = [1, 3, 5, 7, 10] 

    Y,Z = np.meshgrid(np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))

    Y = Y/d 
    Z = Z/d 

    with PdfPages('velocity_deficit_z0_' + str(int(u_infty)) + '.pdf') as pfpgs:
         velocities = ['velocityx']
         plt.figure()
         fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, squeeze=False)
         fig.add_subplot(111, frameon=False)
         for iplane in range(0, 5):
             ax = axs[0, iplane]
             JMA = np.ones(Nyz)

             index = Xtot.index(X[iplane])
 
             uavg_iddes = get_mean_velocity('velocityx', index, 'IDDES')
             uavg_sst = get_mean_velocity('velocityx', index, 'SST')
             uavg_iddes_at_z0 = uavg_iddes[:,44]
             uavg_sst_at_z0 = uavg_sst[:,44]
             u_def_iddes = 1.0 - uavg_iddes_at_z0
             u_def_sst = 1.0 - uavg_sst_at_z0

             sst_plot, = ax.plot(u_def_sst, Y[0,:], label = 'SST', color='blue')
             iddes_plot, = ax.plot(u_def_iddes, Y[0,:], label = 'IDDES', color='red')
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([xmin, xmax])
             ax.set_ylim([ymin,ymax])
         plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
         plt.xlabel('$\Delta \overline{u} / u_\infty$')
         plt.ylabel('y/d')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

if __name__=="__main__":
     plot_velocity_deficit()
