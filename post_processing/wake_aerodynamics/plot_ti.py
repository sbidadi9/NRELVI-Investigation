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

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

rho = 1.246
d = 10.058

NPS_IDDES = 7
NPS_SST = 7
Nyz = 88

k_t = 0.1
k_bp = 0.025 

xmin = -1.2
xmax = 1.2

ymin = 0.0
ymax = 14.0 

crespo_const = 0.73

# SST
NInit_SST = 36000
NFinal_SST = 43200

# IDDES
# u_infty = 7
#NInit_IDDES = 23022 # 14382
#NFinal_IDDES = 28782 #73489 #62160
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_7_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_7_fine_mesh_far_wake/iddes_30r/post_processing/sampling_plane00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

# u_infty = 12.0
#NInit_IDDES = 57600
#NFinal_IDDES = 72100
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_12_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_12_fine_mesh_far_wake/iddes_30r/post_processing/sampling_plane00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

# u_infty = 15.0
#NInit_IDDES = 57600
#NFinal_IDDES = 72100
#sp_sst = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/sst/u_15_sst_fine_mesh_far_wake/post_processing/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
#sp_iddes = nc.Dataset('/scratch/sbidadi/nrel_vi/147s/iddes/u_15_fine_mesh_far_wake/iddes_30r/post_processing/sampling_plane00100.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

u_infty = 20.0
NInit_IDDES = 14400
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

def get_variance(velocity, yz_plane, turb_model):
    """Returns velocity variance on a yz plane for a given velocity component"""

    vel_var = np.zeros((Nyz,Nyz))
    vel_var_flatten = np.zeros((Nyz,Nyz))
    g_vel_var_flatten = np.zeros((Nyz,Nyz))

    vel_mean = get_mean_velocity(velocity, yz_plane, turb_model) * u_infty

    if (turb_model == 'SST'):
       tstep_range = np.array_split( range(NInit_SST, NFinal_SST, 1), size)[rank]
       vel = sp_sst["p_yz"][velocity]
    elif (turb_model == 'IDDES'):
       tstep_range = np.array_split( range(NInit_IDDES, NFinal_IDDES, 1), size)[rank]
       vel = sp_iddes["p_yz"][velocity]
 
    for i in tstep_range: # loop over time steps
        if (turb_model == 'SST'): 
           vel_temp_tsi = vel[i,:].reshape(NPS_SST, Nyz, Nyz)
        elif (turb_model == 'IDDES'): 
           vel_temp_tsi = vel[i,:].reshape(NPS_IDDES, Nyz, Nyz)

        vel_var += (vel_temp_tsi[yz_plane,:,:] - vel_mean) * (vel_temp_tsi[yz_plane,:,:] - vel_mean)

    vel_var_flatten = vel_var.reshape(Nyz*Nyz)

    comm.Reduce([vel_var_flatten, MPI.DOUBLE], [g_vel_var_flatten, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0):
        if (turb_model == 'SST'): 
           g_vel_var_flatten /= NTS_SST 
        elif (turb_model == 'IDDES'):
           g_vel_var_flatten /= NTS_IDDES 

    comm.Bcast([g_vel_var_flatten, MPI.DOUBLE], root = 0)

    vel_var = g_vel_var_flatten.reshape(Nyz,Nyz)

    return vel_var

def get_tke(yz_plane, turb_model):
    """Returns velocity variance on a yz plane for a given velocity component"""

    tke_res_m = np.zeros((Nyz,Nyz))
    tke_sgs_m = np.zeros((Nyz,Nyz))

    tke_res_flatten_m = np.zeros(Nyz*Nyz)
    tke_sgs_flatten_m = np.zeros(Nyz*Nyz)

    g_tke_res_flatten_m = np.zeros(Nyz*Nyz)
    g_tke_sgs_flatten_m = np.zeros(Nyz*Nyz)

    g_tke_res_m = np.zeros((Nyz,Nyz))
    g_tke_sgs_m = np.zeros((Nyz,Nyz))

    uvar = get_variance('velocityx', yz_plane, turb_model)
    vvar = get_variance('velocityy', yz_plane, turb_model)
    wvar = get_variance('velocityz', yz_plane, turb_model)

    # resolved TKE
    for i in range(np.size(uvar,axis=0)):
        for j in range(np.size(uvar,axis=0)):
            tke_res_m[i,j] = 0.5*(uvar[i,j] + vvar[i,j] + wvar[i,j])

    g_tke_res_m = tke_res_m

    if (turb_model == 'SST'):
       tstep_range = np.array_split( range(NInit_SST, NFinal_SST, 1), size)[rank]
       tke_sgs = sp_sst['p_yz']['tke']
    elif (turb_model == 'IDDES'):
       tstep_range = np.array_split( range(NInit_SST, NFinal_IDDES, 1), size)[rank]
       tke_sgs = sp_iddes['p_yz']['tke']
 
    for i in tstep_range: # loop over time steps
        if (turb_model == 'SST'):  
           tke_sgs_tsi = tke_sgs[i,:].reshape(NPS_SST,Nyz,Nyz)
        elif (turb_model == 'IDDES'):  
           tke_sgs_tsi = tke_sgs[i,:].reshape(NPS_IDDES,Nyz,Nyz)

        for j in range(np.size(tke_sgs_tsi,axis=1)):
            for k in range(np.size(tke_sgs_tsi,axis=2)):
                tke_sgs_m[j,k] = tke_sgs_m[j,k] + tke_sgs_tsi[yz_plane,j,k]

    tke_sgs_flatten_m = tke_sgs_m.reshape(Nyz*Nyz)

    comm.Reduce(  [tke_sgs_flatten_m, MPI.DOUBLE], [g_tke_sgs_flatten_m, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank == 0):
        if (turb_model == 'SST'): 
           g_tke_sgs_flatten_m /= NTS_SST 
        elif (turb_model == 'IDDES'):
           g_tke_sgs_flatten_m /= NTS_IDDES 

    comm.Bcast([g_tke_sgs_flatten_m, MPI.DOUBLE], root = 0)

    g_tke_sgs_m = g_tke_sgs_flatten_m.reshape(Nyz,Nyz)

    if (turb_model == 'SST'):
       return np.array(g_tke_sgs_m)
    elif (turb_model == 'IDDES'):
       return (np.array(g_tke_res_m) + np.array(g_tke_sgs_m))
 
def get_turb_intensity(yz_plane, turb_model):
    """Returns turbulent intensity on a yz plane"""

    turb_intensity = np.zeros((88,88))

    tke = get_tke(yz_plane, turb_model)

    for i in range(np.size(turb_intensity,axis=0)):
        for j in range(np.size(turb_intensity,axis=0)):
               turb_intensity[i,j] = ((np.sqrt(2.0 * tke[i,j] / 3.0))/u_infty)*100.0

    return turb_intensity

def plot_ti():

    nrows = 1
    ncols = 4 

    Xtot = [1, 2, 3, 5, 7, 10, 14] 
    X = [1, 3, 5, 7, 10] 

    Y,Z = np.meshgrid(np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))

    Y = Y/d 
    Z = Z/d 

    a_in = 0.1
    ti_up = 5

    with PdfPages('ti_z0_' + str(int(u_infty)) + '.pdf') as pfpgs:
         plt.figure()
         fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, squeeze=False)
         fig.add_subplot(111, frameon=False)
         for iplane in range(0, 5):
             ax = axs[0, iplane]
             ti_ich_a = np.ones(Nyz)

             # Crespo and Hernandez:
             ti_ich = 100*crespo_const*pow(a_in, 0.83)*pow(ti_up, 0.03)*pow(X[iplane],-0.32)
             ti_ich_a = ti_ich_a*ti_ich

             # CFD
             index = Xtot.index(X[iplane])
             ti_iddes = get_turb_intensity(index, 'IDDES')
             ti_sst = get_turb_intensity(index, 'SST')

             ti_iddes_at_z0 = ti_iddes[:,44]
             ti_sst_at_z0 = ti_sst[:,44]

             sst_plot, = ax.plot(Y[0,:], ti_sst_at_z0, label = 'SST', color='blue')          
             iddes_plot, = ax.plot(Y[0,:], ti_iddes_at_z0, label = 'IDDES', color='red')
             ti_ich_a, = ax.plot(Y[0,:], ti_ich_a, label = 'Crespo-Hernandez model', color='black')
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([xmin, xmax])
             ax.set_ylim([ymin,ymax])
         plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
         plt.xlabel('y/d')
         plt.ylabel('TI (%)')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

if __name__=="__main__":

     plot_ti()
