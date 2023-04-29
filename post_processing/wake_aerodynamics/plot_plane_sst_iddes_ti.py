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
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 15.0
#mpl.rcParams['figure.figsize'] = (6.328, 6.328)
mpl.rcParams["figure.figsize"] = [30, 7]
#plt.style.use('classic')

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs

u_infty = 7.0
d = 10.058
NTS_IDDES = 60000
NTS_SST = 14400

NPS_IDDES = 7
NPS_SST = 8
Nyz = 88

k_t = 0.1
k_bp = 0.025 
C_thrust = 0.75

sp_sst = nc.Dataset('/projects/hfm/sbidadi/nrel_phase_vi/nrel_phase_vi_output/ti_calculations/u_7/sst/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())
sp_iddes = nc.Dataset('/projects/hfm/sbidadi/nrel_phase_vi/nrel_phase_vi_output/ti_calculations/u_7/iddes/sampling_plane43200.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

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
       tstep_range = np.array_split( range(NTS_SST, -1, -1), size)[rank]
    elif (turb_model == 'IDDES'):
       vel = sp_iddes["p_yz"][velocity]
       tstep_range = np.array_split( range(NTS_IDDES, -1, -1), size)[rank]
 
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
       tstep_range = np.array_split( range(NTS_SST, -1, -1), size)[rank]
       vel = sp_sst["p_yz"][velocity]
    elif (turb_model == 'IDDES'):
       tstep_range = np.array_split( range(NTS_IDDES, -1, -1), size)[rank]
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
       tstep_range = np.array_split( range(NTS_SST, -1, -1), size)[rank]
       tke_sgs = sp_sst['p_yz']['tke']
    elif (turb_model == 'IDDES'):
       tstep_range = np.array_split( range(NTS_IDDES, -1, -1), size)[rank]
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

    X = [1, 2, 3, 5, 7, 10, 14] 

    Y,Z = np.meshgrid(np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))

    Y = Y/d 
    Z = Z/d 

    a_in = 0.1
    ti_up = 5

    with PdfPages('ti_z0_iddes.pdf') as pfpgs:
         plt.figure()
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]
             ti_ich_a = np.ones(Nyz)

             # Crespo and Hernandez:
             ti_ich = 100*0.73*pow(a_in, 0.8325)*pow(ti_up, 0.0325)*pow(X[iplane],-0.32)
             ti_ich_a = ti_ich_a*ti_ich

             # Xie and Archer:
             # ti_xa = 5.7*pow(C_thrust, 0.5)*pow(ti_up, 0.68)*  
             
             # CFD
             ti_sst = get_turb_intensity(iplane+1, 'SST')
             ti_iddes = get_turb_intensity(iplane, 'IDDES')

             ti_sst_at_z0 = ti_sst[:,44]
             ti_iddes_at_z0 = ti_iddes[:,44]
          
             sst_plot, = ax.plot(Y[0,:], ti_sst_at_z0, linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(Y[0,:], ti_iddes_at_z0, linewidth=2, label = 'IDDES')
             ti_ich_a, = ax.plot(Y[0,:], ti_ich_a, linewidth=2, color='red', label = 'Crespo-Hernandez model')
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([-1.5, 1.5])
             ax.set_ylim([0.0,15.0])
             ax.set_xlabel('y/d')
             ax.set_ylabel('TI (%)')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

         i = 0
         plt.figure() 
         fig, axs = plt.subplots(1, 3, squeeze=False) 
         for iplane in range(4,7):
             ax = axs[0, i]
             ti_ich_a = np.ones(Nyz)

             # Crespo and Hernandez:
             ti_ich = 100*0.73*pow(a_in, 0.8325)*pow(ti_up, 0.0325)*pow(X[iplane],-0.32)
             ti_ich_a = ti_ich_a*ti_ich

             # CFD
             ti_sst = get_turb_intensity(iplane+1, 'SST')
             ti_iddes = get_turb_intensity(iplane, 'IDDES')

             ti_sst_at_z0 = ti_sst[:,44]
             ti_iddes_at_z0 = ti_iddes[:,44]

             sst_plot, = ax.plot(Y[0,:], ti_sst_at_z0, linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(Y[0,:], ti_iddes_at_z0, linewidth=2, label = 'IDDES')
             ti_ich_a, = ax.plot(Y[0,:], ti_ich_a, linewidth=2, color='red', label = 'Crespo-Hernandez model') 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([-1.5, 1.5])
             ax.set_ylim([0.0,15.0]) 
#             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('y/d')
             ax.set_ylabel('TI (%)')
             handles, labels = ax.get_legend_handles_labels()
             fig.legend(handles, labels, loc = 'lower right') 
             i = i + 1
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)


    with PdfPages('ti_y0_iddes.pdf') as pfpgs:
         velocities = ['velocityx']
         plt.figure()  
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]

             # CFD
             ti_sst = get_turb_intensity(iplane+1, 'SST')
             ti_iddes = get_turb_intensity(iplane, 'IDDES')

             ti_sst_at_y0 = ti_sst[44,:]
             ti_iddes_at_y0 = ti_iddes[44,:]
         
             sst_plot, = ax.plot(Y[0,:], ti_sst_at_y0, linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(Y[0,:], ti_iddes_at_y0, linewidth=2, label = 'IDDES')
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([-1.5, 1.5])
             ax.set_ylim([0.0,15.0])
#             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('z/d')
             ax.set_ylabel('TI (%)')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

         i = 0
         plt.figure() 
         fig, axs = plt.subplots(1, 3, squeeze=False)  
         for iplane in range(4,7):
             ax = axs[0, i]

             # CFD
             ti_sst = get_turb_intensity(iplane+1, 'SST')
             ti_iddes = get_turb_intensity(iplane, 'IDDES')

             ti_sst_at_y0 = ti_sst[44,:]
             ti_iddes_at_y0 = ti_iddes[44,:]
          
             sst_plot, = ax.plot(Y[0,:], ti_sst_at_y0, linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(Y[0,:], ti_iddes_at_y0, linewidth=2, label = 'IDDES')
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([-1.5, 1.5])
             ax.set_ylim([0.0,15.0]) 
#             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('z/d')
             ax.set_ylabel('TI (%)')
             handles, labels = ax.get_legend_handles_labels()
             fig.legend(handles, labels, loc = 'lower right') 

             i = i + 1
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)


if __name__=="__main__":

     plot_ti()
