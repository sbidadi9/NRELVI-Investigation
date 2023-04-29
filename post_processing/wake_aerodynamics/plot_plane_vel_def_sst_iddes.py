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

def plot_velocity_deficit():

    nrows = 1
    ncols = 4 

    X = [1, 2, 3, 5, 7, 10, 14] 

    Y,Z = np.meshgrid(np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))

    Y = Y/d 
    Z = Z/d 

    with PdfPages('velocity_deficit_z0_iddes.pdf') as pfpgs:
         velocities = ['velocityx']
#         plt.rcParams["figure.figsize"] = [30, 7]
         plt.figure()
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]
             JMA = np.ones(Nyz)

             # Jensen's model:
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             JMA = JMA*JM
             
             # Bastankhah and Porté-Agel model:
             beta = 0.5*((1.0 + np.sqrt(1.0 - C_thrust))/np.sqrt(1.0 - C_thrust)) 
             eps = 0.25*np.sqrt(beta)
             sigma_by_d = k_bp*X[iplane] + eps
             Cx = 1.0 - np.sqrt(1.0 - C_thrust/(8.0*pow(sigma_by_d,2.0)))
             BPA = Cx * np.exp(-pow(np.absolute(Y[0,:]),2.0)/(2.0*pow(sigma_by_d,2.0)))             

             # CFD
             uavg_sst = get_mean_velocity('velocityx', iplane+1, 'SST')
             uavg_iddes = get_mean_velocity('velocityx', iplane, 'IDDES')

             uavg_sst_at_z0 = uavg_sst[:,44]
             uavg_iddes_at_z0 = uavg_iddes[:,44]

             u_def_sst = 1.0 - uavg_sst_at_z0
             u_def_iddes = 1.0 - uavg_iddes_at_z0
           
             sst_plot, = ax.plot(u_def_sst, Y[0,:], linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(u_def_iddes, Y[0,:], linewidth=2, label = 'IDDES')
             jm_plot, = ax.plot(JMA, Y[0,:], linewidth=2, color='red', label = 'Jensen (1983) model')
             bpa_plot, = ax.plot(BPA, Y[0,:], linewidth=2, label = 'BPA model') 
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([0, 0.7])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('y/d')
#             handles, labels = ax.get_legend_handles_labels()
#             fig.legend(handles, labels, loc = 'upper right')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

         i = 0
 #        plt.rcParams["figure.figsize"] = [30, 7]
         plt.figure() 
         fig, axs = plt.subplots(1, 3, squeeze=False) 
         for iplane in range(4,7):
             ax = axs[0, i]
             JMA = np.ones(Nyz)

             # Jensen's model:
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             JMA = JMA*JM

             # Bastankhah and Porté-Agel model:
             beta = 0.5*((1.0 + np.sqrt(1.0 - C_thrust))/np.sqrt(1.0 - C_thrust)) 
             eps = 0.25*np.sqrt(beta)
             sigma_by_d = k_bp*X[iplane] + eps
             Cx = 1.0 - np.sqrt(1.0 - C_thrust/(8.0*pow(sigma_by_d,2.0)))
             BPA = Cx * np.exp(-pow(np.absolute(Y[0,:]),2.0)/(2.0*pow(sigma_by_d,2.0)))             

             # CFD
             uavg_sst = get_mean_velocity('velocityx', iplane+1, 'SST')
             uavg_iddes = get_mean_velocity('velocityx', iplane, 'IDDES')

             uavg_sst_at_z0 = uavg_sst[:,44]
             uavg_iddes_at_z0 = uavg_iddes[:,44]

             u_def_sst = 1.0 - uavg_sst_at_z0
             u_def_iddes = 1.0 - uavg_iddes_at_z0
          
             sst_plot, = ax.plot(u_def_sst, Y[0,:], linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(u_def_iddes, Y[0,:], linewidth=2, label = 'IDDES')
             jm_plot, = ax.plot(JMA, Y[0,:], linewidth=2, color='red', label = 'Jensen (1983) model')
             bpa_plot, = ax.plot(BPA, Y[0,:], linewidth=2, label = 'Bastankhah and Porte-Agel (2014) model')  
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([0, 0.7])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('y/d')
             handles, labels = ax.get_legend_handles_labels()
             fig.legend(handles, labels, loc = 'lower right') 

             i = i + 1
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)


    with PdfPages('velocity_deficit_y0_iddes.pdf') as pfpgs:
         velocities = ['velocityx']
#         plt.rcParams["figure.figsize"] = [30, 7]
         plt.figure()  
         fig, axs = plt.subplots(1, 4, squeeze=False)
         for iplane in range(0, 4):
             ax = axs[0, iplane]
             JMA = np.ones(Nyz)

             # Jensen's model:
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             JMA = JMA*JM 

             # Bastankhah and Porté-Agel model:
             beta = 0.5*((1.0 + np.sqrt(1.0 - C_thrust))/np.sqrt(1.0 - C_thrust)) 
             eps = 0.25*np.sqrt(beta)
             sigma_by_d = k_bp*X[iplane] + eps
             Cx = 1.0 - np.sqrt(1.0 - C_thrust/(8.0*pow(sigma_by_d,2.0)))
             BPA = Cx * np.exp(-pow(np.absolute(Y[0,:]),2.0)/(2.0*pow(sigma_by_d,2.0)))             

             # CFD
             uavg_sst = get_mean_velocity('velocityx', iplane+1, 'SST')
             uavg_iddes = get_mean_velocity('velocityx', iplane, 'IDDES')

             uavg_sst_at_z0 = uavg_sst[44,:]
             uavg_iddes_at_z0 = uavg_iddes[44,:]

             u_def_sst = 1.0 - uavg_sst_at_z0
             u_def_iddes = 1.0 - uavg_iddes_at_z0
           
             sst_plot, = ax.plot(u_def_sst, Y[0,:], linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(u_def_iddes, Y[0,:], linewidth=2, label = 'IDDES')
             jm_plot, = ax.plot(JMA, Y[0,:], linewidth=2, color='red', label = 'Jensen (1983) model')
             bpa_plot, = ax.plot(BPA, Y[0,:], linewidth=2, label = 'BPA model')
             ax.set_title('x/d = {}'.format(X[iplane])) 
             ax.set_xlim([0, 0.7])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('y/d')
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

         i = 0
         plt.figure() 
         fig, axs = plt.subplots(1, 3, squeeze=False)  
         for iplane in range(4,7):
             ax = axs[0, i]
             JMA = np.ones(Nyz) 

             # Jensen's model:
             JM = (1.0 - np.sqrt(1.0 - C_thrust))/(1.0 + 2.0*k_t*X[iplane])
             JMA = JMA*JM  

             # Bastankhah and Porté-Agel model:
             beta = 0.5*((1.0 + np.sqrt(1.0 - C_thrust))/np.sqrt(1.0 - C_thrust)) 
             eps = 0.25*np.sqrt(beta)
             sigma_by_d = k_bp*X[iplane] + eps
             Cx = 1.0 - np.sqrt(1.0 - C_thrust/(8.0*pow(sigma_by_d,2.0)))
             BPA = Cx * np.exp(-pow(np.absolute(Y[0,:]),2.0)/(2.0*pow(sigma_by_d,2.0)))             

             # CFD
             uavg_sst = get_mean_velocity('velocityx', iplane+1, 'SST')
             uavg_iddes = get_mean_velocity('velocityx', iplane, 'IDDES')

             uavg_sst_at_z0 = uavg_sst[44,:]
             uavg_iddes_at_z0 = uavg_iddes[44,:]

             u_def_sst = 1.0 - uavg_sst_at_z0
             u_def_iddes = 1.0 - uavg_iddes_at_z0
           
             sst_plot, = ax.plot(u_def_sst, Y[0,:], linewidth=2, label = 'SST')
             iddes_plot, = ax.plot(u_def_iddes, Y[0,:], linewidth=2, label = 'IDDES')
             jm_plot, = ax.plot(JMA, Y[0,:], linewidth=2, color='red', label = 'Jensen (1983) model')
             bpa_plot, = ax.plot(BPA, Y[0,:], linewidth=2, label = 'Bastankhah and Porte-Agel (2014) model')  
             ax.set_title('x/d = {}'.format(X[iplane]))
             ax.set_xlim([0, 0.7])
             ax.set_ylim([-1.5,1.5])
             ax.set_xlabel('$\Delta \overline{u} / u_\infty$')
             ax.set_ylabel('y/d')
             handles, labels = ax.get_legend_handles_labels()
             fig.legend(handles, labels, loc = 'lower right') 

             i = i + 1
         plt.tight_layout()
         pfpgs.savefig()
         plt.close(fig)

if __name__=="__main__":

     plot_velocity_deficit()
#     plot_contours()
