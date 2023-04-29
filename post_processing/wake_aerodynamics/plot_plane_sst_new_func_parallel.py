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

mpl.rcParams['lines.linewidth'] = 2

plt.style.use('classic')

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size # Total number of procs


u_infty = 7.0
d = 10.058
NTS = 10
NPS = 8
Nyz = 88

tstep_range = np.array_split( range(NTS), size)[rank]

sp = nc.Dataset('/projects/hfm/sbidadi/nrel_phase_vi/nrel_phase_vi_output/ti_calculations/u_7/sst/sampling_plane00000.nc', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info())

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

def plot_contours():

    nrows = 4
    ncols = 1

    X = [1, 2, 3, 5, 7, 10, 14]

    Y,Z = np.meshgrid( np.linspace(-17.5, 17.5, 88), np.linspace(-17.5, 17.5, 88))
    Y = Y / d
    Z = Z / d

    with PdfPages('velocity_contour_plots_1-4.pdf') as pfpgs:
         velocities = ['velocityx']
         for iplane in range(3):
             plt.rcParams["figure.autolayout"] = True

             fig, axs = plt.subplots(nrows, ncols, squeeze=False)
             for col in range(ncols):
                 for row, velname in enumerate(velocities):
                     ax = axs[row, col]
                     uavg = get_mean_velocity(velname, iplane)
                     pcm = ax.contourf(Y, Z, uavg)
                     cbar = fig.colorbar(pcm, ax=ax)
                     tick_font_size = 5
                     cbar.ax.tick_params(labelsize=tick_font_size)
                     ax.set_title('Normalized ' + velname + ' - x/d = {}'.format(X[iplane]))
                     ax.set(xlabel='y/d', ylabel='z/d')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)



    with PdfPages('velocity_contour.pdf') as pfpgs:
         velocities = ['velocityx', 'velocityy', 'velocityz']
         for iplane in range(7):
             plt.rcParams["figure.autolayout"] = True

             fig, axs = plt.subplots(nrows, ncols, squeeze=False)
             for col in range(ncols):
                 for row, velname in enumerate(velocities):
                     ax = axs[row, col]
                     uavg = get_mean_velocity(velname, iplane)
                     pcm = ax.contourf(Y, Z, uavg)
                     cbar = fig.colorbar(pcm, ax=ax)
                     tick_font_size = 5
                     cbar.ax.tick_params(labelsize=tick_font_size)
                     ax.set_title('Normalized ' + velname + ' - x/d = {}'.format(X[iplane]))
                     ax.set(xlabel='y/d', ylabel='z/d')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)

    with PdfPages('tke_contour.pdf') as pfpgs:
         for iplane in range(7):
             tke = get_tke(iplane)
             fig = plt.figure()
             cp = plt.contourf(Y, Z, tke)
             plt.colorbar(cp)
             plt.title('turbulent kinetic energy - x/d = {}'.format(X[iplane]))
             plt.xlabel('y/d')
             plt.ylabel('z/d')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)

    with PdfPages('turbulent_intensity.pdf') as pfpgs:
         for iplane in range(7):
             turb_intensity = get_turb_intensity(iplane)
             fig = plt.figure()
             cp = plt.contourf(Y, Z, turb_intensity)
             plt.colorbar(cp)
             plt.title('TI(%) - x/d = {}'.format(X[iplane]))
             plt.xlabel('y/d')
             plt.ylabel('z/d')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)

    with PdfPages('velocityx_deficit_at_z=0.pdf') as pfpgs:
         for iplane in range(7):
             fig = plt.figure()
             uavg = get_mean_velocity('velocityx', iplane)
             uavg_at_z0 = uavg[:,44]
             u_def = 1.0 - uavg_at_z0
             plt.plot(u_def, Y[0,:])
             plt.title('velocityx deficit vs. y/d at z=0 on - x/d = {}'.format(X[iplane]))
             plt.ylabel('y/d')
             plt.xlabel('$u_{def}$/U')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)

    with PdfPages('velocityx_at_z=0.pdf') as pfpgs:
         for iplane in range(7):
             fig = plt.figure()
             uavg = get_mean_velocity('velocityx', iplane)
             uavg_at_z0 = uavg[:,44]
             plt.plot(Y[0,:], uavg_at_z0)
             plt.title('velocityx vs. y/d at z=0 on - x/d = {}'.format(X[iplane]))
             plt.xlabel('y/d')
             plt.ylabel('$u_{avg}$/u')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)

    with PdfPages('turbulent_intensity_at_z=0.pdf') as pfpgs:
         for iplane in range(7):
             fig = plt.figure()
             turb_intensity = get_turb_intensity(iplane)
             ti_at_z0 = turb_intensity[:,44]
             plt.plot(Y[0,:], ti_at_z0)
             plt.title('TI vs. y/d at z=0 on - x/d = {}'.format(X[iplane]))
             plt.xlabel('y/d')
             plt.ylabel('TI(%)')
             plt.tight_layout()
             pfpgs.savefig()
             plt.close(fig)


if __name__=="__main__":

     plot_contours()
