# coding: utf-8
import numpy as np
from scipy.fftpack import fft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

rho = 1.2
aoa = [5.0, 17.0, 30.0, 45.0, 60.0, 90.0]
r_R_list = [0.30, 0.47, 0.63, 0.80]
case_dirs = ['aoa_{:.1f}'.format(i) for i in aoa ]

T = 0.00192 # sample spacing
N = 15625 # number of sample points
fs = 1/T 

#########################################################################
#########################################################################
#                   Copy data from .dat files to numpy arrays
#                        - .dat files were generated using matlab script
#########################################################################
#########################################################################

time_data = np.loadtxt("LSSTQ_for_u_12_time.dat", usecols=0, dtype=float)

# LSSTQ
lsstq_data_u_7 = np.loadtxt("LSSTQ_for_u_7_time.dat", usecols=1, dtype=float)
lsstq_data_u_12 = np.loadtxt("LSSTQ_for_u_12_time.dat", usecols=1, dtype=float)
lsstq_data_u_15 = np.loadtxt("LSSTQ_for_u_15_time.dat", usecols=1, dtype=float)
lsstq_data_u_20 = np.loadtxt("LSSTQ_for_u_20_time.dat", usecols=1, dtype=float)

with open('lssq.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(np.average(lsstq_data_u_7), np.average(lsstq_data_u_12), np.average(lsstq_data_u_15), np.average(lsstq_data_u_20)))
     f.write('%f %f %f %f\n' %(np.amin(lsstq_data_u_7), np.amin(lsstq_data_u_12), np.amin(lsstq_data_u_15), np.amin(lsstq_data_u_20)))
     f.write('%f %f %f %f' %(np.amax(lsstq_data_u_7), np.amax(lsstq_data_u_12), np.amax(lsstq_data_u_15), np.amax(lsstq_data_u_20)))

# Thrust (C_thrust) coefficient
cth30_data_u_7 = np.loadtxt("cth30_for_u_7_time.dat", usecols=1, dtype=float)
cth47_data_u_7 = np.loadtxt("cth47_for_u_7_time.dat", usecols=1, dtype=float)
cth63_data_u_7 = np.loadtxt("cth63_for_u_7_time.dat", usecols=1, dtype=float)
cth80_data_u_7 = np.loadtxt("cth80_for_u_7_time.dat", usecols=1, dtype=float)

cth30_data_u_12 = np.loadtxt("cth30_for_u_12_time.dat", usecols=1, dtype=float)
cth47_data_u_12 = np.loadtxt("cth47_for_u_12_time.dat", usecols=1, dtype=float)
cth63_data_u_12 = np.loadtxt("cth63_for_u_12_time.dat", usecols=1, dtype=float)
cth80_data_u_12 = np.loadtxt("cth80_for_u_12_time.dat", usecols=1, dtype=float)

cth30_data_u_15 = np.loadtxt("cth30_for_u_15_time.dat", usecols=1, dtype=float)
cth47_data_u_15 = np.loadtxt("cth47_for_u_15_time.dat", usecols=1, dtype=float)
cth63_data_u_15 = np.loadtxt("cth63_for_u_15_time.dat", usecols=1, dtype=float)
cth80_data_u_15 = np.loadtxt("cth80_for_u_15_time.dat", usecols=1, dtype=float)

cth30_data_u_20 = np.loadtxt("cth30_for_u_20_time.dat", usecols=1, dtype=float)
cth47_data_u_20 = np.loadtxt("cth47_for_u_20_time.dat", usecols=1, dtype=float)
cth63_data_u_20 = np.loadtxt("cth63_for_u_20_time.dat", usecols=1, dtype=float)
cth80_data_u_20 = np.loadtxt("cth80_for_u_20_time.dat", usecols=1, dtype=float)

# Torque (C_torque) coefficient
ctq30_data_u_7 = np.loadtxt("ctq30_for_u_7_time.dat", usecols=1, dtype=float)
ctq47_data_u_7 = np.loadtxt("ctq47_for_u_7_time.dat", usecols=1, dtype=float)
ctq63_data_u_7 = np.loadtxt("ctq63_for_u_7_time.dat", usecols=1, dtype=float)
ctq80_data_u_7 = np.loadtxt("ctq80_for_u_7_time.dat", usecols=1, dtype=float)

ctq30_data_u_12 = np.loadtxt("ctq30_for_u_12_time.dat", usecols=1, dtype=float)
ctq47_data_u_12 = np.loadtxt("ctq47_for_u_12_time.dat", usecols=1, dtype=float)
ctq63_data_u_12 = np.loadtxt("ctq63_for_u_12_time.dat", usecols=1, dtype=float)
ctq80_data_u_12 = np.loadtxt("ctq80_for_u_12_time.dat", usecols=1, dtype=float)

ctq30_data_u_15 = np.loadtxt("ctq30_for_u_15_time.dat", usecols=1, dtype=float)
ctq47_data_u_15 = np.loadtxt("ctq47_for_u_15_time.dat", usecols=1, dtype=float)
ctq63_data_u_15 = np.loadtxt("ctq63_for_u_15_time.dat", usecols=1, dtype=float)
ctq80_data_u_15 = np.loadtxt("ctq80_for_u_15_time.dat", usecols=1, dtype=float)

ctq30_data_u_20 = np.loadtxt("ctq30_for_u_20_time.dat", usecols=1, dtype=float)
ctq47_data_u_20 = np.loadtxt("ctq47_for_u_20_time.dat", usecols=1, dtype=float)
ctq63_data_u_20 = np.loadtxt("ctq63_for_u_20_time.dat", usecols=1, dtype=float)
ctq80_data_u_20 = np.loadtxt("ctq80_for_u_20_time.dat", usecols=1, dtype=float)

# Normal (C_N) coefficient
cn30_data_u_7 = np.loadtxt("cn30_for_u_7_time.dat", usecols=1, dtype=float)
cn47_data_u_7 = np.loadtxt("cn47_for_u_7_time.dat", usecols=1, dtype=float)
cn63_data_u_7 = np.loadtxt("cn63_for_u_7_time.dat", usecols=1, dtype=float)
cn80_data_u_7 = np.loadtxt("cn80_for_u_7_time.dat", usecols=1, dtype=float)

cn30_data_u_12 = np.loadtxt("cn30_for_u_12_time.dat", usecols=1, dtype=float)
cn47_data_u_12 = np.loadtxt("cn47_for_u_12_time.dat", usecols=1, dtype=float)
cn63_data_u_12 = np.loadtxt("cn63_for_u_12_time.dat", usecols=1, dtype=float)
cn80_data_u_12 = np.loadtxt("cn80_for_u_12_time.dat", usecols=1, dtype=float)

cn30_data_u_15 = np.loadtxt("cn30_for_u_15_time.dat", usecols=1, dtype=float)
cn47_data_u_15 = np.loadtxt("cn47_for_u_15_time.dat", usecols=1, dtype=float)
cn63_data_u_15 = np.loadtxt("cn63_for_u_15_time.dat", usecols=1, dtype=float)
cn80_data_u_15 = np.loadtxt("cn80_for_u_15_time.dat", usecols=1, dtype=float)

cn30_data_u_20 = np.loadtxt("cn30_for_u_20_time.dat", usecols=1, dtype=float)
cn47_data_u_20 = np.loadtxt("cn47_for_u_20_time.dat", usecols=1, dtype=float)
cn63_data_u_20 = np.loadtxt("cn63_for_u_20_time.dat", usecols=1, dtype=float)
cn80_data_u_20 = np.loadtxt("cn80_for_u_20_time.dat", usecols=1, dtype=float)

# Tangential (C_T) coefficient
ct30_data_u_7 = np.loadtxt("ct30_for_u_7_time.dat", usecols=1, dtype=float)
ct47_data_u_7 = np.loadtxt("ct47_for_u_7_time.dat", usecols=1, dtype=float)
ct63_data_u_7 = np.loadtxt("ct63_for_u_7_time.dat", usecols=1, dtype=float)
ct80_data_u_7 = np.loadtxt("ct80_for_u_7_time.dat", usecols=1, dtype=float)

ct30_data_u_12 = np.loadtxt("ct30_for_u_12_time.dat", usecols=1, dtype=float)
ct47_data_u_12 = np.loadtxt("ct47_for_u_12_time.dat", usecols=1, dtype=float)
ct63_data_u_12 = np.loadtxt("ct63_for_u_12_time.dat", usecols=1, dtype=float)
ct80_data_u_12 = np.loadtxt("ct80_for_u_12_time.dat", usecols=1, dtype=float)

ct30_data_u_15 = np.loadtxt("ct30_for_u_15_time.dat", usecols=1, dtype=float)
ct47_data_u_15 = np.loadtxt("ct47_for_u_15_time.dat", usecols=1, dtype=float)
ct63_data_u_15 = np.loadtxt("ct63_for_u_15_time.dat", usecols=1, dtype=float)
ct80_data_u_15 = np.loadtxt("ct80_for_u_15_time.dat", usecols=1, dtype=float)

ct30_data_u_20 = np.loadtxt("ct30_for_u_20_time.dat", usecols=1, dtype=float)
ct47_data_u_20 = np.loadtxt("ct47_for_u_20_time.dat", usecols=1, dtype=float)
ct63_data_u_20 = np.loadtxt("ct63_for_u_20_time.dat", usecols=1, dtype=float)
ct80_data_u_20 = np.loadtxt("ct80_for_u_20_time.dat", usecols=1, dtype=float)

####################################################################
####################################################################
#                           LSSQ
####################################################################
####################################################################

with PdfPages('lsstq.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(time_data, lsstq_data_u_7, label='$u_\infty$ = 7 m/s ', linestyle='solid', color='red')
    plt.legend(loc=0)
    plt.xlabel('time [s]')
    plt.ylabel('LSSTQ [N-m]')
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, lsstq_data_u_12, label='$u_\infty$ = 12 m/s ', linestyle='solid', color='red')
    plt.legend(loc=0)
    plt.xlabel('time [s]')
    plt.ylabel('LSSTQ [N-m]')
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, lsstq_data_u_15, label='$u_\infty$ = 15 m/s ', linestyle='solid', color='red')
    plt.legend(loc=0)
    plt.xlabel('time [s]')
    plt.ylabel('LSSTQ [N-m]')
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, lsstq_data_u_20, label='$u_\infty$ = 20 m/s ', linestyle='solid', color='red')
    plt.legend(loc=0)
    plt.xlabel('time [s]')
    plt.ylabel('LSSTQ [N-m]')
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)

#######################################################################
#######################################################################
# C_thrust vs. time, Mean and variation of C_thrust vs. r/R
#######################################################################
#######################################################################

cth_avg_u_7_list = []
cth_avg_u_7_list.append(np.average(cth30_data_u_7))
cth_avg_u_7_list.append(np.average(cth47_data_u_7))
cth_avg_u_7_list.append(np.average(cth63_data_u_7))
cth_avg_u_7_list.append(np.average(cth80_data_u_7))

cth_error_list_u_7 = []
cth_error_list_u_7.append(cth30_data_u_7)
cth_error_list_u_7.append(cth47_data_u_7)
cth_error_list_u_7.append(cth63_data_u_7)
cth_error_list_u_7.append(cth80_data_u_7)

cth_avg_u_12_list = []
cth_avg_u_12_list.append(np.average(cth30_data_u_12))
cth_avg_u_12_list.append(np.average(cth47_data_u_12))
cth_avg_u_12_list.append(np.average(cth63_data_u_12))
cth_avg_u_12_list.append(np.average(cth80_data_u_12))

cth_error_list_u_12 = []
cth_error_list_u_12.append(cth30_data_u_12)
cth_error_list_u_12.append(cth47_data_u_12)
cth_error_list_u_12.append(cth63_data_u_12)
cth_error_list_u_12.append(cth80_data_u_12)

cth_avg_u_15_list = []
cth_avg_u_15_list.append(np.average(cth30_data_u_15))
cth_avg_u_15_list.append(np.average(cth47_data_u_15))
cth_avg_u_15_list.append(np.average(cth63_data_u_15))
cth_avg_u_15_list.append(np.average(cth80_data_u_15))

cth_error_list_u_15 = []
cth_error_list_u_15.append(cth30_data_u_15)
cth_error_list_u_15.append(cth47_data_u_15)
cth_error_list_u_15.append(cth63_data_u_15)
cth_error_list_u_15.append(cth80_data_u_15)

cth_avg_u_20_list = []
cth_avg_u_20_list.append(np.average(cth30_data_u_20))
cth_avg_u_20_list.append(np.average(cth47_data_u_20))
cth_avg_u_20_list.append(np.average(cth63_data_u_20))
cth_avg_u_20_list.append(np.average(cth80_data_u_20))

# Saving C_th vs. r/R for various velocities in .txt files
with open('cth_avg_u_7.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cth_avg_u_7_list[0], cth_avg_u_7_list[1], cth_avg_u_7_list[2], cth_avg_u_7_list[3]))
     f.write('%f %f %f %f\n' %(np.amin(lsstq_data_u_7), np.amin(lsstq_data_u_12), np.amin(lsstq_data_u_15), np.amin(lsstq_data_u_20)))
     Cth30_std = np.std(cth30_data_u_7)
     Cth47_std = np.std(cth47_data_u_7)
     Cth63_std = np.std(cth63_data_u_7)
     Cth80_std = np.std(cth80_data_u_7)
     f.write('%f %f %f %f\n' %(cth_avg_u_7_list[0]-Cth30_std, cth_avg_u_7_list[1]-Cth47_std, cth_avg_u_7_list[2]-Cth63_std, cth_avg_u_7_list[3]-Cth80_std))
     f.write('%f %f %f %f\n' %(cth_avg_u_7_list[0]+Cth30_std, cth_avg_u_7_list[1]+Cth47_std, cth_avg_u_7_list[2]+Cth63_std, cth_avg_u_7_list[3]+Cth80_std))

with open('cth_avg_u_12.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cth_avg_u_12_list[0], cth_avg_u_12_list[1], cth_avg_u_12_list[2], cth_avg_u_12_list[3]))
     Cth30_std = np.std(cth30_data_u_12)
     Cth47_std = np.std(cth47_data_u_12)
     Cth63_std = np.std(cth63_data_u_12)
     Cth80_std = np.std(cth80_data_u_12)
     f.write('%f %f %f %f\n' %(cth_avg_u_12_list[0]-Cth30_std, cth_avg_u_12_list[1]-Cth47_std, cth_avg_u_12_list[2]-Cth63_std, cth_avg_u_12_list[3]-Cth80_std))
     f.write('%f %f %f %f\n' %(cth_avg_u_12_list[0]+Cth30_std, cth_avg_u_12_list[1]+Cth47_std, cth_avg_u_12_list[2]+Cth63_std, cth_avg_u_12_list[3]+Cth80_std))

with open('cth_avg_u_15.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cth_avg_u_15_list[0], cth_avg_u_15_list[1], cth_avg_u_15_list[2], cth_avg_u_15_list[3]))
     Cth30_std = np.std(cth30_data_u_15)
     Cth47_std = np.std(cth47_data_u_15)
     Cth63_std = np.std(cth63_data_u_15)
     Cth80_std = np.std(cth80_data_u_15)
     f.write('%f %f %f %f\n' %(cth_avg_u_15_list[0]-Cth30_std, cth_avg_u_15_list[1]-Cth47_std, cth_avg_u_15_list[2]-Cth63_std, cth_avg_u_15_list[3]-Cth80_std))
     f.write('%f %f %f %f\n' %(cth_avg_u_15_list[0]+Cth30_std, cth_avg_u_15_list[1]+Cth47_std, cth_avg_u_15_list[2]+Cth63_std, cth_avg_u_15_list[3]+Cth80_std))

with open('cth_avg_u_20.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cth_avg_u_20_list[0], cth_avg_u_20_list[1], cth_avg_u_20_list[2], cth_avg_u_20_list[3]))
     Cth30_std = np.std(cth30_data_u_20)
     Cth47_std = np.std(cth47_data_u_20)
     Cth63_std = np.std(cth63_data_u_20)
     Cth80_std = np.std(cth80_data_u_20)
     f.write('%f %f %f %f\n' %(cth_avg_u_20_list[0]-Cth30_std, cth_avg_u_20_list[1]-Cth47_std, cth_avg_u_20_list[2]-Cth63_std, cth_avg_u_20_list[3]-Cth80_std))
     f.write('%f %f %f %f\n' %(cth_avg_u_20_list[0]+Cth30_std, cth_avg_u_20_list[1]+Cth47_std, cth_avg_u_20_list[2]+Cth63_std, cth_avg_u_20_list[3]+Cth80_std))
cth_error_list_u_20 = []
cth_error_list_u_20.append(cth30_data_u_20)
cth_error_list_u_20.append(cth47_data_u_20)
cth_error_list_u_20.append(cth63_data_u_20)
cth_error_list_u_20.append(cth80_data_u_20)

# Plots:
with PdfPages('cth_vs_time_rbyR.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(time_data, cth30_data_u_7, '-', label='r/R = 0.30', color='k')
    plt.plot(time_data, cth47_data_u_7, '-', label='r/R = 0.47', color='b')
    plt.plot(time_data, cth63_data_u_7, '-', label='r/R = 0.63', color='r')
    plt.plot(time_data, cth80_data_u_7, '-', label='r/R = 0.80', color='g')
    plt.legend(loc=0)
    plt.title('$u_\infty$ = 7 m/s')
    plt.xlabel('time')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, cth30_data_u_12, '-', label='r/R = 0.30', color='k')
    plt.plot(time_data, cth47_data_u_12, '-', label='r/R = 0.47', color='b')
    plt.plot(time_data, cth63_data_u_12, '-', label='r/R = 0.63', color='r')
    plt.plot(time_data, cth80_data_u_12, '-', label='r/R = 0.80', color='g')
    plt.title('$u_\infty$ = 12 m/s') 
    plt.legend(loc=0)
    plt.xlabel('time')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, cth30_data_u_15, '-', label='r/R = 0.30', color='k')
    plt.plot(time_data, cth47_data_u_15, '-', label='r/R = 0.47', color='b')
    plt.plot(time_data, cth63_data_u_15, '-', label='r/R = 0.63', color='r')
    plt.plot(time_data, cth80_data_u_15, '-', label='r/R = 0.80', color='g')
    plt.title('$u_\infty$ = 15 m/s') 
    plt.legend(loc=0)
    plt.xlabel('time')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(time_data, cth30_data_u_20, '-', label='r/R = 0.30', color='k')
    plt.plot(time_data, cth47_data_u_20, '-', label='r/R = 0.47', color='b')
    plt.plot(time_data, cth63_data_u_20, '-', label='r/R = 0.63', color='r')
    plt.plot(time_data, cth80_data_u_20, '-', label='r/R = 0.80', color='g')
    plt.title('$u_\infty$ = 20 m/s')  
    plt.legend(loc=0)
    plt.xlabel('time')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

with PdfPages('cth_avg_vs_rbyR.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(r_R_list, cth_avg_u_7_list, label='$u_\infty$ = 7 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cth_error_list_u_7):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, cth_avg_u_12_list, label='$u_\infty$ = 12 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cth_error_list_u_12):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, cth_avg_u_15_list, label='$u_\infty$ = 15 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cth_error_list_u_15):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure() 
    plt.plot(r_R_list, cth_avg_u_20_list, label='$u_\infty$ = 20 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cth_error_list_u_20):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{thrust}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

with PdfPages('cth_psd_u_7.pdf') as pfpgs:
    # r/R = 0.30
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth30_u_7_fft = fft(np.array(cth30_data_u_7) - np.average(cth30_data_u_7))    
    energy = 2.0/N * np.abs(cth30_u_7_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.30')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.47
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth47_u_7_fft = fft(np.array(cth47_data_u_7) - np.average(cth47_data_u_7))   
    energy = 2.0/N * np.abs(cth47_u_7_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.47')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.63
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth63_u_7_fft = fft(np.array(cth63_data_u_7) - np.average(cth63_data_u_7))   
    energy = 2.0/N * np.abs(cth63_u_7_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.63')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.80
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth80_u_7_fft = fft(np.array(cth80_data_u_7) - np.average(cth80_data_u_7))   
    energy = 2.0/N * np.abs(cth80_u_7_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.80')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

with PdfPages('cth_psd_u_12.pdf') as pfpgs:
    # r/R = 0.30
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth30_u_12_fft = fft(np.array(cth30_data_u_12) - np.average(cth30_data_u_12))    
    energy = 2.0/N * np.abs(cth30_u_12_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.30')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.47
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth47_u_12_fft = fft(np.array(cth47_data_u_12) - np.average(cth47_data_u_12))   
    energy = 2.0/N * np.abs(cth47_u_12_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.47')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.63
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth63_u_12_fft = fft(np.array(cth63_data_u_12) - np.average(cth63_data_u_12))   
    energy = 2.0/N * np.abs(cth63_u_12_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.63')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)


    # r/R = 0.80
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth80_u_12_fft = fft(np.array(cth80_data_u_12) - np.average(cth80_data_u_12))   
    energy = 2.0/N * np.abs(cth80_u_12_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.80')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

with PdfPages('cth_psd_u_15.pdf') as pfpgs:
    # r/R = 0.30
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth30_u_15_fft = fft(np.array(cth30_data_u_15) - np.average(cth30_data_u_15))    
    energy = 2.0/N * np.abs(cth30_u_15_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.30')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.47
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth47_u_15_fft = fft(np.array(cth47_data_u_15) - np.average(cth47_data_u_15))   
    energy = 2.0/N * np.abs(cth47_u_15_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.47')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.63
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth63_u_15_fft = fft(np.array(cth63_data_u_15) - np.average(cth63_data_u_15))   
    energy = 2.0/N * np.abs(cth63_u_15_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.63')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.80
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth80_u_15_fft = fft(np.array(cth80_data_u_15) - np.average(cth80_data_u_15))   
    energy = 2.0/N * np.abs(cth80_u_15_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.80')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

with PdfPages('cth_psd_u_20.pdf') as pfpgs:
    # r/R = 0.30
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth30_u_20_fft = fft(np.array(cth30_data_u_20) - np.average(cth30_data_u_20))   
    energy = 2.0/N * np.abs(cth30_u_20_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.30')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.47
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth47_u_20_fft = fft(np.array(cth47_data_u_20) - np.average(cth47_data_u_20))   
    energy = 2.0/N * np.abs(cth47_u_20_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.47')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.63
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth63_u_20_fft = fft(np.array(cth63_data_u_20) - np.average(cth63_data_u_20))   
    energy = 2.0/N * np.abs(cth63_u_20_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.63')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    # r/R = 0.80
    fig = plt.figure()
    sampling_freq = fftfreq(N, T)[1:N//2]
    cth80_u_20_fft = fft(np.array(cth80_data_u_20) - np.average(cth80_data_u_20))   
    energy = 2.0/N * np.abs(cth80_u_20_fft[1:N//2])    
    plt.loglog(sampling_freq, energy, label='r/R = 0.80')
    plt.legend(loc=0)
    plt.xlim(left=1.0)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD ($C_{thrust})$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

#######################################################################################
#######################################################################################
# Mean and variation of C_Torque vs. r/R
#######################################################################################
#######################################################################################

ctq_avg_u_7_list = []
ctq_avg_u_7_list.append(np.average(ctq30_data_u_7))
ctq_avg_u_7_list.append(np.average(ctq47_data_u_7))
ctq_avg_u_7_list.append(np.average(ctq63_data_u_7))
ctq_avg_u_7_list.append(np.average(ctq80_data_u_7))

ctq_error_list_u_7 = []
ctq_error_list_u_7.append(ctq30_data_u_7)
ctq_error_list_u_7.append(ctq47_data_u_7)
ctq_error_list_u_7.append(ctq63_data_u_7)
ctq_error_list_u_7.append(ctq80_data_u_7)

ctq_avg_u_12_list = []
ctq_avg_u_12_list.append(np.average(ctq30_data_u_12))
ctq_avg_u_12_list.append(np.average(ctq47_data_u_12))
ctq_avg_u_12_list.append(np.average(ctq63_data_u_12))
ctq_avg_u_12_list.append(np.average(ctq80_data_u_12))

ctq_error_list_u_12 = []
ctq_error_list_u_12.append(ctq30_data_u_12)
ctq_error_list_u_12.append(ctq47_data_u_12)
ctq_error_list_u_12.append(ctq63_data_u_12)
ctq_error_list_u_12.append(ctq80_data_u_12)

ctq_avg_u_15_list = []
ctq_avg_u_15_list.append(np.average(ctq30_data_u_15))
ctq_avg_u_15_list.append(np.average(ctq47_data_u_15))
ctq_avg_u_15_list.append(np.average(ctq63_data_u_15))
ctq_avg_u_15_list.append(np.average(ctq80_data_u_15))

ctq_error_list_u_15 = []
ctq_error_list_u_15.append(ctq30_data_u_15)
ctq_error_list_u_15.append(ctq47_data_u_15)
ctq_error_list_u_15.append(ctq63_data_u_15)
ctq_error_list_u_15.append(ctq80_data_u_15)

ctq_avg_u_20_list = []
ctq_avg_u_20_list.append(np.average(ctq30_data_u_20))
ctq_avg_u_20_list.append(np.average(ctq47_data_u_20))
ctq_avg_u_20_list.append(np.average(ctq63_data_u_20))
ctq_avg_u_20_list.append(np.average(ctq80_data_u_20))

ctq_error_list_u_20 = []
ctq_error_list_u_20.append(ctq30_data_u_20)
ctq_error_list_u_20.append(ctq47_data_u_20)
ctq_error_list_u_20.append(ctq63_data_u_20)
ctq_error_list_u_20.append(ctq80_data_u_20)

with open('ctq_avg_u_7.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ctq_avg_u_7_list[0], ctq_avg_u_7_list[1], ctq_avg_u_7_list[2], ctq_avg_u_7_list[3])) 
     Ctq30_std = np.std(ctq30_data_u_7)
     Ctq47_std = np.std(ctq47_data_u_7)
     Ctq63_std = np.std(ctq63_data_u_7)
     Ctq80_std = np.std(ctq80_data_u_7)
     f.write('%f %f %f %f\n' %(ctq_avg_u_7_list[0]-Ctq30_std, ctq_avg_u_7_list[1]-Ctq47_std, ctq_avg_u_7_list[2]-Ctq63_std, ctq_avg_u_7_list[3]-Ctq80_std))
     f.write('%f %f %f %f\n' %(ctq_avg_u_7_list[0]+Ctq30_std, ctq_avg_u_7_list[1]+Ctq47_std, ctq_avg_u_7_list[2]+Ctq63_std, ctq_avg_u_7_list[3]+Ctq80_std))

with open('ctq_avg_u_12.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ctq_avg_u_12_list[0], ctq_avg_u_12_list[1], ctq_avg_u_12_list[2], ctq_avg_u_12_list[3]))  
     Ctq30_std = np.std(ctq30_data_u_12)
     Ctq47_std = np.std(ctq47_data_u_12)
     Ctq63_std = np.std(ctq63_data_u_12)
     Ctq80_std = np.std(ctq80_data_u_12)
     f.write('%f %f %f %f\n' %(ctq_avg_u_12_list[0]-Ctq30_std, ctq_avg_u_12_list[1]-Ctq47_std, ctq_avg_u_12_list[2]-Ctq63_std, ctq_avg_u_12_list[3]-Ctq80_std))
     f.write('%f %f %f %f\n' %(ctq_avg_u_12_list[0]+Ctq30_std, ctq_avg_u_12_list[1]+Ctq47_std, ctq_avg_u_12_list[2]+Ctq63_std, ctq_avg_u_12_list[3]+Ctq80_std))

with open('ctq_avg_u_15.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ctq_avg_u_15_list[0], ctq_avg_u_15_list[1], ctq_avg_u_15_list[2], ctq_avg_u_15_list[3]))  
     Ctq30_std = np.std(ctq30_data_u_15)
     Ctq47_std = np.std(ctq47_data_u_15)
     Ctq63_std = np.std(ctq63_data_u_15)
     Ctq80_std = np.std(ctq80_data_u_15)
     f.write('%f %f %f %f\n' %(ctq_avg_u_15_list[0]-Ctq30_std, ctq_avg_u_15_list[1]-Ctq47_std, ctq_avg_u_15_list[2]-Ctq63_std, ctq_avg_u_15_list[3]-Ctq80_std))
     f.write('%f %f %f %f\n' %(ctq_avg_u_15_list[0]+Ctq30_std, ctq_avg_u_15_list[1]+Ctq47_std, ctq_avg_u_15_list[2]+Ctq63_std, ctq_avg_u_15_list[3]+Ctq80_std))

with open('ctq_avg_u_20.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ctq_avg_u_20_list[0], ctq_avg_u_20_list[1], ctq_avg_u_20_list[2], ctq_avg_u_20_list[3]))  
     Ctq30_std = np.std(ctq30_data_u_20)
     Ctq47_std = np.std(ctq47_data_u_20)
     Ctq63_std = np.std(ctq63_data_u_20)
     Ctq80_std = np.std(ctq80_data_u_20)
     f.write('%f %f %f %f\n' %(ctq_avg_u_20_list[0]-Ctq30_std, ctq_avg_u_20_list[1]-Ctq47_std, ctq_avg_u_20_list[2]-Ctq63_std, ctq_avg_u_20_list[3]-Ctq80_std))
     f.write('%f %f %f %f\n' %(ctq_avg_u_20_list[0]+Ctq30_std, ctq_avg_u_20_list[1]+Ctq47_std, ctq_avg_u_20_list[2]+Ctq63_std, ctq_avg_u_20_list[3]+Ctq80_std))

with PdfPages('ctq_avg_vs_rbyR.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(r_R_list, ctq_avg_u_7_list, label='$u_\infty$ = 7 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ctq_error_list_u_7):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{torque}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, ctq_avg_u_12_list, label='$u_\infty$ = 12 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ctq_error_list_u_12):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{torque}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, ctq_avg_u_15_list, label='$u_\infty$ = 15 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ctq_error_list_u_15):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{torque}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure() 
    plt.plot(r_R_list, ctq_avg_u_20_list, label='$u_\infty$ = 20 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ctq_error_list_u_20):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{torque}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

#########################################################################
#########################################################################
# Mean and variation of C_N vs. r/R
#########################################################################
########################################################################

cn_avg_u_7_list = []
cn_avg_u_7_list.append(np.average(cn30_data_u_7))
cn_avg_u_7_list.append(np.average(cn47_data_u_7))
cn_avg_u_7_list.append(np.average(cn63_data_u_7))
cn_avg_u_7_list.append(np.average(cn80_data_u_7))

cn_error_list_u_7 = []
cn_error_list_u_7.append(cn30_data_u_7)
cn_error_list_u_7.append(cn47_data_u_7)
cn_error_list_u_7.append(cn63_data_u_7)
cn_error_list_u_7.append(cn80_data_u_7)

cn_avg_u_12_list = []
cn_avg_u_12_list.append(np.average(cn30_data_u_12))
cn_avg_u_12_list.append(np.average(cn47_data_u_12))
cn_avg_u_12_list.append(np.average(cn63_data_u_12))
cn_avg_u_12_list.append(np.average(cn80_data_u_12))

cn_error_list_u_12 = []
cn_error_list_u_12.append(cn30_data_u_12)
cn_error_list_u_12.append(cn47_data_u_12)
cn_error_list_u_12.append(cn63_data_u_12)
cn_error_list_u_12.append(cn80_data_u_12)

cn_avg_u_15_list = []
cn_avg_u_15_list.append(np.average(cn30_data_u_15))
cn_avg_u_15_list.append(np.average(cn47_data_u_15))
cn_avg_u_15_list.append(np.average(cn63_data_u_15))
cn_avg_u_15_list.append(np.average(cn80_data_u_15))

cn_error_list_u_15 = []
cn_error_list_u_15.append(cn30_data_u_15)
cn_error_list_u_15.append(cn47_data_u_15)
cn_error_list_u_15.append(cn63_data_u_15)
cn_error_list_u_15.append(cn80_data_u_15)

cn_avg_u_20_list = []
cn_avg_u_20_list.append(np.average(cn30_data_u_20))
cn_avg_u_20_list.append(np.average(cn47_data_u_20))
cn_avg_u_20_list.append(np.average(cn63_data_u_20))
cn_avg_u_20_list.append(np.average(cn80_data_u_20))

cn_error_list_u_20 = []
cn_error_list_u_20.append(cn30_data_u_20)
cn_error_list_u_20.append(cn47_data_u_20)
cn_error_list_u_20.append(cn63_data_u_20)
cn_error_list_u_20.append(cn80_data_u_20)

with open('cn_avg_u_7.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cn_avg_u_7_list[0], cn_avg_u_7_list[1], cn_avg_u_7_list[2], cn_avg_u_7_list[3]))
     Cn30_std = np.std(cn30_data_u_7)
     Cn47_std = np.std(cn47_data_u_7)
     Cn63_std = np.std(cn63_data_u_7)
     Cn80_std = np.std(cn80_data_u_7)
     f.write('%f %f %f %f\n' %(cn_avg_u_7_list[0]-Cn30_std, cn_avg_u_7_list[1]-Cn47_std, cn_avg_u_7_list[2]-Cn63_std, cn_avg_u_7_list[3]-Cn80_std))
     f.write('%f %f %f %f\n' %(cn_avg_u_7_list[0]+Cn30_std, cn_avg_u_7_list[1]+Cn47_std, cn_avg_u_7_list[2]+Cn63_std, cn_avg_u_7_list[3]+Cn80_std))

with open('cn_avg_u_12.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cn_avg_u_12_list[0], cn_avg_u_12_list[1], cn_avg_u_12_list[2], cn_avg_u_12_list[3]))
     Cn30_std = np.std(cn30_data_u_12)
     Cn47_std = np.std(cn47_data_u_12)
     Cn63_std = np.std(cn63_data_u_12)
     Cn80_std = np.std(cn80_data_u_12)
     f.write('%f %f %f %f\n' %(cn_avg_u_12_list[0]-Cn30_std, cn_avg_u_12_list[1]-Cn47_std, cn_avg_u_12_list[2]-Cn63_std, cn_avg_u_12_list[3]-Cn80_std))
     f.write('%f %f %f %f\n' %(cn_avg_u_12_list[0]+Cn30_std, cn_avg_u_12_list[1]+Cn47_std, cn_avg_u_12_list[2]+Cn63_std, cn_avg_u_12_list[3]+Cn80_std))

with open('cn_avg_u_15.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cn_avg_u_15_list[0], cn_avg_u_15_list[1], cn_avg_u_15_list[2], cn_avg_u_15_list[3]))
     Cn30_std = np.std(cn30_data_u_15)
     Cn47_std = np.std(cn47_data_u_15)
     Cn63_std = np.std(cn63_data_u_15)
     Cn80_std = np.std(cn80_data_u_15)
     f.write('%f %f %f %f\n' %(cn_avg_u_15_list[0]-Cn30_std, cn_avg_u_15_list[1]-Cn47_std, cn_avg_u_15_list[2]-Cn63_std, cn_avg_u_15_list[3]-Cn80_std))
     f.write('%f %f %f %f\n' %(cn_avg_u_15_list[0]+Cn30_std, cn_avg_u_15_list[1]+Cn47_std, cn_avg_u_15_list[2]+Cn63_std, cn_avg_u_15_list[3]+Cn80_std))

with open('cn_avg_u_20.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(cn_avg_u_20_list[0], cn_avg_u_20_list[1], cn_avg_u_20_list[2], cn_avg_u_20_list[3]))
     Cn30_std = np.std(cn30_data_u_20)
     Cn47_std = np.std(cn47_data_u_20)
     Cn63_std = np.std(cn63_data_u_20)
     Cn80_std = np.std(cn80_data_u_20)
     f.write('%f %f %f %f\n' %(cn_avg_u_20_list[0]-Cn30_std, cn_avg_u_20_list[1]-Cn47_std, cn_avg_u_20_list[2]-Cn63_std, cn_avg_u_20_list[3]-Cn80_std))
     f.write('%f %f %f %f\n' %(cn_avg_u_20_list[0]+Cn30_std, cn_avg_u_20_list[1]+Cn47_std, cn_avg_u_20_list[2]+Cn63_std, cn_avg_u_20_list[3]+Cn80_std))

with PdfPages('cn_avg_vs_rbyR.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(r_R_list, cn_avg_u_7_list, label='$u_\infty$ = 7 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cn_error_list_u_7):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{normal}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, cn_avg_u_12_list, label='$u_\infty$ = 12 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cn_error_list_u_12):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{normal}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, cn_avg_u_15_list, label='$u_\infty$ = 15 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cn_error_list_u_15):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{normal}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure() 
    plt.plot(r_R_list, cn_avg_u_20_list, label='$u_\infty$ = 20 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, cn_error_list_u_20):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{normal}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

##################################################################
##################################################################
# Mean and variation of C_T vs. r/R
##################################################################
##################################################################
ct_avg_u_7_list = []
ct_avg_u_7_list.append(np.average(ct30_data_u_7))
ct_avg_u_7_list.append(np.average(ct47_data_u_7))
ct_avg_u_7_list.append(np.average(ct63_data_u_7))
ct_avg_u_7_list.append(np.average(ct80_data_u_7))

ct_error_list_u_7 = []
ct_error_list_u_7.append(ct30_data_u_7)
ct_error_list_u_7.append(ct47_data_u_7)
ct_error_list_u_7.append(ct63_data_u_7)
ct_error_list_u_7.append(ct80_data_u_7)

ct_avg_u_12_list = []
ct_avg_u_12_list.append(np.average(ct30_data_u_12))
ct_avg_u_12_list.append(np.average(ct47_data_u_12))
ct_avg_u_12_list.append(np.average(ct63_data_u_12))
ct_avg_u_12_list.append(np.average(ct80_data_u_12))

ct_error_list_u_12 = []
ct_error_list_u_12.append(ct30_data_u_12)
ct_error_list_u_12.append(ct47_data_u_12)
ct_error_list_u_12.append(ct63_data_u_12)
ct_error_list_u_12.append(ct80_data_u_12)

ct_avg_u_15_list = []
ct_avg_u_15_list.append(np.average(ct30_data_u_15))
ct_avg_u_15_list.append(np.average(ct47_data_u_15))
ct_avg_u_15_list.append(np.average(ct63_data_u_15))
ct_avg_u_15_list.append(np.average(ct80_data_u_15))

ct_error_list_u_15 = []
ct_error_list_u_15.append(ct30_data_u_15)
ct_error_list_u_15.append(ct47_data_u_15)
ct_error_list_u_15.append(ct63_data_u_15)
ct_error_list_u_15.append(ct80_data_u_15)

ct_avg_u_20_list = []
ct_avg_u_20_list.append(np.average(ct30_data_u_20))
ct_avg_u_20_list.append(np.average(ct47_data_u_20))
ct_avg_u_20_list.append(np.average(ct63_data_u_20))
ct_avg_u_20_list.append(np.average(ct80_data_u_20))

ct_error_list_u_20 = []
ct_error_list_u_20.append(ct30_data_u_20)
ct_error_list_u_20.append(ct47_data_u_20)
ct_error_list_u_20.append(ct63_data_u_20)
ct_error_list_u_20.append(ct80_data_u_20)

with open('ct_avg_u_7.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ct_avg_u_7_list[0], ct_avg_u_7_list[1], ct_avg_u_7_list[2], ct_avg_u_7_list[3]))
     Ct30_std = np.std(ct30_data_u_7)
     Ct47_std = np.std(ct47_data_u_7)
     Ct63_std = np.std(ct63_data_u_7)
     Ct80_std = np.std(ct80_data_u_7)
     f.write('%f %f %f %f\n' %(ct_avg_u_7_list[0]-Ct30_std, ct_avg_u_7_list[1]-Ct47_std, ct_avg_u_7_list[2]-Ct63_std, ct_avg_u_7_list[3]-Ct80_std))
     f.write('%f %f %f %f\n' %(ct_avg_u_7_list[0]+Ct30_std, ct_avg_u_7_list[1]+Ct47_std, ct_avg_u_7_list[2]+Ct63_std, ct_avg_u_7_list[3]+Ct80_std))

with open('ct_avg_u_12.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ct_avg_u_12_list[0], ct_avg_u_12_list[1], ct_avg_u_12_list[2], ct_avg_u_12_list[3]))
     Ct30_std = np.std(ct30_data_u_12)
     Ct47_std = np.std(ct47_data_u_12)
     Ct63_std = np.std(ct63_data_u_12)
     Ct80_std = np.std(ct80_data_u_12)
     f.write('%f %f %f %f\n' %(ct_avg_u_12_list[0]-Ct30_std, ct_avg_u_12_list[1]-Ct47_std, ct_avg_u_12_list[2]-Ct63_std, ct_avg_u_12_list[3]-Ct80_std))
     f.write('%f %f %f %f\n' %(ct_avg_u_12_list[0]+Ct30_std, ct_avg_u_12_list[1]+Ct47_std, ct_avg_u_12_list[2]+Ct63_std, ct_avg_u_12_list[3]+Ct80_std))

with open('ct_avg_u_15.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ct_avg_u_15_list[0], ct_avg_u_15_list[1], ct_avg_u_15_list[2], ct_avg_u_15_list[3]))
     Ct30_std = np.std(ct30_data_u_15)
     Ct47_std = np.std(ct47_data_u_15)
     Ct63_std = np.std(ct63_data_u_15)
     Ct80_std = np.std(ct80_data_u_15)
     f.write('%f %f %f %f\n' %(ct_avg_u_15_list[0]-Ct30_std, ct_avg_u_15_list[1]-Ct47_std, ct_avg_u_15_list[2]-Ct63_std, ct_avg_u_15_list[3]-Ct80_std))
     f.write('%f %f %f %f\n' %(ct_avg_u_15_list[0]+Ct30_std, ct_avg_u_15_list[1]+Ct47_std, ct_avg_u_15_list[2]+Ct63_std, ct_avg_u_15_list[3]+Ct80_std))

with open('ct_avg_u_20.txt', 'w') as f:
     f.write('%f %f %f %f\n' %(ct_avg_u_20_list[0], ct_avg_u_20_list[1], ct_avg_u_20_list[2], ct_avg_u_20_list[3]))
     Ct30_std = np.std(ct30_data_u_20)
     Ct47_std = np.std(ct47_data_u_20)
     Ct63_std = np.std(ct63_data_u_20)
     Ct80_std = np.std(ct80_data_u_20)
     f.write('%f %f %f %f\n' %(ct_avg_u_20_list[0]-Ct30_std, ct_avg_u_20_list[1]-Ct47_std, ct_avg_u_20_list[2]-Ct63_std, ct_avg_u_20_list[3]-Ct80_std))
     f.write('%f %f %f %f\n' %(ct_avg_u_20_list[0]+Ct30_std, ct_avg_u_20_list[1]+Ct47_std, ct_avg_u_20_list[2]+Ct63_std, ct_avg_u_20_list[3]+Ct80_std))

with PdfPages('ct_avg_vs_rbyR.pdf') as pfpgs:
    fig = plt.figure()
    plt.plot(r_R_list, ct_avg_u_7_list, label='$u_\infty$ = 7 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ct_error_list_u_7):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{tangential}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, ct_avg_u_12_list, label='$u_\infty$ = 12 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ct_error_list_u_12):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{tangential}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r_R_list, ct_avg_u_15_list, label='$u_\infty$ = 15 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ct_error_list_u_15):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{tangential}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)

    fig = plt.figure() 
    plt.plot(r_R_list, ct_avg_u_20_list, label='$u_\infty$ = 20 m/s ', color='red', marker = 'x')
    for xe, ye in zip(r_R_list, ct_error_list_u_20):
      plt.scatter([xe]*len(ye), ye, color='k', marker = 'o')
    plt.legend(loc=0)
    plt.xlabel('r/R')
    plt.ylabel('$C_{tangential}$')
    plt.tight_layout()
    pfpgs.savefig()    
    plt.close(fig)
