Repository contains files for pre-processing, test cases and post-processing for NRELVI

Near body mesh (in .exo format):  
/projects/hfm/sbidadi/nrel_phase_vi/mesh/nrelvi_nearbody_mesh.exo

######################
# Synthetic Turbulence
######################

Synthetic turbulence netcdf files can be found here:  
/projects/hfm/sbidadi/nrel_phase_vi/synthetic_turbulence/iddes/bin_to_netcdf

Each subdirectory corresponds to a particular wind speed
and contains,  
	- boxturb.yaml for converting bin to netcdf format  
        - Binary files are generated with the command:  
                mann_turb_x64.exe test <AlphaEpsilon> <LengthScale> <Gamma>  
		1209 <Nx> <Ny> <Nz> <dx> <dy> <dz> true

More information on synthetic turbulence can be found here:  
	https://github.com/lawrenceccheung/SyntheticTurbulenceTest  
        https://sima.sintef.no/doc/4.4.0/windfield/context/MannWindGenerator.html

######################
# Post Processing
######################

Post Processing scripts directory contains subdirectories:  
	blade_aerodynamics  - For load calculations on the blades  
	wake_aerodynamics - For plotting velocity deficit and TI profiles
