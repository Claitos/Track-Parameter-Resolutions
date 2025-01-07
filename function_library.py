import numpy as np

##########################################################################
#############            Helper Functions            #####################
##########################################################################

def get_rms_ScatteringAngle(momentum, mass, material_thickness):             # calculates the variance of the scattering angle of a particle with given momentum and mass and given layer thickness
    energy = np.sqrt(momentum**2 + mass**2)                              # energy of particle in GeV    
    beta = momentum / energy                                             # velocity of particle in units of speed of light c
    f = 0.013 * np.sqrt(material_thickness) * (1 + 0.038*np.log(material_thickness))
    return 1/(beta*momentum) * f 


def get_circ_arc_length(momentum, chord_length):                                # calculates the length of a circular arc with given particle momentum and chord length
    radius_of_curvature = 1/(0.3*magnetic_field_strength) * momentum                              # radius of the curvature of the particle trajectory in meters
    return 2 * radius_of_curvature * np.arcsin(chord_length / (2 * radius_of_curvature))     # length of the circular arc in meters


def get_air_thickness_straight(radiation_length_air, average_layer_radii, IU_layer):    # helper function to calculate the amount of air in between the ITS layers in units of radiation length
    distances_between_layers = []
    distances_between_layers.append(10**(-90))                                          # to create 7th entry
    for i in range(len(average_layer_radii)-1):
        distances_between_layers.append(average_layer_radii[i+1] - average_layer_radii[i])
    air_thickness = np.array(distances_between_layers) / radiation_length_air
    return air_thickness[IU_layer:]


def get_air_thickness_parabolic(radiation_length_air, average_layer_radii, IU_layer, momentum):    # helper function to calculate the amount of air in between the ITS layers in units of radiation length
    distances_between_layers = []
    distances_between_layers.append(10**(-90))                                          # to create 7th entry
    for i in range(len(average_layer_radii)-1):
        distances_between_layers.append(average_layer_radii[i+1] - average_layer_radii[i])
    distances_between_layers_corr = get_circ_arc_length(momentum, np.array(distances_between_layers))
    air_thickness = np.array(distances_between_layers_corr) / radiation_length_air
    return air_thickness[IU_layer:]


def get_layers(inner, outer, IU_layer):                                   #helper function to get ITS2 layers starting from IU layer
    layer_array = np.array([inner, inner, inner, outer, outer, outer, outer])
    return layer_array[IU_layer:]


def get_cov_det(detector_resolutions, N):                                 # covarinace matrix due to detector resolution helper function
    cov_det = np.zeros((N, N))
    for n in range(N):
        cov_det[n][n] = detector_resolutions[n]**2
    return cov_det


def get_cov_MS(sigma_ScatteringAngle, layer_positions, N):               # the covariance matrix due to multiple scattering helper function
    cov_MS = np.zeros((N, N))                                             
    for m in range(N):
        for n in range(N):
            sum = 0      
            for j in range(np.min(np.array([m,n]))):
                sum += sigma_ScatteringAngle[j]**2 * (layer_positions[m] - layer_positions[j]) * (layer_positions[n] - layer_positions[j])
            cov_MS[m][n] = sum
    return cov_MS


def get_trackmodel_matrix(layer_positions, trackmodel):                  #  trackmodel matrix G helper function
    trackmodel_list = []
    for f in trackmodel:
        trackmodel_list.append(f(layer_positions))
    return np.stack(trackmodel_list).T


def get_cov_para(cov, trackmodel_matrix):                                # Covariance matrix of parameters helper function
    return np.linalg.inv(trackmodel_matrix.T @ np.linalg.inv(cov) @ trackmodel_matrix)


def apply(trackmodel, extrapolation_radius):                             # helper function to apply array of functions on one argument
    _list = []
    for f in trackmodel:
        _list.append(f(extrapolation_radius))
    return np.array(_list)


def get_pos_reso(cov_para, trackmodel, extrapolation_radius):             # Position Resolution helper function
    return np.sqrt(apply(trackmodel, -extrapolation_radius).T @ cov_para @ apply(trackmodel, -extrapolation_radius))


def get_momentum_reso(cov_para, momentum, magnetic_field_strength):       # Momentum Resolution helper function
    return momentum / (0.3*magnetic_field_strength) * np.sqrt(cov_para[2][2])


def get_pos_reso_MS_extrapolation_straight(momentum, mass, extrapolation_radius, radiation_length_air):                         # Position resolution contribution from MS in air during extrapolation (straight extrapolation simulated)
    air_thickness_extrapolation = extrapolation_radius / radiation_length_air
    sigma_ScatteringAngle_extrapolation = get_rms_ScatteringAngle(momentum, mass, air_thickness_extrapolation)
    posreso_MS_extrapolation = sigma_ScatteringAngle_extrapolation * extrapolation_radius
    return posreso_MS_extrapolation

def get_pos_reso_MS_extrapolation_parabolic(momentum, mass, extrapolation_radius, radiation_length_air):                # Position resolution contribution from MS in air during extrapolation (parabolic extrapolation simulated)
    radius_of_curvature = 1/(0.3*magnetic_field_strength) * momentum
    circ_length = 2*radius_of_curvature * np.arcsin(extrapolation_radius / (2*radius_of_curvature))
    air_thickness_extrapolation = circ_length / radiation_length_air
    sigma_ScatteringAngle_extrapolation = get_rms_ScatteringAngle(momentum, mass, air_thickness_extrapolation)
    posreso_MS_extrapolation = sigma_ScatteringAngle_extrapolation * circ_length
    return posreso_MS_extrapolation


def total_transverse_resolution(det_reso, MS_reso, theta):
    MS_reso_scaled = MS_reso / np.sqrt(np.sin(np.deg2rad(theta))) 
    return np.sqrt(det_reso**2 + MS_reso_scaled**2)

def total_longitudinal_resolution(det_reso, MS_reso, theta):
    MS_reso_scaled = MS_reso / np.sin(np.deg2rad(theta))**1.5
    return np.sqrt(det_reso**2 + MS_reso_scaled**2)


def g0(x):               #  trackmodel helper functions 
    return x**0     
def g1(x):
    return x**1
def g2(x):
    return x**2 /2

##########################################################################
#############            Global Variables            #####################
##########################################################################


sigma_detectorresolution_inner_rphi = 5 * 10**-6   #spatial resolution of the ITS2 inner layers in rphi in meters
sigma_detectorresolution_inner_z = 5 * 10**-6      #spatial resolution of the ITS2 inner layers in z    in meters

sigma_detectorresolution_outer_rphi = 5 * 10**-6   #spatial resolution of the ITS2 outer layers in rphi in meters
sigma_detectorresolution_outer_z = 5 * 10**-6      #spatial resolution of the ITS2 outer layers in z    in meters

layerthickness_inner = 0.0036   # thickness of an inner detector plane in units of radiation length 0.36%
layerthickness_outer = 0.0110   # thickness of an outer detector plane in units of radiation length 1.10%

radiation_length_air = 303.9           # radiation length of air in meters

average_layer_radii = np.array([23, 31, 39, 196, 245, 344, 393]) * 10**-3   # average radius of ITS2 layers in meters

magnetic_field_strength = 0.5           # strength of magnetic field in Alice in Tesla 



##########################################################################
#############            Main Functions            #######################
##########################################################################


def transverse_impactparameter_reso(momentum, mass, N, r, theta):
    IU_layer = len(average_layer_radii) - N  
    sigma_detectorresolution_rphi = get_layers(sigma_detectorresolution_inner_rphi, sigma_detectorresolution_outer_rphi, IU_layer)

    layer_thickness = get_layers(layerthickness_inner, layerthickness_outer, IU_layer)
    sigma_ScatteringAngle_layer = get_rms_ScatteringAngle(momentum, mass, layer_thickness)
    air_thickness = get_air_thickness_parabolic(radiation_length_air, average_layer_radii, IU_layer, momentum)
    sigma_ScatteringAngle_air = get_rms_ScatteringAngle(momentum, mass, air_thickness)
    sigma_ScatteringAngle_air[0] = 0
    sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air

    layer_positions = average_layer_radii[IU_layer:]-average_layer_radii[IU_layer]

    cov_det = get_cov_det(sigma_detectorresolution_rphi, N) 
    cov_MS = get_cov_MS(sigma_ScatteringAngle_total, layer_positions, N)
    cov_MS[0][0] = 10**-99 
  
    trackmodel_parabolic = np.array([g0, g1, g2])
    trackmodel_matrix_parabolic = get_trackmodel_matrix(layer_positions, trackmodel_parabolic)   

    cov_para_det_parabolic = get_cov_para(cov_det, trackmodel_matrix_parabolic)   
    cov_para_MS_parabolic = get_cov_para(cov_MS, trackmodel_matrix_parabolic)    

    posreso_MS_extrapolation = get_pos_reso_MS_extrapolation_parabolic(momentum, mass, r, radiation_length_air)   

    posreso_det_parabolic = get_pos_reso(cov_para_det_parabolic, trackmodel_parabolic, r)
    posreso_MS_parabolic = np.sqrt(get_pos_reso(cov_para_MS_parabolic, trackmodel_parabolic, r)**2 + posreso_MS_extrapolation**2)

    posreso_tot_parabolic = total_transverse_resolution(posreso_det_parabolic, posreso_MS_parabolic, theta)

    return posreso_tot_parabolic



def longitudinal_impactparameter_reso(momentum, mass, N , r, theta):
    IU_layer = len(average_layer_radii) - N  
    sigma_detectorresolution_z = get_layers(sigma_detectorresolution_inner_z, sigma_detectorresolution_outer_z, IU_layer)

    layer_thickness = get_layers(layerthickness_inner, layerthickness_outer, IU_layer)
    sigma_ScatteringAngle_layer = get_rms_ScatteringAngle(momentum, mass, layer_thickness)
    air_thickness = get_air_thickness_straight(radiation_length_air, average_layer_radii, IU_layer)
    sigma_ScatteringAngle_air = get_rms_ScatteringAngle(momentum, mass, air_thickness)
    sigma_ScatteringAngle_air[0] = 0
    sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air

    layer_positions = average_layer_radii[IU_layer:]-average_layer_radii[IU_layer]

    cov_det = get_cov_det(sigma_detectorresolution_z, N) 
    cov_MS = get_cov_MS(sigma_ScatteringAngle_total, layer_positions, N)
    cov_MS[0][0] = 10**-99 

    trackmodel_straight = np.array([g0, g1])    
    trackmodel_matrix_straight = get_trackmodel_matrix(layer_positions, trackmodel_straight)  

    cov_para_det_straight = get_cov_para(cov_det, trackmodel_matrix_straight)   
    cov_para_MS_straight = get_cov_para(cov_MS, trackmodel_matrix_straight)  

    posreso_MS_extrapolation = get_pos_reso_MS_extrapolation_straight(momentum, mass, r, radiation_length_air)    

    posreso_det_straight = get_pos_reso(cov_para_det_straight, trackmodel_straight, r)
    posreso_MS_straight = np.sqrt(get_pos_reso(cov_para_MS_straight, trackmodel_straight, r)**2 + posreso_MS_extrapolation**2)

    posreso_tot_straight = total_longitudinal_resolution(posreso_det_straight, posreso_MS_straight, theta)

    return posreso_tot_straight



def transverse_momentum_reso(momentum, mass, N, theta):
    IU_layer = len(average_layer_radii) - N  
    sigma_detectorresolution_rphi = get_layers(sigma_detectorresolution_inner_rphi, sigma_detectorresolution_outer_rphi, IU_layer)

    layer_thickness = get_layers(layerthickness_inner, layerthickness_outer, IU_layer)
    sigma_ScatteringAngle_layer = get_rms_ScatteringAngle(momentum, mass, layer_thickness)
    air_thickness = get_air_thickness_parabolic(radiation_length_air, average_layer_radii, IU_layer, momentum)
    sigma_ScatteringAngle_air = get_rms_ScatteringAngle(momentum, mass, air_thickness)
    sigma_ScatteringAngle_air[0] = 0
    sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air

    layer_positions = average_layer_radii[IU_layer:]-average_layer_radii[IU_layer]

    cov_det = get_cov_det(sigma_detectorresolution_rphi, N) 
    cov_MS = get_cov_MS(sigma_ScatteringAngle_total, layer_positions, N)
    cov_MS[0][0] = 10**-99 
 
    trackmodel_parabolic = np.array([g0, g1, g2])
    trackmodel_matrix_parabolic = get_trackmodel_matrix(layer_positions, trackmodel_parabolic)   

    cov_para_det_parabolic = get_cov_para(cov_det, trackmodel_matrix_parabolic)   
    cov_para_MS_parabolic = get_cov_para(cov_MS, trackmodel_matrix_parabolic)        

    momentumreso_det_parabolic = get_momentum_reso(cov_para_det_parabolic, momentum, magnetic_field_strength)
    momentumreso_MS_parabolic = get_momentum_reso(cov_para_MS_parabolic, momentum, magnetic_field_strength)

    momentumreso_tot_parabolic = total_transverse_resolution(momentumreso_det_parabolic, momentumreso_MS_parabolic, theta)

    return momentumreso_tot_parabolic




