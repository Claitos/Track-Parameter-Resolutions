import numpy as np

class DetectorSetup:
    def __init__(self, average_layer_radii: list[float], layerthickness: list[float], detector_resolutions_rphi: list[float], detector_resolutions_z: list[float], radiation_length_medium: float, magnetic_field_strength: float):

        """Initializes a detector setup.

        Ordered arguments:
        - Radial positions of each detector layer in meters. Can be a list or numpy array.
        - Thickness of each detector layer in units of radiation length (spatial thickness of layer divided by the radiation length of the detector material). Can be a list or numpy array.
        - Spatial resolution of each detector layer in transversal direction in meters. Can be a list or numpy array.
        - Spatial resolution of each detector layer in longitudinal direction in meters. Can be a list or numpy array.
        - Radiation length of the medium in between the layers in meters (for air : 303.9)
        - Strength of the magentic field in Tesla
        """
         
        self.__average_layer_radii_setup = np.array(average_layer_radii)
        self.__layerthickness_setup = np.array(layerthickness)
        self.__detector_resolutions_rphi_setup = np.array(detector_resolutions_rphi)
        self.__detector_resolutions_z_setup = np.array(detector_resolutions_z)
        self.radiation_length_medium = radiation_length_medium
        self.magnetic_field_strength = magnetic_field_strength
        self.number_of_layers = len(average_layer_radii)
        self.MS_in_medium = True
        self.logging = False

        if self.radiation_length_medium <= 0:
            raise ValueError("The radiation length of the medium must be positive")
        
        if self.magnetic_field_strength <= 0:
            raise ValueError("The magnetic field strength must be positive")
        
        if self.number_of_layers <= 1:
            raise ValueError("The number of layers must be greater than 1")

        if len(average_layer_radii) == len(layerthickness) == len(detector_resolutions_rphi) == len(detector_resolutions_z):
            print("Detector Setup initialized")
        else:
            raise ValueError("The lists do not have the same length (the number of layers must be the same for all lists)")
        
        def _g0(x):               #  trackmodel helper functions 
            return x**0     
        def _g1(x):
            return x**1
        def _g2(x):
            return x**2 /2
        
        self.__g0 = _g0
        self.__g1 = _g1
        self.__g2 = _g2

##########################################################################
#############            Helper Functions            #####################
##########################################################################
        
    def _get_layers_with_hits(self, N: int):                                                    #helper function to get ITS2 layers starting from IU layer
        if N > self.number_of_layers:
            raise ValueError("The number of layers with hits cannot exceed the total number of layers")
        if N < 0:
            raise ValueError("The number of layers with hits cannot be negative")
        if N != int(N):
            raise ValueError("The number of layers with hits must be an integer")
        
        self.IU_layer = self.number_of_layers - N
        self.average_layer_radii = self.__average_layer_radii_setup[self.IU_layer:] - self.__average_layer_radii_setup[self.IU_layer]
        self.layerthickness = self.__layerthickness_setup[self.IU_layer:]
        self.detector_resolutions_rphi = self.__detector_resolutions_rphi_setup[self.IU_layer:]
        self.detector_resolutions_z = self.__detector_resolutions_z_setup[self.IU_layer:]


    def _get_circ_arc_length(self, momentum: float, chord_length):                                                  # calculates the length of a circular arc with given particle momentum and chord length
        radius_of_curvature = 1/(0.3*self.magnetic_field_strength) * momentum                              # radius of the curvature of the particle trajectory in meters
        try:
            if chord_length.all() > 2*radius_of_curvature:
                raise ValueError("The distances between detector layers cannot exceed 2*radius of curvature of the particle track inside the magnetic field")
        except:
            if chord_length > 2*radius_of_curvature:
                raise ValueError("The extrapolation radius cannot exceed 2*radius of curvature of the particle track inside the magnetic field")
        return 2 * radius_of_curvature * np.arcsin(chord_length / (2 * radius_of_curvature))                # length of the circular arc
    
        
    def _get_air_thickness_straight(self):               # helper function to calculate the amount of air in between the ITS layers in units of radiation length
        distances_between_layers = []
        distances_between_layers.append(10**(-90))                                          # to create extra entry
        for i in range(len(self.average_layer_radii)-1):
            distances_between_layers.append(self.average_layer_radii[i+1] - self.average_layer_radii[i])
        air_thickness = np.array(distances_between_layers) / self.radiation_length_medium
        return air_thickness
    
    def _get_air_thickness_parabolic(self, momentum: float):        # helper function to calculate the amount of air in between the ITS layers in units of radiation length
        distances_between_layers = []
        distances_between_layers.append(10**(-90))                                          # to create extra entry
        for i in range(len(self.average_layer_radii)-1):
            distances_between_layers.append(self.average_layer_radii[i+1] - self.average_layer_radii[i])
        distances_between_layers_corr = self._get_circ_arc_length(momentum, np.array(distances_between_layers))
        air_thickness = np.array(distances_between_layers_corr) / self.radiation_length_medium
        return air_thickness


    def _get_rms_ScatteringAngle(self, momentum: float, mass: float, material_thickness):             # calculates the variance of the scattering angle of a particle with given momentum and mass
        if momentum < 0:
            raise ValueError("Momentum cannot be negative")
        if mass < 0:
            raise ValueError("Mass cannot be negative")
        
        energy = np.sqrt(momentum**2 + mass**2)                         # energy of particle in GeV    
        beta = momentum / energy                                        # velocity of particle in units of speed of light c
        f = 0.013 * np.sqrt(material_thickness) * (1 + 0.038*np.log(material_thickness)) 
        sigma_ScatteringAngle = 1/(beta*momentum) * f                  # scattering angle variance
        return sigma_ScatteringAngle
    
    
    def _get_cov_det(self, N: int):                                 # covarinace matrix due to detector resolution helper function
        cov_det = np.zeros((N, N))
        for n in range(N):
            cov_det[n][n] = self.detector_resolutions_rphi[n]**2
        return cov_det
    
    def _get_cov_MS(self, N: int, sigma_ScatteringAngle):                # the covariance matrix due to multiple scattering helper function
        cov_MS = np.zeros((N, N))                                             
        for m in range(N):
            for n in range(N):
                sum = 0      
                for j in range(np.min(np.array([m,n]))):
                    sum += sigma_ScatteringAngle[j]**2 * (self.average_layer_radii[m] - self.average_layer_radii[j]) * (self.average_layer_radii[n] - self.average_layer_radii[j])
                cov_MS[m][n] = sum
        cov_MS[0][0] = 10**-99                                       #  to avoid singular matrix
        return cov_MS
    

    def _get_trackmodel_matrix(self, trackmodel):                #  trackmodel matrix G helper function
        trackmodel_list = []
        for f in trackmodel:
            trackmodel_list.append(f(self.average_layer_radii))
        self.trackmodel_matrix = np.stack(trackmodel_list).T
        
    def _get_cov_para(self, cov):                                                 # Covariance matrix of parameters helper function
        return np.linalg.inv(self.trackmodel_matrix.T @ np.linalg.inv(cov) @ self.trackmodel_matrix)
    
    
    def _apply(self, trackmodel, extrapolation_radius: float):                # helper function to apply array of functions on one argument
        _list = []
        for f in trackmodel:
            _list.append(f(extrapolation_radius))
        return np.array(_list)
    
    def _get_pos_reso(self, cov_para, trackmodel, extrapolation_radius: float):                                                             # Position Resolution helper function
        return np.sqrt(self._apply(trackmodel, -extrapolation_radius).T @ cov_para @ self._apply(trackmodel, -extrapolation_radius))

    def _get_momentum_reso(self, cov_para, momentum: float):                                                      # Momentum Resolution helper function
        return momentum / (0.3*self.magnetic_field_strength) * np.sqrt(cov_para[2][2])


    def _get_pos_reso_MS_extrapolation_straight(self, momentum: float, mass: float, extrapolation_radius: float):                         # Position resolution contribution from MS in air during extrapolation (straight extrapolation simulated)
        air_thickness_extrapolation = extrapolation_radius / self.radiation_length_medium
        sigma_ScatteringAngle_extrapolation = self._get_rms_ScatteringAngle(momentum, mass, air_thickness_extrapolation)
        posreso_MS_extrapolation = sigma_ScatteringAngle_extrapolation * extrapolation_radius
        return posreso_MS_extrapolation
    
    def _get_pos_reso_MS_extrapolation_parabolic(self, momentum: float, mass: float, extrapolation_radius: float):                # Position resolution contribution from MS in air during extrapolation (parabolic extrapolation simulated)
        circ_length = self._get_circ_arc_length(momentum, extrapolation_radius)
        air_thickness_extrapolation = circ_length / self.radiation_length_medium
        sigma_ScatteringAngle_extrapolation = self._get_rms_ScatteringAngle(momentum, mass, air_thickness_extrapolation)
        posreso_MS_extrapolation = sigma_ScatteringAngle_extrapolation * circ_length
        return posreso_MS_extrapolation


    def _total_transverse_resolution(self, det_reso, MS_reso, theta: float):
        MS_reso_scaled = MS_reso / np.sqrt(np.sin(np.deg2rad(theta))) 
        return np.sqrt(det_reso**2 + MS_reso_scaled**2)
    
    def _total_longitudinal_resolution(self, det_reso, MS_reso, theta: float):
        MS_reso_scaled = MS_reso / np.sin(np.deg2rad(theta))**1.5
        return np.sqrt(det_reso**2 + MS_reso_scaled**2)
    

    def _logging_pos(self, momentum, mass, number_of_hits, extrapolation_radius, polar_angle, mediumthickness, sigma_scatteringangle_medium, sigma_scatteringangle_layer, sigma_scatteringangle_total, cov_det, cov_MS, pos_uncertainty_det, pos_uncertainty_MS):               # logging helper function
        np.set_printoptions(precision=5)
        with open("log.txt", "a") as f:
            f.write("Logging initialized\n")
            f.write(f"mass: {mass} , momentum: {momentum} , number of hits: {number_of_hits}, extrapolation radius: {extrapolation_radius}, polar angle: {polar_angle} \n")
            f.write(f"medium thickness: {mediumthickness} \n")
            f.write(f"sigma scattering angle medium: {sigma_scatteringangle_medium} \n")
            f.write(f"sigma scattering angle layer: {sigma_scatteringangle_layer} \n")
            f.write(f"sigma scattering angle total: {sigma_scatteringangle_total} \n")
            f.write(f"covariance matrix due to detector resolution: \n {cov_det} \n")
            f.write(f"covariance matrix due to multiple scattering: \n {cov_MS} \n")
            f.write(f"trackmodel matrix used: \n {self.trackmodel_matrix.T} \n")
            f.write(f"position uncertainty due to detector resolution: {pos_uncertainty_det} \n")
            f.write(f"position uncertainty due to multiple scattering: {pos_uncertainty_MS} \n")
            f.write("Logging finished\n")
            f.write("\n")

    def _logging_pT(self, momentum, mass, number_of_hits, polar_angle, mediumthickness, sigma_scatteringangle_medium, sigma_scatteringangle_layer, sigma_scatteringangle_total, cov_det, cov_MS, pos_uncertainty_det, pos_uncertainty_MS):               # logging helper function
        np.set_printoptions(precision=5)
        with open("log.txt", "a") as f:
            f.write("Logging initialized\n")
            f.write(f"mass: {mass} , momentum: {momentum} , number of hits: {number_of_hits}, polar angle: {polar_angle} \n")
            f.write(f"medium thickness: {mediumthickness} \n")
            f.write(f"sigma scattering angle medium: {sigma_scatteringangle_medium} \n")
            f.write(f"sigma scattering angle layer: {sigma_scatteringangle_layer} \n")
            f.write(f"sigma scattering angle total: {sigma_scatteringangle_total} \n")
            f.write(f"covariance matrix due to detector resolution: \n {cov_det} \n")
            f.write(f"covariance matrix due to multiple scattering: \n {cov_MS} \n")
            f.write(f"trackmodel matrix used: \n {self.trackmodel_matrix.T} \n")
            f.write(f"position uncertainty due to detector resolution: {pos_uncertainty_det} \n")
            f.write(f"position uncertainty due to multiple scattering: {pos_uncertainty_MS} \n")
            f.write("Logging finished\n")
            f.write("\n")
    

##########################################################################
#############            Main Functions            #######################
##########################################################################   

    def transverse_track_position_uncertainty(self, momentum: float, mass: float, number_of_hits: int, extrapolation_radius: float, polar_angle: float) -> float:

        """Calculates the track position uncertainty in the transversal direction of a defined track in the detector setup. Returned value is in meters.

        Ordered arguments:
        - Mass of the particle in GeV/c
        - Transverse momentum of the particle in GeV/c²
        - Number of layers in which the particle has hits (in the current implementation this will the correspond to the last n layers which were defined earlier)
        - Extrapolation length in meters (in the current implementation this is defined away from layers with hits)
        - Polar angle of the particle in the detector setup in degrees
        """

        self._get_layers_with_hits(number_of_hits)

        if self.MS_in_medium:
            air_thickness = self._get_air_thickness_parabolic(momentum)
            sigma_ScatteringAngle_air = self._get_rms_ScatteringAngle(momentum, mass, air_thickness)
            sigma_ScatteringAngle_layer = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)
            sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air
        else:
            sigma_ScatteringAngle_total = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)

        cov_det = self._get_cov_det(number_of_hits)
        cov_MS = self._get_cov_MS(number_of_hits, sigma_ScatteringAngle_total)

        trackmodel_parabolic = np.array([self.__g0, self.__g1, self.__g2])
        self._get_trackmodel_matrix(trackmodel_parabolic)

        cov_para_det = self._get_cov_para(cov_det)
        cov_para_MS = self._get_cov_para(cov_MS)

        posreso_det = self._get_pos_reso(cov_para_det, trackmodel_parabolic, extrapolation_radius)
        
        if self.MS_in_medium:
            posreso_MS = np.sqrt(self._get_pos_reso(cov_para_MS, trackmodel_parabolic, extrapolation_radius)**2 + self._get_pos_reso_MS_extrapolation_parabolic(momentum, mass, extrapolation_radius)**2)
        else:
            posreso_MS = self._get_pos_reso(cov_para_MS, trackmodel_parabolic, extrapolation_radius)

        if self.logging and self.MS_in_medium:
            self._logging_pos(momentum, mass, number_of_hits, extrapolation_radius, polar_angle, air_thickness, sigma_ScatteringAngle_air, sigma_ScatteringAngle_layer, sigma_ScatteringAngle_total, cov_det, cov_MS, posreso_det, posreso_MS)

        return self._total_transverse_resolution(posreso_det, posreso_MS, polar_angle)



    def longitudinal_track_position_uncertainty(self, momentum: float, mass: float, number_of_hits: int, extrapolation_radius: float, polar_angle: float) -> float:

        """Calculates the track position uncertainty in the longitudinal direction of a defined track in the detector setup. Returned value is in meters.

        Ordered arguments:
        - Mass of the particle in GeV/c
        - Transverse momentum of the particle in GeV/c²
        - Number of layers in which the particle has hits (in the current implementation this will the correspond to the last n layers which were defined earlier)
        - Extrapolation length in meters (in the current implementation this is defined away from layers with hits)
        - Polar angle of the particle in the detector setup in degrees
        """

        self._get_layers_with_hits(number_of_hits)

        if self.MS_in_medium:
            air_thickness = self._get_air_thickness_straight()
            sigma_ScatteringAngle_air = self._get_rms_ScatteringAngle(momentum, mass, air_thickness)
            sigma_ScatteringAngle_layer = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)
            sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air
        else:
            sigma_ScatteringAngle_total = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)

        cov_det = self._get_cov_det(number_of_hits)
        cov_MS = self._get_cov_MS(number_of_hits, sigma_ScatteringAngle_total)

        trackmodel_straight = np.array([self.__g0, self.__g1])
        self._get_trackmodel_matrix(trackmodel_straight)

        cov_para_det = self._get_cov_para(cov_det)
        cov_para_MS = self._get_cov_para(cov_MS)

        posreso_det = self._get_pos_reso(cov_para_det, trackmodel_straight, extrapolation_radius)

        if self.MS_in_medium:
            posreso_MS = np.sqrt(self._get_pos_reso(cov_para_MS, trackmodel_straight, extrapolation_radius)**2 + self._get_pos_reso_MS_extrapolation_straight(momentum, mass, extrapolation_radius)**2)
        else:
            posreso_MS = self._get_pos_reso(cov_para_MS, trackmodel_straight, extrapolation_radius)

        if self.logging and self.MS_in_medium:
            self._logging_pos(momentum, mass, number_of_hits, extrapolation_radius, polar_angle, air_thickness, sigma_ScatteringAngle_air, sigma_ScatteringAngle_layer, sigma_ScatteringAngle_total, cov_det, cov_MS, posreso_det, posreso_MS)

        return self._total_longitudinal_resolution(posreso_det, posreso_MS, polar_angle)



    def transverse_momentum_reso(self, momentum: float, mass: float, number_of_hits: int, polar_angle: float)-> float:

        """Calculates the realtive transverse momentum uncertainty (transverse momentum uncertainty divided by transverse momentum) of a defined track in the detector setup. Returned value is unitless.

        Ordered arguments:
        - Mass of the particle in GeV/c
        - Transverse momentum of the particle in GeV/c²
        - Number of layers in which the particle has hits (in the current implementation this will the correspond to the last n layers which were defined earlier)
        - Polar angle of the particle in the detector setup in degrees
        """

        self._get_layers_with_hits(number_of_hits)

        if self.MS_in_medium:
            air_thickness = self._get_air_thickness_parabolic(momentum)
            sigma_ScatteringAngle_air = self._get_rms_ScatteringAngle(momentum, mass, air_thickness)
            sigma_ScatteringAngle_layer = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)
            sigma_ScatteringAngle_total = sigma_ScatteringAngle_layer + sigma_ScatteringAngle_air
        else:
            sigma_ScatteringAngle_total = self._get_rms_ScatteringAngle(momentum, mass, self.layerthickness)

        cov_det = self._get_cov_det(number_of_hits)
        cov_MS = self._get_cov_MS(number_of_hits, sigma_ScatteringAngle_total)

        trackmodel_parabolic = np.array([self.__g0, self.__g1, self.__g2])
        self._get_trackmodel_matrix(trackmodel_parabolic)

        cov_para_det = self._get_cov_para(cov_det)
        cov_para_MS = self._get_cov_para(cov_MS)

        momentumreso_det = self._get_momentum_reso(cov_para_det, momentum)
        momentumreso_MS = self._get_momentum_reso(cov_para_MS, momentum)

        if self.logging and self.MS_in_medium:
            self._logging_pT(momentum, mass, number_of_hits, polar_angle, air_thickness, sigma_ScatteringAngle_air, sigma_ScatteringAngle_layer, sigma_ScatteringAngle_total, cov_det, cov_MS, momentumreso_det, momentumreso_MS)

        return self._total_transverse_resolution(momentumreso_det, momentumreso_MS, polar_angle)



