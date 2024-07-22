import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

import matplotlib.pyplot as plt

print("Model Imported Successfully!")

#########################################################
################# SUBSTANCE PROPERTIES ##################
#########################################################

# NaOH property functions
def TemperatureNaOH(P, x1):
    """
    last change: brought to python by Dorian Höffner
    author: 16.7.2012 Anna Jahnke */
    Berechnung nach J. Olsson et. al "Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations",
    International Journal of Thermophysics, Vol. 18, No. 3, 1997
    Boiling point temperature of aqueous NaOH-H2O Solution in °C
    
    Gültigkeitsbereich:
    T in [°C]
    MassFraction X[:]
    X[1]: [kg [NaOH)/ kg [H2O + NaOH)] 
    X[2]: [kg [H2O)/ kg [H2O + NaOH)]
    0<=T<20     0.582<=X[2]<=1
    20<=T<60    0.500<=X[2]<=1
    60<=T<70    0.353<=X[2]<=1
    70<=T<150    0.300<=X[2]<=1
    150<=T<=200  0.200<=X[2]<=1
    p in [Pa]"""

    # x X[1]
    x = 1 - x1 # X[2]

    k=[-113.93947, 209.82305, 494.77153, 6860.8330, 2676.6433, -21740.328, -34750.872, -20122.157, -4102.9890]
    l=[16.240074, -11.864008, -223.47305, -1650.3997, -5997.3118, -12318.744, -15303.153, -11707.480, -5364.9554, -1338.5412, -137.96889]
    m=[-226.80157, 293.17155, 5081.8791, 36752.126, 131262.00, 259399.54, 301696.22, 208617.90, 81774.024, 15648.526, 906.29769]

    a1=k[0] + k[1]*np.log(x) + k[2]*(np.log(x))**2 + k[3]*(np.log(x))**3 + k[4]*(np.log(x))**4 + k[5]*(np.log(x))**5 + k[6]*(np.log(x))**6 + k[7]*(np.log(x))**7 + k[8]*(np.log(x))**8
    a2=l[0] + l[1]*np.log(x) + l[2]*(np.log(x))**2 + l[3]*(np.log(x))**3 + l[4]*(np.log(x))**4 + l[5]*(np.log(x))**5 + l[6]*(np.log(x))**6 + l[7]*(np.log(x))**7 + l[8]*(np.log(x))**8 + l[9]*(np.log(x))**9 + l[10]*(np.log(x))**10
    a3=m[0] + m[1]*np.log(x) + m[2]*(np.log(x))**2 + m[3]*(np.log(x))**3 + m[4]*(np.log(x))**4 + m[5]*(np.log(x))**5 + m[6]*(np.log(x))**6 + m[7]*(np.log(x))**7 + m[8]*(np.log(x))**8 + m[9]*(np.log(x))**9 + m[10]*(np.log(x))**10

    p_kPa=P/1000  # Druck in [kPa]
    # Die Gleichung für den Dampfdruck der Lösung wurde umgestellt nach der Siedetemperatur T
    T = (a1 + a3 * np.log(p_kPa)) / (np.log(p_kPa) - a2)
    
    return T

def PressureNaOH(x, T):
    """
    Calculates the pressure of aqueous NaOH-H2O solutions at high concentrations.
    
    Last Change: Dorian Höffner 2024-04-26 (translated to Python, changed input T to [°C], changed return value to [Pa])
    Author: Anna Jahnke, Roman Ziegenhardt
    Source: Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations, J. Olsson, A. Jernqvist, G. Aly,
            International Journal of Thermophysics Vol. 18, No. 3, 1997
    
    Parameters:
    T (array-like): Temperature in [°C].
    x (array-like): Mass fraction, defined as m_NaOH / (m_H2O + m_NaOH).
    
    Returns:
    p (float or array-like): Pressure in [Pa]

    Notes:
    Wertebereich: t in Grad Celsius!
    0<=t<20    		0.582<=x<=1
    20<=t<60		0.500<=x<=1
    60<=t<70       	0.353<=x<=1
    70<=t<150		0.300<=x<=1
    150<=t<=200		0.200<=x<=1
    """

    # convert NaOH mass fraction to water mass fraction
    x = 1 - x
    
    if isinstance(x, (int, float)) or isinstance(T, (int, float)):
        pass
    elif len(x) != len(T):
        raise ValueError('x and T must have the same length')

    # Constants
    k = np.array([-113.93947, 209.82305, 494.77153, 6860.8330, 2676.6433,
                  -21740.328, -34750.872, -20122.157, -4102.9890])
    l = np.array([16.240074, -11.864008, -223.47305, -1650.3997, -5997.3118,
                  -12318.744, -15303.153, -11707.480, -5364.9554, -1338.5412,
                  -137.96889])
    m = np.array([-226.80157, 293.17155, 5081.8791, 36752.126, 131262.00,
                  259399.54, 301696.22, 208617.90, 81774.024, 15648.526,
                  906.29769])

    # Calculate pressure
    log_x = np.log(x)
    a1 = np.polyval(k[::-1], log_x)
    a2 = np.polyval(l[::-1], log_x)
    a3 = np.polyval(m[::-1], log_x)

    logP = (a1 + a2 * T) / (T - a3)
    p = np.exp(logP) * 1000

    return p # pressure in Pa

def EnthalpieNaOH(x, T):
    """
    Autor: 	Roman Ziegenhardt
    Last Change: brought to python by Dorian Höffner
    Quelle: 	Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations, J. OIsson, A. Jernqvist, G. Aly, 
    		International Journal of Thermophysics Vol. 18. No. 3. 1997
    zuletzt geändert: Elisabeth Thiele: Temperatur in °C und salt mass
    fraction statt water mass fraction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Parameter:
     T in [°C]
     x = m_NaOH/(m_H2O + m_NaOH)
     h in [kJ/kg]

    Wertebereich: t in Grad Celsius!
    0<=t<4			0.780 <=xi<=1
    4=<t<10	    0.680<=xi<=1
    10=<t<15		0.580<=xi<=1
    15=<t<26		0.540<=xi<=1
    26 =<t<37		0.440<=xi<=1
    37=<t<48		0.400<=xi<=1
    48=<t<60		0.340<=xi<=1
    60=<t<71		0.300<=xi<=1
    71=<t<82		0.280<=xi<=1
    82=<t<93		0.240<=xi<=1
    93=< t=<204	0.220<=xi<=1 """

    # Convert NaOH mass fraction to water mass fraction
    xi = 1 - x

    # Coefficients
    k = np.array([1288.4485, -0.49649131, -4387.8908, -4.0915144, 4938.2298, 7.2887292, -1841.1890, -3.0202651])
    l = np.array([2.3087919, -9.0004252, 167.59914, -1051.6368, 3394.3378, -6115.0986, 6220.8249, -3348.8098, 743.87432])
    m = np.array([0.02302860, -0.37866056, 2.4529593, -8.2693542, 15.728833, -16.944427, 9.6254192, -2.2410628])
    n = np.array([-8.5131313e-5, 136.52823e-5, -875.68741e-5, 2920.0398e-5, -5488.2983e-5, 5841.8034e-5, -3278.7483e-5, 754.45993e-5])

    # Calculation of coefficients
    c1 = (k[0] + k[2]*xi + k[4]*(xi**2) + k[6]*(xi**3)) / (1 + k[1]*xi + k[3]*(xi**2) + k[5]*(xi**3) + k[7]*(xi**4))
    c2 = l[0] + l[1]*xi + l[2]*(xi**2) + l[3]*(xi**3) + l[4]*(xi**4) + l[5]*(xi**5) + l[6]*(xi**6) + l[7]*(xi**7) + l[8]*(xi**8)
    c3 = m[0] + m[1]*xi + m[2]*(xi**2) + m[3]*(xi**3) + m[4]*(xi**4) + m[5]*(xi**5) + m[6]*(xi**6) + m[7]*(xi**7)
    c4 = n[0] + n[1]*xi + n[2]*(xi**2) + n[3]*(xi**3) + n[4]*(xi**4) + n[5]*(xi**5) + n[6]*(xi**6) + n[7]*(xi**7)

    # Calculate enthalpy
    h = c1 + c2*T + c3*(T**2) + c4*(T**3)

    return h

def dhdx_NaOH(x,T):
    # Author: Elisabeth Thiele
    # brought to python by Dorian Höffner
    # partielles Differential dh/dx

    # Variables:
    # T:        Temperatur            in °C
    # x:        NaOH concentration in kg NaOH/kg solution
    # dhdx:     partielle Ableitung Enthalpie nach Massenanteil in [kJ/kg] EINHEIT?

    delta = 0.000001
    x1 = x - delta
    x2 = x + delta

    dhdx = (EnthalpieNaOH(x2,T) - EnthalpieNaOH(x1,T)) / (2*delta)
    return dhdx

def dhdT_NaOH(x,T):
    # Author: Elisabeth Thiele
    # partielles Differential dh/dx

    # Variables:
    # T:        Temperatur [°C]
    # x:        NaOH concentration [kg NaOH/kg solution]
    # dhdx:     partielle Ableitung Enthalpie nach Temperatur in [kJ/kgK] EINHEIT?

    delta = 0.0001
    T1 = T - delta
    T2 = T + delta

    dhdT = (EnthalpieNaOH(x,T1) - EnthalpieNaOH(x,T2)) / (2*delta)
    return dhdT

def cp_NaOH(x,T):
    # Author: Dorina Höffner
    
    # Variables:
    # T:        Temperatur [°C]
    # x:        NaOH concentration [kg NaOH/kg solution]
    # cp:       specific heat capacity [kJ/kgK]

    # cp is the partial derivative of the enthalpy
    # with respect to temperature at constant pressure
    cp = -dhdT_NaOH(x,T)

    return cp


#########################################################
################# RECUPERATION FUNCTION #################
#########################################################

# Recuperator function used by "run_absorption" and "run_desorption"
def recuperation(T_AD_out, T_AD_from_tank, m_sol_in, m_sol_out, cp_in, cp_out, U_hx, A_hx):


    # Ensure that the inputs are within valid physical ranges
    if T_AD_out < 0:
        raise ValueError("The temperature of the solution leaving the AD must be positive.")
    if T_AD_from_tank < 0:
        raise ValueError("The temperature of the solution leaving the tank must be positive.")
    if m_sol_in <= 0:
        raise ValueError("The mass flow rate of the solution entering the AD must be positive.")
    if m_sol_out <= 0:
        raise ValueError("The mass flow rate of the solution leaving the AD must be positive.")
    if cp_in <= 0: 
        raise ValueError("The specific heat capacity of the solution entering the AD must be positive.")
    if cp_out <= 0:
        raise ValueError("The specific heat capacity of the solution leaving the AD and the heat exchanger must be positive.")
    if A_hx < 0 or U_hx < 0:
        raise ValueError("The heat exchanger area and U-value must be non-negative.")
    # if T_AD_out < T_AD_from_tank:
    #     raise ValueError("The temperature of the solution leaving the AD must be higher than the temperature of the solution leaving the tank.")


    ### Temperature Explanation
    # T_AD_out = T_hot
    # T_AD_from_tank = T_cold

    # determine C_min
    if m_sol_in * cp_in > m_sol_out * cp_out:
        C_min = m_sol_out * cp_out
        C_max = m_sol_in * cp_in
    else:
        C_min = m_sol_in * cp_in
        C_max = m_sol_out * cp_out

    # prepare variables for NTU calculation
    C_r = C_min / C_max
    NTU_hx = U_hx * A_hx / C_min
    Q_max = C_min * (T_AD_out - T_AD_from_tank)

    # Calculate effectiveness (eps) with handling for the case where 1 - C_r approaches zero
    if abs(1 - C_r) < 1e-6:
        eps_hx = NTU_hx / (1 + NTU_hx)
    else:
        eps_hx = (1 - np.exp(-NTU_hx * (1 - C_r))) / (1 - C_r * np.exp(-NTU_hx * (1 - C_r)))

    # Calculate real heat transfer
    Q_real = eps_hx * Q_max
    T_preheated = T_AD_from_tank + Q_real / (m_sol_in * cp_in)

    return T_preheated, NTU_hx, eps_hx #  = T_AD_in [°C] (temperature entering AD)


#########################################################
################# Obj.-Oriented Model ###################
#########################################################


class ThermochemicalStorageSingleNode:
    def __init__(self, params):
        # Parameter collection
        self.all_original_params = params

        # Model parameters
        self.U_EC = params['U_EC']
        self.A_EC = params['A_EC']
        self.U_AD = params['U_AD']
        self.A_AD = params['A_AD']
        self.U_hx = params['U_hx']
        self.A_hx = params['A_hx']
        self.A_pipe = params['A_pipe']
        self.zeta = params['zeta']
        self.beta = params['beta']
        self.m_ext_EC = params['m_ext_EC']
        self.m_ext_AD = params['m_ext_AD']
        self.m_sol_in = params['m_sol_in']
        self.x_in = params['x_in']
        #self.T_AD_in = params['T_AD_in'] # deleted because of recuperator
        self.T_AD_from_tank = params['T_AD_from_tank']
        self.t_EC_in = params['t_EC_in']
        #self.t_AD_out = params['t_AD_out']
        self.t_AD_in = params["t_AD_in"]
        self.m_highC_tank = params['m_highC_tank']
        self.d_film = params['d_film']
        self.lambda_film = params['lambda_film']

        # Adjusted parameter set for desorption
        self.U_AD_des = params['U_AD_des']
        self.m_ext_EC_des = params['m_ext_EC_des']
        self.m_ext_AD_des = params['m_ext_AD_des']
        self.m_sol_in_des = params['m_sol_in_des']
        self.x_in_des = params['x_in_des']
        self.T_AD_in_des = params['T_AD_in_des']
        self.t_EC_in_des = params['t_EC_in_des']
        self.t_AD_in_des = params['t_AD_in_des']

        # Initialize results
        self.absorption_results = None
        self.desorption_results = None
        
    # Absorption calculation    
    def run_absorption(self, verbose=False):
        
        def system_of_equations(variables):
            
            # redefine variables for better readibility of equation system
            U_EC = self.U_EC
            A_EC = self.A_EC
            U_AD = self.U_AD
            A_AD = self.A_AD
            U_hx = self.U_hx
            A_hx = self.A_hx
            A_pipe = self.A_pipe
            zeta = self.zeta
            beta = self.beta
            m_ext_EC = self.m_ext_EC
            m_ext_AD = self.m_ext_AD
            m_sol_in = self.m_sol_in
            x_in = self.x_in
            #T_AD_in = self.T_AD_in
            T_AD_from_tank = self.T_AD_from_tank
            t_EC_in = self.t_EC_in
            t_AD_in = self.t_AD_in
            m_highC_tank = self.m_highC_tank
            d_film = self.d_film
            lambda_film = self.lambda_film

            # TEST IF BOUNDARY CONDITIONS ARE REALISTIC
            # reachable temperature [°C] in AD
            t_AD_reachable = TemperatureNaOH(steamTable.psat_t(t_EC_in) * 10**5, x_in)

          
            if t_AD_in >= t_AD_reachable:
                print("ERROR: t_AD_in is larger than max equilibrium Temperature. Check boundary conditions.")
                print(f"t_AD_in = {t_AD_in}")
                print(f"t_AD_reachable = {t_AD_reachable}")
                return        

            # # ALTERNATIVELY: Check if t_AD_out can be reached physically
            # if t_AD_out > t_AD_reachable:
            #     print("ERROR: t_AD_out is not physically reachable. Check boundary conditions.")
            #     print(f"t_AD_out = {t_AD_out}")
            #     print(f"t_AD_reachable = {t_AD_reachable}")
            #     return     

            # Unpack variables
            t_EC_out, m_vap_AD, T_AD_out = variables

            # set up necessary paremeters
            C_EC_ext = m_ext_EC * steamTable.CpL_t(t_EC_in) * 10**3    # [W/K]
            C_AD_ext = m_ext_AD * steamTable.CpL_t(t_AD_in) * 10**3   # [W/K]
            NTU_EC  =  U_EC * A_EC / C_EC_ext
            NTU_AD  =  U_AD * A_AD / C_AD_ext
            NTU_b   =  beta * A_AD / m_sol_in
            eps_EC  =  1 - np.exp(-NTU_EC)
            eps_AD  =  1 - np.exp(-NTU_AD)
            eps_b   =  1 - np.exp(-NTU_b)
                        

            # calculate model equations
            ### --> evaporator-condenser
            T_EC     = t_EC_in - (t_EC_in - t_EC_out) / eps_EC      # NTU-method for EC
            if T_EC < 0.0:
                raise ValueError(f"T_EC = {T_EC} is below 0°C.\nt_EC_in = {t_EC_in},\nt_EC_out = {t_EC_out},\neps_EC = {eps_EC},\nm_ext_EC = {m_ext_EC},\nNTU_EC = {NTU_EC}.\nCheck boundary conditions.")
                
            hV       = steamTable.hV_t(T_EC) * 10**3                # [J] enthalpy vapor phase (using XSteam library)
            hL       = steamTable.hL_t(T_EC) * 10**3                # [J] enthalpy liquid phase (using XSteam library)
            dh_lv    = hV - hL
            rhoV     = steamTable.rhoV_t(T_EC)
            Q_EC     = C_EC_ext * (t_EC_in - t_EC_out)              # [W]   heat transfer from EC
            m_vap_EC = Q_EC / dh_lv 
            p_EC_sat = steamTable.psat_t(T_EC) * 10**5 # [Pa]
            
            ### --> heat recuperation
            m_sol_out = m_sol_in + m_vap_AD                         # mass balance AD (full)
            x_out     = m_sol_in / m_sol_out * x_in                 # mass balance AD (NaOH)
            cp_in     = cp_NaOH(x_in, T_AD_from_tank) * 10**3       # [J/kgK] specific heat capacity NaOH
            cp_out    = cp_NaOH(x_out, T_AD_out) * 10**3            # [J/kgK] specific heat capacity NaOH
            # Temperature entering AD (coming from recuperator)
            T_AD_in, NTU_hx, eps_hx   = recuperation(T_AD_out, T_AD_from_tank,
                                                     m_sol_in, m_sol_out,
                                                     cp_in, cp_out,
                                                     U_hx, A_hx)
    
            ### --> absorber-desorber
            p_AD     = p_EC_sat - zeta * 0.5 * m_vap_EC**2 / (rhoV * A_pipe**2) # [Pa] pressure in AD
            h_sol_in = EnthalpieNaOH(x_in, T_AD_in) * 10**3 # QUASI-PARAMETER
            #m_sol_out= m_sol_in + m_vap_AD                          # mass balance AD (full)
            #x_out    = m_sol_in / m_sol_out * x_in                  # mass balance AD (NaOH)
            x_AD_sat = x_in - 1/eps_b * (x_in - x_out)              # NTU-method for mass transfer in AD
            T_AD_sat = TemperatureNaOH(p_AD, x_AD_sat)
            #t_AD_in  = (t_AD_out - eps_AD * T_AD_sat) / (1 - eps_AD) # NTU-method for AD
            t_AD_out = t_AD_in + eps_AD * (T_AD_sat - t_AD_in)
            Q_AD     = C_AD_ext * (t_AD_in - t_AD_out)               # [W]   heat transfer from AD
            #T_AD_out = T_AD_sat - 0.25 * d_film * U_AD / lambda_film * (2*T_AD_sat - t_AD_in - t_AD_out) # simplified heat conduction equation for film in AD
            # ABOVE EQUATION IS NOW residual[2]
            h_sol_out= EnthalpieNaOH(x_out, T_AD_out) * 10**3

            # calculate residuals 
            residual = np.zeros(3)
            # MASS BALANCE VAP
            residual[0] = -m_vap_AD + m_vap_EC
            # ENERGY BALANCE A/D
            residual[1] = (-Q_AD - m_sol_in * h_sol_in - m_vap_AD * hV) / m_sol_out + h_sol_out
            # SIMPLIFIED HEAT CONDUCTION EQUATION FOR FILM IN AD
            residual[2] = - T_AD_out + T_AD_sat - 0.25 * d_film * U_AD / lambda_film * (2*T_AD_sat - t_AD_in - t_AD_out)

            # calculate capacity for steady-state conditions
            t_discharge = self.m_highC_tank / m_sol_in    # [s]
            capacity = -Q_AD * t_discharge / 3600 / 1000  # [kWh] = [Ws] / 3600 [s/h] / 1000 [W/kW]
            
            # Create a dictionary of all calculated variables
            variables_dict = {
                "t_EC_out": t_EC_out,
                "m_vap_AD": m_vap_AD,
                "T_EC": T_EC,
                "hV": hV,
                "hL": hL,
                "dh_lv": dh_lv,
                "rhoV": rhoV,
                "Q_EC": Q_EC,
                "m_vap_EC": m_vap_EC,
                "m_ext_EC": m_ext_EC,
                "m_ext_AD": m_ext_AD,
                "p_EC_sat": p_EC_sat,
                "p_AD": p_AD,
                "h_sol_in": h_sol_in,
                "m_sol_in": m_sol_in,
                "m_sol_out": m_sol_out,
                "x_in": x_in,
                "x_out": x_out,
                "x_AD_sat": x_AD_sat,
                "T_AD_sat": T_AD_sat,
                "t_AD_in": t_AD_in,
                "t_AD_out": t_AD_out,
                "Q_AD": Q_AD,
                "T_AD_from_tank": T_AD_from_tank,
                "T_AD_in": T_AD_in, # added with recuperator
                "T_AD_out": T_AD_out,
                "h_sol_out": h_sol_out,
                "C_EC_ext": C_EC_ext,
                "C_AD_ext": C_AD_ext,
                "NTU_EC": NTU_EC,
                "NTU_AD": NTU_AD,
                "NTU_b": NTU_b,
                "NTU_hx": NTU_hx,
                "eps_EC": eps_EC,
                "eps_AD": eps_AD,
                "eps_b": eps_b,
                "eps_hx": eps_hx,
                "t_discharge":t_discharge,
                "t_discharge_h":t_discharge/3600,
                "capacity":capacity
            }

            return residual, variables_dict

        
        ### RUN MODEL
        # initial guess: [t_EC_out, m_vap_AD, T_AD_out]
        t_EC_out_guess = self.t_EC_in - 1.0
        vars_guess = [t_EC_out_guess, 0.00, 25.0]

        # Wrapper function for fsolve that discards the variables_dict when solving
        def equations(vars_guess):
            residual, _ = system_of_equations(variables=vars_guess)
            return residual
        
        # solve
        vars_solution = fsolve(equations, vars_guess)
        
        # Get the full output including all calculated variables
        _, absorption_results = system_of_equations(vars_solution)
        
        self.absorption_results = absorption_results
        
        
        ### IF VERBOSE PRINT RESULTS
        if verbose:
            print("Results from the Absorption Process:\n")
            max_var_length = max(len(variable) for variable in absorption_results.keys())

            # Determine the maximum length required for the numerical part to align decimal points
            max_val_length = max(len(f"{value:.4f}") for value in absorption_results.values())

            for variable, value in absorption_results.items():
                # Align the variable names and colons
                var_with_colon = f"{variable.ljust(max_var_length)} :"

                # Format the value with sufficient padding to align decimal points, and print
                print(f"{var_with_colon} {value:>{max_val_length}.4f}")  # Using > for right alignment

        return absorption_results
        
    # Desorption calculation
    def run_desorption(self, verbose=False):
        
        def system_of_equations(variables):
            
            # redefine variables for better readibility of equation system
            U_EC = self.U_EC
            A_EC = self.A_EC
            U_AD_des = self.U_AD_des
            A_AD = self.A_AD
            U_hx = self.U_hx
            A_hx = self.A_hx
            A_pipe = self.A_pipe
            zeta = self.zeta
            beta = self.beta
            m_ext_EC_des = self.m_ext_EC_des
            m_ext_AD_des = self.m_ext_AD_des
            m_sol_in_des = self.m_sol_in_des
            x_in_des = self.x_in_des
            #T_AD_in_des = self.T_AD_in_des
            T_AD_from_tank = self.T_AD_from_tank
            t_EC_in_des = self.t_EC_in_des
            t_AD_in_des = self.t_AD_in_des
            m_highC_tank = self.m_highC_tank
            d_film = self.d_film
            lambda_film = self.lambda_film

            # Unpack variables
            t_EC_out, m_vap_AD, T_AD_out = variables

            # TEST IF WORKING BOUNDARY CONDITIONS ARE REALISTIC
            ## 1. Check if t_EC_in_des is low enough
            if t_EC_in_des >= steamTable.tsat_p(PressureNaOH(x_in_des, t_AD_in_des) / 10**5):
                print("ERROR: t_EC_in_des is not low enough or t_AD_in_des too low. Check boundary conditions.")
                print(f"t_EC_in_des = {t_EC_in_des}")
                print(f"t_sat_p = {steamTable.tsat_p(PressureNaOH(x_in_des, T_AD_in_des)) / 10**5}")
                print(f"t_AD_in_des = {t_AD_in_des}")
                return
            
            # set up necessary paremeters
            C_EC_ext = m_ext_EC_des * steamTable.CpL_t(t_EC_in_des) * 10**3    # [W/K]
            C_AD_ext = m_ext_AD_des * steamTable.CpL_t(t_AD_in_des) * 10**3   # [W/K]

            NTU_EC_des  =  U_EC * A_EC / C_EC_ext
            NTU_AD_des  =  U_AD_des * A_AD / C_AD_ext
            eps_EC_des  =  1 - np.exp(-NTU_EC_des)
            eps_AD_des  =  1 - np.exp(-NTU_AD_des)
            # SHOULD BE CALCULATED LATER with m_sol_avg
            NTU_b_des   =  beta * A_AD / m_sol_in_des
            eps_b_des   =  1 - np.exp(-NTU_b_des)           
            

            ### calculate model equations
            # --> CONDENSER
            T_EC     = t_EC_in_des - (t_EC_in_des - t_EC_out) / eps_EC_des
            if T_EC < 0.0 or T_EC!=T_EC:
                raise ValueError(f"T_EC = {T_EC} is below 0°C.\nt_EC_in_des = {t_EC_in_des},\nt_EC_out = {t_EC_out},\neps_EC = {eps_EC_des},\nm_ext_EC = {m_ext_EC_des},\nNTU_EC = {NTU_EC_des}.\nCheck boundary conditions.")
            hV       = steamTable.hV_t(T_EC) * 10**3    # [J] enthalpy vapor phase (using XSteam library)
            hL       = steamTable.hL_t(T_EC) * 10**3    # [J] enthalpy liquid phase (using XSteam library)
            dh_lv    = hV - hL
            rhoV     = steamTable.rhoV_t(T_EC)
            Q_EC     = C_EC_ext * (t_EC_in_des - t_EC_out)
            m_vap_EC = -Q_EC / dh_lv # changed sign for positive massflow
            p_EC_sat = steamTable.psat_t(T_EC) * 10**5 # [Pa]
            # --> RECUPERATOR
            m_sol_out = m_sol_in_des - m_vap_AD                         # mass balance AD (full)
            x_out    = m_sol_in_des / m_sol_out * x_in_des          # mass balance AD (NaOH)
            cp_in     = cp_NaOH(x_in_des, T_AD_from_tank) * 10**3   # [J/kgK] specific heat capacity NaOH
            cp_out    = cp_NaOH(x_out, T_AD_out) * 10**3            # [J/kgK] specific heat capacity NaOH
            T_AD_in_des, NTU_hx_des, eps_hx_des   = recuperation(T_AD_out, T_AD_from_tank,
                                                         m_sol_in_des, m_sol_out,
                                                         cp_in, cp_out,
                                                         U_hx, A_hx)
            # --> DESORBER
            p_AD     = p_EC_sat + zeta * 0.5 * m_vap_EC**2 / (rhoV * A_pipe**2)
            h_sol_in = EnthalpieNaOH(x_in_des, T_AD_in_des) * 10**3 # QUASI-PARAMETER
            #m_sol_out= m_sol_in_des - m_vap_AD # changed sign before m_vap_AD
            #x_out    = m_sol_in_des / m_sol_out * x_in_des
            x_AD_sat = x_in_des - 1/eps_b_des * (x_in_des - x_out)
            T_AD_sat = TemperatureNaOH(p_AD, x_AD_sat)
            t_AD_out = t_AD_in_des - eps_AD_des * (t_AD_in_des - T_AD_sat) # changed completely
            Q_AD     = C_AD_ext * (t_AD_in_des - t_AD_out)
            #T_AD_out = T_AD_sat - 0.25 * d_film * U_AD_des / lambda_film * (2*T_AD_sat - t_AD_in_des - t_AD_out)
            # ABOVE EQUATION IS NOW RESIDUAL(2)
            h_sol_out= EnthalpieNaOH(x_out, T_AD_out) * 10**3


            # calculate residuals 
            residual = np.zeros(3)#
            # MASS BALANCE VAP
            residual[0] = -m_vap_AD + m_vap_EC
            # ENERGY BALANCE A/D
            residual[1] = (Q_AD + m_sol_in_des * h_sol_in - m_vap_AD * hV) / m_sol_out - h_sol_out
            # SIMPLIFIED HEAT CONDUCTION EQUATION FOR FILM IN AD
            residual[2] = - T_AD_out + T_AD_sat - 0.25 * d_film * U_AD_des / lambda_film * (2*T_AD_sat - t_AD_in_des - t_AD_out)

            # calculate capacity for steady-state conditions
            t_charge = self.m_highC_tank / m_sol_in_des        # [s]
            charged_energy = Q_AD * t_charge / 3600 / 1000  # [kWh] = [Ws] / 3600 [s/h] / 1000 [W/kW]
            
            # Create a dictionary of all calculated variables
            variables_dict = {
                "t_EC_out": t_EC_out,
                "m_vap_AD": m_vap_AD,
                "T_EC": T_EC,
                "hV": hV,
                "hL": hL,
                "dh_lv": dh_lv,
                "rhoV": rhoV,
                "Q_EC": Q_EC,
                "m_vap_EC": m_vap_EC,
                "m_ext_EC_des": m_ext_EC_des,
                "m_ext_AD_des": m_ext_AD_des,
                "p_EC_sat": p_EC_sat,
                "p_AD": p_AD,
                "h_sol_in": h_sol_in,
                "m_sol_out": m_sol_out,
                "x_out": x_out,
                "x_AD_sat": x_AD_sat,
                "T_AD_sat": T_AD_sat,
                "t_AD_out": t_AD_out,
                "Q_AD": Q_AD,
                "T_AD_from_tank": T_AD_from_tank,
                "T_AD_in": T_AD_in_des,
                "T_AD_out": T_AD_out,
                "h_sol_out": h_sol_out,
                "C_EC_ext": C_EC_ext,
                "C_AD_ext": C_AD_ext,
                "NTU_EC_des": NTU_EC_des,
                "NTU_AD_des": NTU_AD_des,
                "NTU_b_des": NTU_b_des,
                "NTU_hx_des": NTU_hx_des,
                "eps_EC_des": eps_EC_des,
                "eps_AD_des": eps_AD_des,
                "eps_b_des": eps_b_des,
                "eps_hx_des": eps_hx_des,
                "t_charge": t_charge,
                "t_charge_h": t_charge/3600,
                "charged_energy":charged_energy
            }

            return residual, variables_dict

        
        ### RUN MODEL
        # initial guess [t_EC_in, m_vap_AD, T_AD_out]
        t_EC_out_guess = self.t_EC_in_des + 1.0
        vars_guess = [t_EC_out_guess, 0.00, 50.0]

        # Wrapper function for fsolve that discards the variables_dict when solving
        def equations(vars_guess):
            residual, _ = system_of_equations(variables=vars_guess)
            return residual
                
        # solve
        vars_solution = fsolve(equations, vars_guess)
        
        # Get the full output including all calculated variables
        _, desorption_results = system_of_equations(vars_solution)
        
        self.desorption_results = desorption_results
        
        
        ### PRINT RESULTS
        if verbose:
            print("Results from the Desorption Process:\n")
            max_var_length = max(len(variable) for variable in desorption_results.keys())

            # Determine the maximum length required for the numerical part to align decimal points
            max_val_length = max(len(f"{value:.4f}") for value in desorption_results.values())

            for variable, value in desorption_results.items():
                # Align the variable names and colons
                var_with_colon = f"{variable.ljust(max_var_length)} :"

                # Format the value with sufficient padding to align decimal points, and print
                print(f"{var_with_colon} {value:>{max_val_length}.4f}")  # Using > for right alignment
                
            
        return desorption_results

    # Parameter study with plotting options
    def single_parameter_variation(self, variable_name, variable_array, mode, plot_variable=None, editable_plot=False, color=None):
        results_list = []
        params_frames = []  # Use a list to collect DataFrames

        for value in variable_array:
            setattr(self, variable_name, value)  # Change parameter

            # Run the model based on the specified mode
            if mode == "absorption":
                results = self.run_absorption(verbose=False)
            elif mode == "desorption":
                results = self.run_desorption(verbose=False)
            else:
                print("ERROR: Invalid Mode. CHOOSE mode='absorption' OR mode='desorption'")
                return

            results_list.append(results)  # Append results

            # Extract current parameters
            params_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}

            # Convert params_dict to DataFrame and store in list
            params_frames.append(pd.DataFrame([params_dict]))

            # Progress tracking
            #print(f"Progress: {variable_array.index(value) + 1} of {len(variable_array)}")

        # Reset to original parameter set
        setattr(self, variable_name, self.all_original_params[variable_name])

        # Concatenate all parameter DataFrames stored in the list into a single DataFrame
        params_df = pd.concat(params_frames, ignore_index=True)

        # Convert the results list to a DataFrame
        results_df = pd.DataFrame(results_list)

        if plot_variable:
            ### Plot
            if not editable_plot:
                plt.figure(figsize=(3,3))
                plt.rcParams.update({'font.size': 8})
            else:
                plt.ioff()
            if plot_variable=="Q_AD" and mode=="absorption":
                results_df["Q_AD"] = -results_df["Q_AD"]
            if color:
                plt.plot(variable_array, results_df[plot_variable], color=color)
            else:
                plt.plot(variable_array, results_df[plot_variable])
            plt.xlabel(variable_name)#, fontsize=8)
            plt.ylabel(plot_variable)#, fontsize=8)
            if not editable_plot:
                plt.show()
        
        if not editable_plot:
            return params_df, results_df

    # Parameter study of two parameters at the same time with plotting options 
    def two_parameters_variation(self, variable_name1, variable_array1, variable_name2, variable_array2,
                                 plot_variable=None,
                                 mode=None,
                                 editable_plot=False,
                                 cmap='viridis'):

        params_frames = []  # List to collect DataFrames of parameters
        results_dfs = [[None for _ in range(len(variable_array2))] for _ in range(len(variable_array1))]
        
        for i, value1 in enumerate(variable_array1):
            for j, value2 in enumerate(variable_array2):
                # Change the parameters
                setattr(self, variable_name1, value1)
                setattr(self, variable_name2, value2)

                # Run the model (assuming absorption mode for simplicity; adjust as needed)
                if mode == "absorption":
                    results = self.run_absorption(verbose=False)
                elif mode == "desorption":
                    results = self.run_desorption(verbose=False)
                else:
                    raise ValueError(f"Invalid mode: {mode}. Choose 'absorption' or 'desorption'.")


                # Append results to the list
                results_dfs[i][j] = pd.Series(results)

                # Extract current parameters and store in a dictionary
                params_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}

                # Convert params_dict to DataFrame and store in the list
                params_frames.append(pd.DataFrame([params_dict]))

                # Optional: Progress tracking can be added here

        # Reset to original parameter sets
        setattr(self, variable_name1, self.all_original_params[variable_name1])
        setattr(self, variable_name2, self.all_original_params[variable_name2])

        # Concatenate all parameter DataFrames stored in the list into a single DataFrame
        params_df = pd.concat(params_frames, ignore_index=True)

        if plot_variable:

            ### Plot
            # prepare data
            data_matrix = np.zeros([len(variable_array1), len(variable_array2)])

            for i in range(len(variable_array1)):
                for j in range(len(variable_array2)):
                    # Access the stored result, ensure plot_variable exists as a key, and negate its value for plotting
                    if plot_variable=="Q_AD" and mode=="absorption":
                        data_matrix[i, j] = -results_dfs[i][j][plot_variable]
                    else:
                        data_matrix[i, j] = results_dfs[i][j][plot_variable]

            if not editable_plot:
                plt.figure(figsize=(5,3))

            # Get the colormap
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(variable_array1)))

            for i, val in enumerate(variable_array1):
                # Correctly using variable_array2 for the x-axis, and matching y-axis data from data_matrix
                plt.plot(variable_array2, data_matrix[i, :], label=f"{variable_name1}={np.round(val,3)}", color=colors[i])
            plt.xlabel(variable_name2)
            plt.ylabel(plot_variable)

            # Adjust the legend to be outside the plot
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.33)
            #plt.tight_layout()
            if not editable_plot:
                plt.show()
                return params_df, results_dfs
            
        else:
            return params_df, results_dfs

    
