import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# SUBSTANCE PROPERTIES
import propertiesNaOH as NaOH                # this is our own module available on pypi
from pyXSteam.XSteam import XSteam           # this is the XSteam module available on pypi
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W


print("Model Imported Successfully!")


#########################################################
################# Obj.-Oriented Model ###################
#########################################################


class SorptionStorageSystem:
    def __init__(self, folder_path="./"):
        # parameter collection
        paths = [f"{folder_path}/parameters.xlsx", f"{folder_path}/variables.xlsx", f"{folder_path}/correlation_parameters.xlsx"]
        
        params          = pd.read_excel(paths[0], index_col=0)
        variables       = pd.read_excel(paths[1], index_col=0)
        params_corr     = pd.read_excel(paths[2], index_col=0)

        for df in [params, variables, params_corr]:
            # if contains nan, raise error
            if df["Value"].isnull().values.any():
                raise ValueError(f"Dataframe contains NaN values:\n{df}")    

        params = params.to_dict()['Value']
        variables = variables.to_dict()['Value']
        params_corr = params_corr.to_dict()['Value']
        
        
        # Parameter collection of all dictionaries (one dict)
        self.all_original_params = {**params, **variables, **params_corr}

        ### Model PARAMETERS
        # EC: Evaporator-condenser parameters
        self.di_tube_EC = params['di_tube_EC']
        self.do_tube_EC = params['do_tube_EC']
        self.l_tube_EC = params['l_tube_EC']
        self.A_EC = params['A_EC']                   # FLAG  
        self.nrow_EC = params['nrow_EC']
        self.ncol_EC = params['ncol_EC']
        #self.A_EC = self.nrow_EC * self.ncol_EC * np.pi * self.do_tube_EC * self.l_tube_EC
        self.lambda_tubeEC = params['lambda_tubeEC']
        self.lambda_h2o = params['lambda_h2o']
        self.A_EC_amb = params['A_EC_amb']
        # AD: Absorber-desorber parameters
        self.di_tube_AD = params['di_tube_AD']
        self.do_tube_AD = params['do_tube_AD']
        self.l_tube_AD = params['l_tube_AD']
        self.nrow_AD = params['nrow_AD']
        self.ncol_AD = params['ncol_AD']
        self.A_AD = self.nrow_AD * self.ncol_AD * np.pi * self.do_tube_AD * self.l_tube_AD
        self.lambda_tubeAD = params['lambda_tubeAD']
        self.lambda_naoh = params['lambda_naoh']
        self.A_AD_amb = params['A_AD_amb']
        # CP: Connection pipe parameters
        self.A_pipe = params['A_pipe']
        # RE: Recuperator parameters
        self.A_rec = params['A_rec']
        # System Volume
        self.V_system = params['V_system']
        

        ### FITTED PARAMETERS
        self.a1 = params_corr['a1']     # for alpha EC
        self.a2 = params_corr['a2']     # for alpha EC
        self.b1 = params_corr['b1']     # for alpha AD
        self.b2 = params_corr['b2']     # for alpha AD
        self.b3 = params_corr['b3']     # for alpha AD
        self.b4 = params_corr['b4']     # for alpha AD
        self.c1 = params_corr['c1']     # for beta (AD)
        self.c2 = params_corr['c2']     # for beta (AD)
        self.d1 = params_corr['d1']     # for wetting factor, but currently unused
        self.d2 = params_corr['d2']     # for wetting factor, but currently unused
        self.e1 = params_corr['e1']     # for fitting T_AD_out (heat conduction equation film at outlet)
        self.e2 = params_corr['e2']     # for fitting T_AD_out (heat conduction equation film at outlet)
        # other parameters for fitting
        self.alpha_film_EC  = params_corr['alpha_film_EC']
        self.alpha_film_AD  = params_corr['alpha_film_AD']
        self.U_AD_amb       = params_corr['U_AD_amb']
        self.U_EC_amb       = params_corr['U_EC_amb']
        self.U_rec          = params_corr['U_rec']
        self.zeta           = params_corr['zeta']
        # TEMPORARY FIT PARAMETERS
        self.eps_AD         = params_corr['eps_AD']
        self.eps_b          = params_corr['eps_b']

        ### SOME NOTES FOR CORRELATIONS
        # for alpha AD:
        # --> alpha_AD = cAD * Re_GAMMA^(n1_AD) * Pr^(n2_AD)
        # --> with Re_GAMMA = 4*GAMMA/mu ]]
        # --> wieth GAMMA = m_vap_AD/2nl [kg/ms]
        # for alpha EC:
        # --> alpha_EC = cEC * Re_GAMMA^(n1_EC) * Pr^(n2_EC)
        # --> ALTERNATIVE MÖGLICHKEIT: alpha_EC = cEC * Re_GAMMA^(n1_EC) * Pr^(n2_EC) * Ar^(n3_EC)
        # for beta: Sh = Nu * (Sc/Pr)^n

        ### Model VARIABLES
        self.m_ext_EC       = variables['m_ext_EC']
        self.m_ext_AD       = variables['m_ext_AD']
        self.m_h2o          = variables['m_h2o']
        self.m_sol_in       = variables['m_sol_in']
        self.x_in           = variables['x_in']
        self.T_amb          = variables['T_amb']
        self.T_AD_from_tank = variables['T_AD_from_tank']
        self.t_EC_in        = variables['t_EC_in']
        self.t_AD_in        = variables["t_AD_in"]
        self.m_highC_tank   = variables['m_highC_tank']
        self.p_EC           = variables['p_EC']
        self.p_AD           = variables['p_AD']
        
        # Initialize results
        self.absorption_results = None
        self.desorption_results = None
        
    # Absorption calculation    
    def run_absorption(self, AD_only=False, verbose=False, turn_off_warnings=False):
        
        def system_of_equations(variables):
            
            ### redefine variables for better readibility of equation system
            # GEOMETRY
            di_tube_EC      = self.di_tube_EC
            do_tube_EC      = self.do_tube_EC
            l_tube_EC       = self.l_tube_EC
            A_EC            = self.A_EC            # FLAG
            nrow_EC         = self.nrow_EC
            ncol_EC         = self.ncol_EC
            lambda_tubeEC   = self.lambda_tubeEC
            lambda_h2o      = self.lambda_h2o
            V_system        = self.V_system

            di_tube_AD      = self.di_tube_AD
            do_tube_AD      = self.do_tube_AD
            l_tube_AD       = self.l_tube_AD
            nrow_AD         = self.nrow_AD
            ncol_AD         = self.ncol_AD
            lambda_tubeAD   = self.lambda_tubeAD
            lambda_naoh     = self.lambda_naoh
            
            U_rec        = self.U_rec
            A_rec        = self.A_rec
            T_amb        = self.T_amb
            U_EC_amb     = self.U_EC_amb
            A_EC_amb     = self.A_EC_amb
            U_AD_amb     = self.U_AD_amb
            A_AD_amb     = self.A_AD_amb
            A_pipe       = self.A_pipe
            zeta         = self.zeta
            m_highC_tank = self.m_highC_tank
            
            ### PROCESS VARIABLES
            m_ext_EC = self.m_ext_EC
            m_ext_AD = self.m_ext_AD
            m_h2o = self.m_h2o
            m_sol_in = self.m_sol_in
            x_in = self.x_in
            T_AD_from_tank = self.T_AD_from_tank
            t_EC_in = self.t_EC_in
            t_AD_in = self.t_AD_in
            p_EC = self.p_EC
            p_AD = self.p_AD

            ### FITTED PARAMETERS
            a1 = self.a1           # for alpha EC
            a2 = self.a2           # for alpha EC
            b1 = self.b1           # for alpha AD
            b2 = self.b2           # for alpha AD
            b3 = self.b3           # for alpha AD
            b4 = self.b4           # for alpha AD
            c1 = self.c1           # for beta
            c2 = self.c2           # for beta
            d1 = self.d1           # for effective heat transfer area EC
            d2 = self.d2           # for effective heat transfer area AD
            e1 = self.e1           # fitting T_AD_out (heat conduction equation film at outlet)
            e2 = self.e2           # fitting T_AD_out (heat conduction equation film at outlet)
            alpha_film_EC = self.alpha_film_EC
            alpha_film_AD = self.alpha_film_AD

            # TEST: Check if t_AD_in is <= reachable Temperature
            t_AD_reachable = NaOH.saturation_temperature(steamTable.psat_t(t_EC_in) * 10**5, x_in)
            if not turn_off_warnings:
                if t_AD_in >= t_AD_reachable:
                    print("ERROR: t_AD_in is larger than max equilibrium Temperature. Check boundary conditions.")
                    print(f"t_AD_in        = {t_AD_in}")
                    print(f"t_AD_reachable = {t_AD_reachable}")
                    return        

            
            # Unpack guessed variables
            if AD_only:
                m_vap_AD, T_AD_out = variables
            else:
                t_EC_out, m_vap_AD, T_AD_out = variables

            # Calculate specific solution mass flow
            GAMMA_AD = m_sol_in / (2 * ncol_AD * l_tube_AD) # [kg/ms]

            # Calculate heat transfer coefficients
            my_water = 0.001 # [Pa s] approx. dynamic viscosity of water at 20°C
            cp_water = 4180  # [J/kgK] specific heat capacity of water
            alpha_pipeflow_EC = htc_convective_pipeflow(m_ext_EC, di_tube_EC, my_water, cp_water, lambda_h2o, mode="cooled")
            alpha_pipeflow_AD = htc_convective_pipeflow(m_ext_AD, di_tube_AD, my_water, cp_water, lambda_naoh, mode="heated")
            R_tubewall_EC = thermal_resistance_tubewall(di_tube_EC, do_tube_EC, l_tube_EC, lambda_tubeEC)
            R_tubewall_AD = thermal_resistance_tubewall(di_tube_AD, do_tube_AD, l_tube_AD, lambda_tubeAD)
            alpha_film_AD     = htc_horizontal_falling_film(GAMMA_AD, do_tube_AD, my_water, cp_water, lambda_naoh, b1, b2)
            U_EC              = 1 / (1 / alpha_pipeflow_EC + R_tubewall_EC + 1 / alpha_film_EC)
            U_AD              = 1 / (1 / alpha_pipeflow_AD + R_tubewall_AD + 1 / alpha_film_AD)

            # Calculate mass transfer coefficient
            Diff_NaOH = 1.5e-10
            beta      = mass_transfer_coefficient(GAMMA_AD, Diff_NaOH, my_water, do_tube_AD, rho_NaOH(x_in), c1, c2)

            #A_EC = nrow_EC * ncol_EC * np.pi * do_tube_EC * l_tube_EC # [m^2] # FLAG
            A_AD = nrow_AD * ncol_AD * np.pi * do_tube_AD * l_tube_AD # [m^2]

            # maybe add correction factor for effective heat transfer area

            # set up necessary paremeters
            C_EC_ext = m_ext_EC * steamTable.CpL_t(t_EC_in) * 10**3    # [W/K]
            C_AD_ext = m_ext_AD * steamTable.CpL_t(t_AD_in) * 10**3    # [W/K]
            NTU_EC  =  U_EC * A_EC / C_EC_ext
            NTU_AD  =  U_AD * A_AD / C_AD_ext
            NTU_b   =  beta * A_AD / m_sol_in
            # AUßERDEM: LOG MEAN CONCENTRATION DIFFERENCE HERLEITUNG VON NTU FÜR MASS TRANSFER
            # beta * A * LMCD = m_sol_water * (Beladung_in - Beladung_out)
            eps_EC  =  1 - np.exp(-NTU_EC)
            eps_AD  =  1 - np.exp(-NTU_AD)
            #eps_b   =  1 - np.exp(-NTU_b) ### FLAG: in current model, mass transfer seems not inhibiting
            eps_b = 1.0
            # eps_AD = self.eps_AD
            # eps_b  = self.eps_b

            if not turn_off_warnings:
                # TEST IF WORKING CONDITIONS ARE REALISTIC
                ## 1. Check if epsilon values are in bounds to prevent zero division
                if eps_EC < 0.001:
                    print("ERROR: eps_EC is too low. Check boundary conditions.")
                    print(f"eps_EC = {eps_EC}")
                    return
                if eps_AD < 0.001:
                    print("ERROR: eps_AD is too low. Check boundary conditions.")
                    print(f"eps_AD = {eps_AD}")
                    return
                if eps_b < 0.001:
                    print("ERROR: eps_b is too low. Check boundary conditions.")
                    print(f"eps_b = {eps_b}")
                    return
                        

            # CALCULATE MODEL EQUATIONS

            # conventions
            # 1. positive Q = inward, negative Q = outward)
            # 2. mass flows always positively defined
            
            ### --> evaporator-condenser
            if AD_only:
                T_EC_new = steamTable.tsat_p(p_EC / 10**5)              # [°C]   saturation temperature water    
                hV       = steamTable.hV_t(T_EC_new) * 10**3            # [J] enthalpy vapor phase (using XSteam library)
                hL       = steamTable.hL_t(T_EC_new) * 10**3            # [J] enthalpy liquid phase (using XSteam library)
                dh_lv    = hV - hL
                rhoV     = steamTable.rhoV_t(T_EC_new)                  # [kg/m^3] density of vapor phase

            else:
                T_EC     = t_EC_in - (t_EC_in - t_EC_out) / eps_EC      # NTU-method for EC
                if T_EC < 0.0:
                    raise ValueError(f"T_EC = {T_EC} is below 0°C.\nt_EC_in = {t_EC_in},\nt_EC_out = {t_EC_out},\neps_EC = {eps_EC},\nm_ext_EC = {m_ext_EC},\nNTU_EC = {NTU_EC}.\nCheck boundary conditions.")
                    
                hV       = steamTable.hV_t(T_EC) * 10**3                # [J/K] enthalpy vapor phase (using XSteam library)
                hL       = steamTable.hL_t(T_EC) * 10**3                # [J/K] enthalpy liquid phase (using XSteam library)
                dh_lv    = hV - hL
                rhoV     = steamTable.rhoV_t(T_EC)                      # [kg/m^3] density of vapor phase at T_EC equilibrium
                Q_EC     = C_EC_ext * (t_EC_in - t_EC_out)              # [W]    heat transfer from EC
                Q_EC_amb = U_EC_amb * A_EC_amb * (T_amb - T_EC)         # [W]    heat transfer to environment (positive=gain, negative=loss)
                m_vap_EC = (Q_EC + Q_EC_amb) / dh_lv                    # [kg/s] mass flow rate of evaporated water (from energy balance)
                p_EC     = steamTable.psat_t(T_EC) * 10**5              # [Pa]   saturation pressure water


            ### --> heat recuperation
            m_sol_out = m_sol_in + m_vap_AD                             # mass balance AD (full)
            x_out     = m_sol_in / m_sol_out * x_in                     # mass balance AD (NaOH)
            cp_in     = NaOH.specific_heat_capacity(x_in, T_AD_from_tank) * 10**3       # [J/kgK] specific heat capacity NaOH
            cp_out    = NaOH.specific_heat_capacity(x_out, T_AD_out) * 10**3            # [J/kgK] specific heat capacity NaOH
            
            # Temperature entering AD (coming from recuperator)
            T_AD_in, T_AD_to_tank, NTU_rec, eps_hx = recuperation(T_AD_out, T_AD_from_tank,
                                                                  m_sol_in, m_sol_out,
                                                                  cp_in, cp_out,
                                                                  U_rec, A_rec)
    
            ### --> absorber-desorber
            
            # overwrite p_AD, if EC is considered!
            #if not AD_only:
            p_AD  = p_EC - zeta * 0.5 * m_vap_AD**2 / (rhoV * A_pipe**2)     # [Pa] pressure in AD

            h_sol_in  = NaOH.enthalpy(x_in, T_AD_in) * 10**3                     # QUASI-PARAMETER
            #h_sol_in = NaOH.specific_heat_capacity(x_in, T_AD_in) * T_AD_in
            x_AD_sat  = x_in - 1/eps_b * (x_in - x_out)                          # NTU-method for mass transfer in AD
            T_AD_sat  = NaOH.saturation_temperature(p_AD, x_AD_sat)
            t_AD_out  = t_AD_in + eps_AD * (T_AD_sat - t_AD_in)
            Q_AD      = C_AD_ext * (t_AD_in - t_AD_out)                          # [W] heat transfer from AD
            h_sol_out = NaOH.enthalpy(x_out, T_AD_out) * 10**3
            #h_sol_out = NaOH.specific_heat_capacity(x_out, T_AD_out) * T_AD_out

            # heat losses/gains to environment from AD
            Q_AD_amb = U_AD_amb * A_AD_amb * (T_amb - T_AD_sat)                  # [W] heat transfer to environment (positive=gain, negative=loss)

            # calculate residuals 
            if AD_only:
                residual = np.zeros(2)
                # ENERGY BALANCE A/D
                residual[0] = Q_AD + Q_AD_amb + m_vap_AD * hV   + m_sol_in * h_sol_in - m_sol_out * h_sol_out
                # SIMPLIFIED HEAT CONDUCTION EQUATION FOR FILM IN AD
                d_film = 0.0005 # [m] film thickness FLAG: dynamize this!
                residual[1] = - T_AD_out + T_AD_sat - 0.25 * d_film * U_AD / lambda_naoh * (2*T_AD_sat - t_AD_in - t_AD_out)
                #residual[1] = - T_AD_out + T_AD_sat - d_film * U_AD / lambda_naoh * (T_AD_sat - t_AD_in)

                # calculate capacity for steady-state conditions
                t_discharge = m_highC_tank / m_sol_in           # [s]
                capacity = -Q_AD * t_discharge / 3600 / 1000    # [kWh] = [Ws] / 3600 [s/h] / 1000 [W/kW]

                # calculate change in solution enthalpy
                dQ_sol_abs = m_sol_in * h_sol_in - m_sol_out * h_sol_out

                # calculate overall energy balance
                #energy_balance    = Q_EC + Q_AD + Q_EC_amb + Q_AD_amb + dQ_sol_abs + m_vap_EC * hL
                #energy_balance_EC = Q_EC + Q_EC_amb + m_vap_EC * hL - m_vap_AD * hV
                energy_balance_AD  = Q_AD + Q_AD_amb + m_vap_AD * hV   + m_sol_in * h_sol_in - m_sol_out * h_sol_out
                # UNIT CHECK........ [W]  + [W]      + [kg/s] * [J/kg] + [kg/s] * [J/kg]     - [kg/s] * [J/kg] = 
            else:
                residual = np.zeros(3)
                # MASS BALANCE VAP
                residual[0] = -m_vap_AD + m_vap_EC
                # ENERGY BALANCE A/D
                residual[1] = (Q_AD + m_sol_in * h_sol_in + m_vap_AD * hV + Q_AD_amb) / m_sol_out - h_sol_out
                # SIMPLIFIED HEAT CONDUCTION EQUATION FOR FILM IN AD
                d_film = 0.0005 # [m] film thickness FLAG: dynamize this!
                residual[2] = - T_AD_out + T_AD_sat - 0.25 * d_film * U_AD / lambda_naoh * (2*T_AD_sat - t_AD_in - t_AD_out)

                # calculate capacity for steady-state conditions
                t_discharge                = m_highC_tank / m_sol_in                        # [s]
                capacity                   = -Q_AD * t_discharge / 3600 / 1000              # [kWh] = [Ws] / 3600 [s/h] / 1000 [W/kW]
                m_lowC_tank                = m_highC_tank * x_in / x_out                    # [kg]  
                V_lowC_tank                = m_lowC_tank / rho_NaOH(x_out)                  # [m^3]
                energy_density_m3_diluted  = capacity / V_lowC_tank                         # [kWh/m^3]
                energy_density_system      = capacity / V_system                            # [kWh/m^3]

                # calculate change in solution enthalpy
                dQ_sol_abs = m_sol_in * h_sol_in - m_sol_out * h_sol_out

                # calculate overall energy balance
                energy_balance    = Q_EC + Q_AD + Q_EC_amb + Q_AD_amb + dQ_sol_abs + m_vap_EC * hL
                energy_balance_EC = Q_EC + Q_EC_amb + m_vap_EC * hL - m_vap_AD * hV
                energy_balance_AD = Q_AD + Q_AD_amb + m_vap_AD * hV + m_sol_in * h_sol_in - m_sol_out * h_sol_out
            
            if AD_only:
                # Define EC-related variables as NaN in AD_only calculations
                Q_EC, Q_EC_amb                      = np.nan, np.nan
                energy_balance, energy_balance_EC   = np.nan, np.nan
                T_EC, t_EC_in, t_EC_out             = np.nan, np.nan, np.nan
                m_vap_EC, m_ext_EC,                 = np.nan, np.nan
                alpha_pipeflow_EC                   = np.nan
                R_tubewall_EC, alpha_film_EC    = np.nan, np.nan
                U_EC, U_EC_amb, A_EC, A_EC_amb      = np.nan, np.nan, np.nan, np.nan
                NTU_EC, eps_EC                      = np.nan, np.nan

            # Create a dictionary of all variables
            variables_dict = {
                "Q_EC": Q_EC,
                "Q_AD": Q_AD,
                "Q_EC_amb": Q_EC_amb,
                "Q_AD_amb": Q_AD_amb,
                "dQ_sol_abs": dQ_sol_abs,
                "energy_balance": energy_balance,
                "energy_balance_EC": energy_balance_EC,
                "energy_balance_AD": energy_balance_AD,
                "capacity":capacity,
                "energy_density_m3_diluted": energy_density_m3_diluted,
                "energy_density_system": energy_density_system,
                "t_discharge":t_discharge,
                "t_discharge_h":t_discharge/3600,
                "m_highC_tank": m_highC_tank,
                "m_lowC_tank": m_lowC_tank,
                "V_lowC_tank": V_lowC_tank,
                
                "x_in": x_in,
                "x_out": x_out,
                "dx": x_in - x_out,
                "x_AD_sat": x_AD_sat,

                "T_amb": T_amb,
                "T_AD_from_tank": T_AD_from_tank,
                "T_AD_sat": T_AD_sat,
                "T_AD_in": T_AD_in, # added with recuperator
                "T_AD_out": T_AD_out,
                "T_AD_to_tank":T_AD_to_tank, # added with recuperator
                "t_AD_in": t_AD_in,
                "t_AD_out": t_AD_out,
                "T_EC": T_EC,
                "t_EC_in": t_EC_in,
                "t_EC_out": t_EC_out,

                "m_vap_EC": m_vap_EC,
                "m_vap_AD": m_vap_AD,
                "m_ext_EC": m_ext_EC,
                "m_ext_AD": m_ext_AD,
                "m_sol_in": m_sol_in,
                "m_sol_out": m_sol_out,

                "p_EC": p_EC,
                "p_AD": p_AD,
                
                "h_sol_in": h_sol_in,
                "h_sol_out": h_sol_out,
                "hV": hV,
                "hL": hL,
                "dh_lv": dh_lv,
                "rhoV": rhoV,
                "C_EC_ext": C_EC_ext,
                "C_AD_ext": C_AD_ext,
                "alpha_pipeflow_EC": alpha_pipeflow_EC,
                "alpha_pipeflow_AD": alpha_pipeflow_AD,
                "R_tubewall_EC": R_tubewall_EC,
                "R_tubewall_AD": R_tubewall_AD,
                "alpha_film_EC": alpha_film_EC,
                "alpha_film_AD": alpha_film_AD,
                "U_EC": U_EC,
                "U_AD": U_AD,
                "beta": beta,
                "U_rec": U_rec,
                "U_EC_amb": U_EC_amb,
                "U_AD_amb": U_AD_amb,
                "A_EC": A_EC,
                "A_AD": A_AD,
                "A_rec": A_rec,
                "A_EC_amb": A_EC_amb,
                "A_AD_amb": A_AD_amb,
                "NTU_EC": NTU_EC,
                "NTU_AD": NTU_AD,
                "NTU_b": NTU_b,
                "NTU_rec": NTU_rec,
                "eps_EC": eps_EC,
                "eps_AD": eps_AD,
                "eps_b": eps_b,
                "eps_hx": eps_hx,
                
            }

            return residual, variables_dict

        
        ### RUN MODEL
        if AD_only:
            # initial guess: [m_vap_AD, T_AD_out]
            vars_guess     = [0.00    , 25.0    ]
        else:
            t_EC_out_guess = self.t_EC_in - 1.0
            # initial guess: [t_EC_out      , m_vap_AD, T_AD_out]
            vars_guess     = [t_EC_out_guess, 0.00    , 25.0]

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
    def run_desorption(self, verbose=False, turn_off_warnings=False):
        
        def system_of_equations(variables):
            
            # redefine variables for better readibility of equation system
            U_EC = self.U_EC
            A_EC = self.A_EC
            U_AD_des = self.U_AD_des
            A_AD = self.A_AD
            U_rec = self.U_rec
            A_rec = self.A_rec
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

            if not turn_off_warnings:
                # TEST IF WORKING CONDITIONS ARE REALISTIC
                ## 1. Check if epsilon values are in bounds to prevent zero division
                if eps_EC_des > 0.99999:
                    print("ERROR: eps_EC is too high. Check boundary conditions.")
                    print(f"eps_EC = {eps_EC_des}")
                    return
                if eps_AD_des > 0.99999:
                    print("ERROR: eps_AD is too high. Check boundary conditions.")
                    print(f"eps_AD = {eps_AD_des}")
                    return
                if eps_b_des > 0.99999:
                    print("ERROR: eps_b is too high. Check boundary conditions.")
                    print(f"eps_b = {eps_b_des}")
                    return
                if eps_EC_des < 0.001:
                    print("ERROR: eps_EC is too low. Check boundary conditions.")
                    print(f"eps_EC = {eps_EC_des}")
                    return
                if eps_AD_des < 0.001:
                    print("ERROR: eps_AD is too low. Check boundary conditions.")
                    print(f"eps_AD = {eps_AD_des}")
                    return
                if eps_b_des < 0.001:
                    print("ERROR: eps_b is too low. Check boundary conditions.")
                    print(f"eps_b = {eps_b_des}")
                    return
            
            

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
            cp_in     = NaOH.specific_heat_capacity(x_in_des, T_AD_from_tank) * 10**3   # [J/kgK] specific heat capacity NaOH
            cp_out    = NaOH.specific_heat_capacity(x_out, T_AD_out) * 10**3            # [J/kgK] specific heat capacity NaOH
            T_AD_in_des, T_AD_to_tank, NTU_rec_des, eps_hx_des   = recuperation(T_AD_out, T_AD_from_tank,
                                                         m_sol_in_des, m_sol_out,
                                                         cp_in, cp_out,
                                                         U_rec, A_rec)
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
            residual[0] = - m_vap_AD + m_vap_EC
            # ENERGY BALANCE A/D
            residual[1] = - h_sol_out + (Q_AD + m_sol_in_des * h_sol_in - m_vap_AD * hV) / m_sol_out
            # SIMPLIFIED HEAT CONDUCTION EQUATION FOR FILM IN AD
            residual[2] = - T_AD_out + T_AD_sat - 0.25 * d_film * U_AD_des / lambda_film * (2*T_AD_sat - t_AD_in_des - t_AD_out)

            # calculate charged energy for steady-state conditions
            t_charge = self.m_highC_tank / m_sol_in_des        # [s]
            charged_energy = Q_AD * t_charge / 3600 / 1000  # [kWh] = [Ws] / 3600 [s/h] / 1000 [W/kW]
            
            # calculate desorption energy (from solution)
            dQ_sol_des = m_sol_in_des * h_sol_in - m_sol_out * h_sol_out
            
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
                "NTU_rec_des": NTU_rec_des,
                "eps_EC_des": eps_EC_des,
                "eps_AD_des": eps_AD_des,
                "eps_b_des": eps_b_des,
                "eps_hx_des": eps_hx_des,
                "t_charge": t_charge,
                "t_charge_h": t_charge/3600,
                "charged_energy":charged_energy,
                "dQ_sol_des": dQ_sol_des
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
    
    def parameter_variation_recirculation(self, variable_name, variable_array, mode="absorption", plot=True, editable_plot=False, return_results=False):
        
        dx1_list = []
        dx2_list = []
        dx3_list = []
        dx_list = []

        xout1_list = []
        xout2_list = []
        xout3_list = []

        Q_AD1_list = []
        Q_AD2_list = []
        Q_AD3_list = []
        Q_ADavg_list = []
        
        Q_discharged1_list = []
        Q_discharged2_list = []
        Q_discharged3_list = []
        Q_discharged_list = []

        t_discharge1_h_list = []
        t_discharge2_h_list = []
        t_discharge3_h_list = []
        t_discharge_h_list = []

        m_sol_out1_list = []
        m_sol_out2_list = []
        m_sol_out3_list = []

        t_AD_out1_list = []
        t_AD_out2_list = []
        t_AD_out3_list = []
        t_AD_in_list = []

        ESD_list = []
        ESD_ideal_list = []
        ESD_pass1_list = []
        ESD_pass2_list = []
        ESD_pass3_list = []


        for value in variable_array:
            setattr(self, variable_name, value)  # Change parameter
        
            # run model three times and update solution concentration
            x_in = self.x_in
            m_highC_tank = self.m_highC_tank
            results1 = self.run_absorption(verbose=False, turn_off_warnings=True) # 1st pass
            
            self.x_in = results1["x_out"] 
            self.m_highC_tank = self.m_highC_tank * x_in / results1["x_out"]
            results2 = self.run_absorption(verbose=False, turn_off_warnings=True) # 2nd pass
            
            self.x_in = results2["x_out"]
            self.m_highC_tank = self.m_highC_tank * x_in / results2["x_out"]
            results3 = self.run_absorption(verbose=False, turn_off_warnings=True) # 3rd pass

            # prepare results
            dx1_list.append(x_in - results1["x_out"])
            dx2_list.append(results1["x_out"] - results2["x_out"])
            dx3_list.append(results2["x_out"] - results3["x_out"])
            dx_list.append(x_in - results3["x_out"])

            xout1_list.append(results1["x_out"])
            xout2_list.append(results2["x_out"])
            xout3_list.append(results3["x_out"])

            Q_AD1_list.append(-results1["Q_AD"])
            Q_AD2_list.append(-results2["Q_AD"])
            Q_AD3_list.append(-results3["Q_AD"])

            t_discharge1_h_list.append(results1["t_discharge_h"])
            t_discharge2_h_list.append(results2["t_discharge_h"])
            t_discharge3_h_list.append(results3["t_discharge_h"])
            t_discharge_h_list.append(results1["t_discharge_h"] + results2["t_discharge_h"] + results3["t_discharge_h"])
            
            Q_discharged1 = -results1["Q_AD"] * results1["t_discharge_h"] / 1000 # [kWh]
            Q_discharged2 = -results2["Q_AD"] * results2["t_discharge_h"] / 1000 # [kWh]
            Q_discharged3 = -results3["Q_AD"] * results3["t_discharge_h"] / 1000 # [kWh]
            Q_discharged1_list.append(Q_discharged1)
            Q_discharged2_list.append(Q_discharged2)
            Q_discharged3_list.append(Q_discharged3)
            Q_discharged  = Q_discharged1 + Q_discharged2 + Q_discharged3
            t_discharge_h = results1["t_discharge_h"] + results2["t_discharge_h"] + results3["t_discharge_h"]
            Q_discharged_list.append(Q_discharged)

            Q_ADavg = Q_discharged / t_discharge_h * 1000 # [kWh] / [h] * 1000 = [W]
            Q_ADavg_list.append(Q_ADavg)

            m_sol_out1_list.append(results1["m_sol_out"])
            m_sol_out2_list.append(results2["m_sol_out"])
            m_sol_out3_list.append(results3["m_sol_out"])

            t_AD_out1_list.append(results1["t_AD_out"])
            t_AD_out2_list.append(results2["t_AD_out"])
            t_AD_out3_list.append(results3["t_AD_out"])
            t_AD_in_list.append(self.t_AD_in)

            V_diluted = m_highC_tank * x_in / results3["x_out"] / rho_NaOH(results3["x_out"])  #results3["m_sol_out"] * results3["t_discharge_h"]*3600 / rho_NaOH(results3["x_out"])
            ESD       = Q_discharged / V_diluted # [kWh/m^3]
            ESD_pass1 = Q_discharged1 / (results1["m_sol_out"] * results1["t_discharge_h"]*3600 / rho_NaOH(results1["x_out"])) # [kWh/m^3]
            ESD_pass2 = Q_discharged2 / (results2["m_sol_out"] * results2["t_discharge_h"]*3600 / rho_NaOH(results2["x_out"])) # [kWh/m^3]
            ESD_pass3 = Q_discharged3 / (results3["m_sol_out"] * results3["t_discharge_h"]*3600 / rho_NaOH(results3["x_out"])) # [kWh/m^3]
            ESD_pass1_list.append(ESD_pass1)
            ESD_pass2_list.append(ESD_pass2)
            ESD_pass3_list.append(ESD_pass3)
            ESD_list.append(ESD)
            ESD_ideal = energy_density_limit(x_in, self.t_EC_in, self.t_AD_in)
            ESD_ideal_list.append(ESD_ideal)

            self.x_in         = x_in          # Reset x_in
            self.m_highC_tank = m_highC_tank  # Reset m_highC_tank

        if plot:
            # plots
            fig, ax = plt.subplots(1, 4, figsize=(8, 2), dpi=300)

            ax[0].plot(variable_array, [x_in]*len(variable_array), label="Inlet")
            ax[0].plot(variable_array, xout1_list, label="1st pass")
            ax[0].plot(variable_array, xout2_list, label="2nd pass")
            ax[0].plot(variable_array, xout3_list, label="3rd pass")
            ax[0].set_xlabel(variable_name)
            ax[0].set_ylabel("$x_{out}$")
            ax[0].set_ylim(0.2, 0.52)

            ax[1].plot([],[]) # empty because no inlet value
            ax[1].plot(variable_array, Q_AD1_list)#, label="1st pass")
            ax[1].plot(variable_array, Q_AD2_list)#, label="2nd pass")
            ax[1].plot(variable_array, Q_AD3_list)#, label="3rd pass")
            ax[1].set_xlabel(variable_name)
            ax[1].set_ylabel("$Q_{AD}$ [W]")
            ax[1].set_ylim(0, None)
            
            ax[2].plot(variable_array, t_AD_in_list)#,   label="t_AD_in")
            ax[2].plot(variable_array, t_AD_out1_list)#, label="1st pass")
            ax[2].plot(variable_array, t_AD_out2_list)#, label="2nd pass")
            ax[2].plot(variable_array, t_AD_out3_list)#, label="3rd pass")
            ax[2].set_xlabel(variable_name)
            ax[2].set_ylabel("$t_{AD_{out}}$ [°C]")
            print(results1["t_AD_out"])
            ax[2].set_ylim(self.t_AD_in-2, None)
            #ax[2].legend()

            ax[3].plot(variable_array, ESD_list,        color="black",  label="$ESD_{sim}$")
            ax[3].plot(variable_array, ESD_ideal_list,  color="grey", label="$ESD_{ideal}$")
            ax[3].plot(variable_array, np.array(ESD_pass1_list),  color="blue", label="$ESD_{1st-pass}$")
            ax[3].set_xlabel(variable_name)
            ax[3].set_ylabel("ESD [kWh/m³]")
            ax[3].set_ylim(0, 500)
            
            # figure legend below plot
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=7, fontsize=8)


            if not editable_plot:
                plt.tight_layout()
                plt.show()
        
            if editable_plot:
                return fig, ax
            
        if return_results:
            return {"dx1": dx1_list, "dx2": dx2_list, "dx3": dx3_list, "dx": dx_list,
                    "xout1": xout1_list, "xout2": xout2_list, "xout3": xout3_list,
                    "Q_AD1": Q_AD1_list, "Q_AD2": Q_AD2_list, "Q_AD3": Q_AD3_list, "Q_ADavg": Q_ADavg_list,
                    "Q_discharged1": Q_discharged1_list, "Q_discharged2": Q_discharged2_list, "Q_discharged3": Q_discharged3_list, "Q_discharged": Q_discharged_list,
                    "t_discharge1_h": t_discharge1_h_list, "t_discharge2_h": t_discharge2_h_list, "t_discharge3_h": t_discharge3_h_list, "t_discharge_h": t_discharge_h_list,
                    "m_sol_out1": m_sol_out1_list, "m_sol_out2": m_sol_out2_list, "m_sol_out3": m_sol_out3_list,
                    "t_AD_out1": t_AD_out1_list, "t_AD_out2": t_AD_out2_list, "t_AD_out3": t_AD_out3_list, "t_AD_in": t_AD_in_list,
                    "ESD": ESD_list, "ESD_ideal": ESD_ideal_list, "ESD_pass1": ESD_pass1_list, "ESD_pass2": ESD_pass2_list, "ESD_pass3": ESD_pass3_list,
                    "V_diluted": V_diluted}


    def run_recirculation(self, mode="absorption"):

        if mode == "absorption":

            # get set of original parameters
            original_params = self.all_original_params.copy()

            # first pass (normal operation)
            r1 = self.run_absorption()

            # second pass (recirculation)
            self.x_in           = r1["x_out"]
            self.m_highC_tank   = self.m_highC_tank * r1["x_in"] / r1["x_out"]
            self.T_AD_from_tank = r1["T_AD_to_tank"]
            r2                  = self.run_absorption()

            # third pass (recirculation)
            self.x_in           = r2["x_out"]
            self.m_highC_tank   = self.m_highC_tank * r2["x_in"] / r2["x_out"]
            self.T_AD_from_tank = r2["T_AD_to_tank"]
            r3                  = self.run_absorption()

            return [r1, r2, r3]
            



    # Parameter study with plotting options
    def single_parameter_variation(self, variable_name, variable_array, mode, plot_variable=None, editable_plot=False, color=None, label=None, ax=None):
        results_list = []
        params_frames = []  # Use a list to collect DataFrames

        for value in variable_array:
            setattr(self, variable_name, value)  # Change parameter

            # Run the model based on the specified mode
            if mode == "absorption":
                results = self.run_absorption(verbose=False, turn_off_warnings=True)
            elif mode == "absorption_AD_only":
                results = self.run_absorption(verbose=False, AD_only=True, turn_off_warnings=True)
            elif mode == "desorption":
                results = self.run_desorption(verbose=False, turn_off_warnings=True)
            elif mode == "desorption_AD_only":
                results = self.run_desorption(verbose=False, AD_only=True, turn_off_warnings=True)
            else:
                print("ERROR: Invalid Mode. CHOOSE mode='absorption' OR mode='desorption' OR mode='absorption_AD_only' OR mode='desorption_AD_only'")
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

        # if plot_variable:
        #     ### Plot
        #     if not editable_plot:
        #         plt.figure(figsize=(3,3))
        #         plt.rcParams.update({'font.size': 8})
        #     else:
        #         plt.ioff()
        #     if plot_variable=="Q_AD" and mode=="absorption":
        #         results_df["Q_AD"] = -results_df["Q_AD"]
        #     if color:
        #         plt.plot(variable_array, results_df[plot_variable], color=color)
        #     else:
        #         plt.plot(variable_array, results_df[plot_variable])
        #     plt.xlabel(variable_name)#, fontsize=8)
        #     plt.ylabel(plot_variable)#, fontsize=8)
        #     if not editable_plot:
        #         plt.show()

        if plot_variable:
        ### Plot
            if ax is not None:
                # Plot on the provided axis
                if plot_variable=="Q_AD" and mode=="absorption":
                    results_df["Q_AD"] = -results_df["Q_AD"]
                if color:
                    ax.plot(variable_array, results_df[plot_variable], color=color, label=label)
                else:
                    ax.plot(variable_array, results_df[plot_variable], label=label)
                ax.set_xlabel(variable_name)
                ax.set_ylabel(plot_variable)
            else:
                # Default plotting behavior
                if not editable_plot:
                    plt.figure(figsize=(3,3))
                    plt.rcParams.update({'font.size': 8})
                else:
                    plt.ioff()
                if plot_variable=="Q_AD" and mode=="absorption":
                    results_df["Q_AD"] = -results_df["Q_AD"]
                if color:
                    plt.plot(variable_array, results_df[plot_variable], color=color, label=label)
                else:
                    plt.plot(variable_array, results_df[plot_variable], label=label)
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
                                 cmap='viridis',
                                 ax=None):

        params_frames = []  # List to collect DataFrames of parameters
        results_dfs = [[None for _ in range(len(variable_array2))] for _ in range(len(variable_array1))]
        
        for i, value1 in enumerate(variable_array1):
            for j, value2 in enumerate(variable_array2):
                # Change the parameters
                setattr(self, variable_name1, value1)
                setattr(self, variable_name2, value2)

                # Run the model (assuming absorption mode for simplicity; adjust as needed)
                if mode == "absorption":
                    results = self.run_absorption(verbose=False, turn_off_warnings=True)
                elif mode == "absorption_AD_only":
                    results = self.run_absorption(verbose=False, AD_only=True, turn_off_warnings=True)
                elif mode == "desorption":
                    results = self.run_desorption(verbose=False, turn_off_warnings=True)
                elif mode == "desorption_AD_only":
                    results = self.run_desorption(verbose=False, AD_only=True, turn_off_warnings=True)
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

        # if plot_variable:

        #     ### Plot
        #     # prepare data
        #     data_matrix = np.zeros([len(variable_array1), len(variable_array2)])

        #     for i in range(len(variable_array1)):
        #         for j in range(len(variable_array2)):
        #             # Access the stored result, ensure plot_variable exists as a key, and negate its value for plotting
        #             if plot_variable=="Q_AD" and mode=="absorption":
        #                 data_matrix[i, j] = -results_dfs[i][j][plot_variable]
        #             else:
        #                 data_matrix[i, j] = results_dfs[i][j][plot_variable]

        #     if not editable_plot:
        #         plt.figure(figsize=(5,3))

        #     # Get the colormap
        #     colormap = plt.get_cmap(cmap)
        #     colors = colormap(np.linspace(0, 1, len(variable_array1)))

        #     for i, val in enumerate(variable_array1):
        #         # Correctly using variable_array2 for the x-axis, and matching y-axis data from data_matrix
        #         plt.plot(variable_array2, data_matrix[i, :], label=f"{variable_name1}={np.round(val,3)}", color=colors[i])
        #     plt.xlabel(variable_name2)
        #     plt.ylabel(plot_variable)

        #     # Adjust the legend to be outside the plot
        #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.33)
        #     #plt.tight_layout()
        #     if not editable_plot:
        #         plt.show()
        #         return params_df, results_dfs

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

            if ax is not None:
                # Plot on the provided axis
                colormap = plt.get_cmap(cmap)
                colors = colormap(np.linspace(0, 1, len(variable_array1)))

                for i, val in enumerate(variable_array1):
                    ax.plot(variable_array2, data_matrix[i, :], label=f"{variable_name1}={np.round(val,3)}", 
                            color=colors[i])
                ax.set_xlabel(variable_name2)
                ax.set_ylabel(plot_variable)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.33)

            else:
                # Default plotting behavior
                if not editable_plot:
                    plt.figure(figsize=(5,3))

                colormap = plt.get_cmap(cmap)
                colors = colormap(np.linspace(0, 1, len(variable_array1)))

                for i, val in enumerate(variable_array1):
                    plt.plot(variable_array2, data_matrix[i, :], label=f"{variable_name1}={np.round(val,3)}", 
                            color=colors[i])
                plt.xlabel(variable_name2)
                plt.ylabel(plot_variable)

                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.33)

                if not editable_plot:
                    plt.show()
            
        else:
            return params_df, results_dfs
        

#########################################################
################# ADDITIONAL FUNCTIONS ##################
#########################################################

def rho_NaOH(x):
        return (1 + x * 0.515/(0.5)) *1e3

# function calculates theoretical max energy storage density based on:
# - boundary conditions (x_in, T_EC_in, T_AD_in)
# - substance properties
def energy_density_theoretical_limit_old(x_in, T_EC_max, T_AD_min, return_all=False):
    """
    Calculates maximum energy storage density for the thermochemical storage unit.
    Parameters
    ----------
    x_in : float
        Mass fraction of NaOH in the solution entering the AD-unit.
    T_EC_max : float
        Maximum temperature [°C] of the low temperature heat source (evaporator)
        = supply temperature of the heat source.
    T_AD_min : float
        Minimum temperature [°C] of the high temperature heat sink (absorber)
        = return temperature of the heat sink.
    Returns:
        ESD_max (float): Maximum energy storage density in [kWh/m3]
    """

    # Constants
    m_sol_conc = 1000                           # [kg] mass of concentrated NaOH solution

    # possible conditions (substance prop.)
    p_max     = steamTable.psat_t(T_EC_max) *1e5          # max system pressure
    T_AD_max  = NaOH.saturation_temperature(p_max, x_in)  # max AD temperature

    # max possible dilution
    x_out_min = fsolve(lambda x: NaOH.saturation_temperature(p_max, x) - T_AD_min, 0.45)[0]
    m_sol_dil = x_in * m_sol_conc / x_out_min               # [kg] mass of diluted NaOH solution
    m_h2o     = m_sol_dil - m_sol_conc                      # [kg] mass of water absorbed by solution
    V_conc    = m_sol_conc / rho_NaOH(x_in)                 # [m3] volume of concentrated solution
    V_h2o     = m_h2o / 1000                                # [m3] volume of absorbed water
    V_dil     = V_conc + V_h2o                              # [m3] total volume of solution
    V_dil     = (m_h2o + m_sol_conc) / rho_NaOH(x_out_min)  # [m3] total volume of solution
    
    # condensation enthalpy
    h_liq     = steamTable.hL_p(p_max*1e-5)     # [kJ/kg] liquid enthalpy
    h_vap     = steamTable.hV_p(p_max*1e-5)     # [kJ/kg] vapor enthalpy
    dh_lv     = h_vap - h_liq                   # [kJ/kg] condensation enthalpy
    dH_cond   = m_h2o * h_vap                   # [kJ/kg]*[kg]=[kJ] total enthalpy from vapor

    # max possible solution enthalpy difference
    #h_sol_in    = NaOH.enthalpy(x_in, T_AD_max)                # [kJ/kg] upper limit
    h_sol_in    = NaOH.enthalpy(x_in,      T_AD_min)            # [kJ/kg] upper limit
                                                                #  WHY?    --> max. dx reached, therefore T_AD_min @ outlet.
                                                                #          --> T_AD_min @ inlet for perfect recuperator!
    h_sol_out   = NaOH.enthalpy(x_out_min, T_AD_min)            # [kJ/kg] lower limit
    dH_sol      = h_sol_in * m_sol_conc - h_sol_out * m_sol_dil # [kJ/kg] * [kg] = [kJ]

    # max possible energy storage capacity + density
    storage_cap = (dH_cond + dH_sol) / 3600     # [kJ] / [3600s/h] = [kWh]
    ESD_max     = storage_cap / V_dil           # [kWh/m3]  

    var_dict = {"ESD_max":ESD_max,
                "p_max":p_max,
                "T_AD_max":T_AD_max,
                "x_out_min":x_out_min,
                "m_sol_conc":m_sol_conc,
                "m_sol_dil":m_sol_dil,
                "m_h2o":m_h2o,
                "V_conc":V_conc,
                "V_h2o":V_h2o,
                "V_dil":V_dil,
                "dh_lv":dh_lv,
                "dH_cond":dH_cond,
                "h_sol_in":h_sol_in,
                "h_sol_out":h_sol_out,
                "dH_sol":dH_sol,
                "storage_cap":storage_cap}
    

    if return_all:
        return var_dict
    else:
        return ESD_max
    

def energy_density_limit(x_in, t_EC_in, t_AD_in, return_all=False):
    
    # set best possible boundary conditions
    p_EC        = steamTable.psat_t(t_EC_in) * 1e5 # [Pa]
    x_out       = fsolve(lambda x: (NaOH.saturation_temperature(p_EC, x)-t_AD_in), 0.49)[0]

    # calculate masses / mass flows
    rho_NaOH_dil = rho_NaOH(x_out)              # [kg] (per m3) density diluted solution
    rho_NaOH_conc= rho_NaOH(x_in)               # [kg] (per m3) density conc. solution
    m_sol_diluted = rho_NaOH_dil                # [kg] (per m3) diluted solution
    m_sol_conc  = m_sol_diluted * x_out / x_in # [kg]
    m_vapor     = m_sol_diluted - m_sol_conc   # [kg]

    # enthalpies
    h_sol_in    = NaOH.enthalpy(x_in, t_AD_in)  * 1000  # [J/kg] sol entering the AD
    h_sol_out   = NaOH.enthalpy(x_out, t_AD_in) * 1000  # [J/kg] sol leaving the AD
    h_vap       = steamTable.hV_t(t_EC_in)      * 1000  # [J/kg] vapor entering AD

    # Energy Balance
    Q_AD_J = m_vapor * h_vap + m_sol_conc * h_sol_in - m_sol_diluted * h_sol_out
    # J    = kg      * J/kg  + kg         * J/kg     - kg            * J/kg    

    Q_AD_kWh = Q_AD_J * 1/1000 * 1/3600 # [kWh] (per m3)
    # kWh    = Ws     * kWs/Ws * h/s    = [kWh]
    
    if not return_all:
        return Q_AD_kWh
    else:
        return {"Q_AD_kWh":Q_AD_kWh,
                "p_EC":p_EC,
                "x_out":x_out,
                "rho_NaOH_dil":rho_NaOH_dil,
                "rho_NaOH_conc":rho_NaOH_conc,
                "m_sol_diluted":m_sol_diluted,
                "m_sol_conc":m_sol_conc,
                "m_vapor":m_vapor,
                "h_sol_in":h_sol_in,
                "h_sol_out":h_sol_out,
                "h_vap":h_vap,
                "Q_AD_J":Q_AD_J}





# Recuperator function used by "run_absorption" and "run_desorption"
def recuperation(T_AD_out, T_AD_from_tank, m_sol_in, m_sol_out, cp_in, cp_out, U_rec, A_rec):

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
    if A_rec < 0 or U_rec < 0:
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
    NTU_rec = U_rec * A_rec / C_min
    Q_max = C_min * (T_AD_out - T_AD_from_tank)

    # Calculate effectiveness (eps) with handling for the case where 1 - C_r approaches zero
    if abs(1 - C_r) < 1e-6:
        eps_hx = NTU_rec / (1 + NTU_rec)
    else:
        eps_hx = (1 - np.exp(-NTU_rec * (1 - C_r))) / (1 - C_r * np.exp(-NTU_rec * (1 - C_r)))

    # Calculate real heat transfer
    Q_real = eps_hx * Q_max
    T_preheated = T_AD_from_tank + Q_real / (m_sol_in * cp_in)

    T_away = T_AD_out - Q_real / (m_sol_out * cp_out)

    return T_preheated, T_away, NTU_rec, eps_hx #  = T_AD_in [°C] (temperature entering AD)


def thermal_resistance_tubewall(di, do, l, lambda_pipe):
    """
    Function to calculate the thermal resistance of a pipe.
    ---
    di: float
        Inner diameter of the pipe [m]
    do: float
        Outer diameter of the pipe [m]
    lambda_pipe: float
        Thermal conductivity of the pipe material [W/mK]
    ---
    returns: float
        Thermal resistance of the pipe [K/W]
    """
    
    thermal_resistance = (do-di) / lambda_pipe

    return thermal_resistance # [W/m^2K]


# FLAG: FIX THIS CORRELATION
def htc_convective_pipeflow(m_flow, di_tube, my_fluid, cp_fluid, lambda_fluid, mode="cooled"):
    """
    Function to calculate the heat transfer coefficient for the convective flow inside a pipe.
    ---
    m_flow: float
        Mass flow rate of the fluid [kg/s]
    di_tube: float
        Inner diameter of the tube [m]
    my_fluid: float
        Dynamic viscosity of the fluid [Pa s]
    cp_fluid: float
        Specific heat capacity of the fluid [J/kgK]
    lambda_fluid: float
        Thermal conductivity of the fluid [W/mK]
    mode: str
        Mode of the heat transfer, either "cooled" or "heated"
    ---
    returns: float
        Heat transfer coefficient for the pipe flow [W/m^2K]    
    """

    # calculation
    Re_tube = m_flow / (np.pi * di_tube * my_fluid)        # [-]
    Pr_tube = my_fluid * cp_fluid / lambda_fluid           # [-]

    # LAMINAR FLOW
    Nu_tube_lam = 0.332 * Re_tube**(1/2) * Pr_tube**(1/3)
    alpha_tube_lam = Nu_tube_lam * lambda_fluid / di_tube

    # TURBULENT FLOW
    n = 0.3 if mode == "heated" else 0.4
    Nu_tube_turb = 0.023 * Re_tube ** 0.8 * Pr_tube ** 0.4
    alpha_tube_turb = Nu_tube_turb * lambda_fluid / di_tube

    # RETURN DEPENDING ON FLOW STATE
    if Re_tube < 2300:
        return alpha_tube_lam
    elif Re_tube >= 10000:
        return alpha_tube_turb
    else:
        # take weighted average
        x_Re = (Re_tube-2300) / 7700
        return alpha_tube_lam * (1-x_Re) + alpha_tube_turb * x_Re # [W/m^2K]


def htc_horizontal_falling_film(GAMMA, do_tube, my_sol, cp_sol, lambda_sol, a1, a2):
    """
    Function to calculate the heat transfer coefficient for a horizontal falling film.
    ---
    gamma: float
        Specific mass flow rate of the solution [kg/m s]
    do_tube: float
        Outer diameter of the tube [m]
    my_sol: float
        Dynamic viscosity of the solution [Pa s]
    cp_sol: float
        Specific heat capacity of the solution [J/kgK]
    lambda_sol: float
        Thermal conductivity of the solution [W/mK]
    ---
    returns: float
        Heat transfer coefficient for the falling film [W/m^2K]
    """
    # FLAG: LAMINAR FLOW, BUT CORRELATION FOR TURBULENT FLOW --> TOO LOW htc
    # FLAG: FOR ALL CORRELATIONS: IMPLEMENT FUNCTIONS
    Re_GAMMA = 4 * GAMMA / my_sol                        # [-]
    Pr = my_sol * cp_sol / lambda_sol                    # [-]
    Nu_film = a1 * Re_GAMMA ** a2                        # [-]
    alpha_EC_film = Nu_film * lambda_sol / do_tube       # [W/m^2K] --> FOR NOW ONLY FOR TURBULENT FLOW

    return alpha_EC_film #, Nu_film, Re_GAMMA, Pr

def htc_horizontal_falling_film_testing(GAMMA, do_tube, my_sol, cp_sol, lambda_sol, a1, a2, a3, a4, t_AD_in):
    """
    Function to calculate the heat transfer coefficient for a horizontal falling film.
    ---
    gamma: float
        Specific mass flow rate of the solution [kg/m s]
    my_sol: float
        Dynamic viscosity of the solution [Pa s]
    ---
    returns: float
        Heat transfer coefficient for the falling film [W/m^2K]
    """

    Re_GAMMA        = 4  * GAMMA  / my_sol           # [-]
    f_filmthickness = a1 * GAMMA              # [-] factor representing the film thickness
    alpha_film1 = 1/f_filmthickness * Re_GAMMA ** a2  # [W/m^2K] film heat transfer coefficient

    Nu_film = a3 * Re_GAMMA                          # [-]
    alpha_film2 = Nu_film * lambda_sol / do_tube       # [W/m^2K] --> FOR NOW ONLY FOR TURBULENT FLOW

    alpha_film3 = a4 * (50-t_AD_in)

    return alpha_film1*alpha_film2*alpha_film3


def mass_transfer_coefficient(GAMMA, Diff, my_sol, do_tube, rho_sol, c1, c2):
    Re_GAMMA = 4 * GAMMA / my_sol
    Sc = my_sol / (rho_sol * Diff)
    Sh = c1 * Re_GAMMA ** c2 * Sc
    beta = Sh * Diff / do_tube

    # #old
    # Sc = my_naoh / (rho_naoh * Diff_NaOH)       # Schmidt number
    # Sh = Nu_AD_film * (Sc / Pr_AD) ** n         # Sherwood number
    # beta = Sh * Diff_NaOH / do_tube_AD   *10       # mass transfer coefficient [m/s]

    return beta#, Sh, Sc
