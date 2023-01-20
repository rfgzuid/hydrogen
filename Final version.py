'''Modules'''
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

'''Simulation information'''
N = 10000 # amount of steps for Euler differential
dt = 0.1 # time step 

amount = 7 #amount of tanks
mass_tot = 40 #kg - total hydrogen mass

t = np.zeros(N) # time array for plot
K = 273.15 # Kelvin
fluid = 'H2'


'''Tank geometry'''
D_inlet = 0.003 # inlet diameter [m]
A_inlet = (1/4)*np.pi*(D_inlet)**2 # area of the inlet orifice

D_pipe = 0.1 #piping diameter from tanking station (gas slang)
A_pipe = (1/4)*np.pi*(D_pipe)**2

length_t = 2 #m - tank length of cilindrical part
radius_t = 0.175 #m - inner radius of the tank
V_t = np.pi*(radius_t**2)*length_t + (4/3)*np.pi*(radius_t**3) #tank volume (cilinder + spherical caps)


'''Tank material CARBON FIBER'''
k = 40 # Heat transfer coefficient of carbon fiber [W/(m*K)]
Rp02 = 500e6 #yield strength (carbon fiber reinforced carbon matrix composite Vf 50%)
cv_tank = 756 #J/(kg K)

w = 70e6*(radius_t/Rp02) # minimal wall thickness where the cilindrical section remains elastic (sterkteleer HC13)
s0 = 1 # safety factor
w_s = w*s0 # wall thickness with safety factor

r2 = radius_t + w_s #Outside radius measurement
A_outer = 2*np.pi*r2*length_t + 4*np.pi*(r2**2) #outer surface area
A_inner = 2*np.pi*radius_t*length_t + 4*np.pi*(radius_t**2) 

V_wall = np.pi*(r2**2)*length_t + (4/3)*np.pi*(r2**3) - V_t
rho_wall = 1700 #kg/m3
m_tank = round(V_wall*rho_wall, 2)

print("Tank capacity of:", round(V_t*1000, 1), "liters")
print("The tank on itself weighs:", m_tank, "kg\n")


'''Heat transfer through tank (see lecture 4 heat transfer)'''
R_cilinder = np.log(r2/radius_t)/(2*np.pi*k*length_t)
R_sphere = ((1/radius_t)-(1/r2))/(4*np.pi*k)
R_conduction = 1/((1/R_cilinder)+(1/R_sphere)) # parallel resistances

h_conv_out = 200 #air fan cooling
h_conv_in = 250 #hydrogen (= gas) convection (i don't know this value...)
T_cooling = 15 + K #cooled air


eta_f = 0.82 # fin efficiency - see source below
e_f = (67841.3/6784.5)*eta_f  # (A_fin / A_bare) * eta => ratio of heat transfer with and without fin
#A measurements from solidworks file

A_fins = 4*6784.5/3006279 #percentage of outer shell covered by fins

sink_factor = A_fins*e_f + (1-A_fins) # Holding brackets e.g. could help dissipating heat
'''This factor is based on the fin effectiveness and fin covered area; see the report
https://www-sciencedirect-com.tudelft.idm.oclc.org/science/article/pii/B9780128022962000032#s9000 EXAMPLE 3.12'''


R_convection = 1/(h_conv_out*A_outer*sink_factor) + 1/(h_conv_in*A_inner)

R_tot = R_conduction + R_convection # serial resistances


#Thermal radiation is negligible (tested) https://mechanicsandmachines.com/?p=682


'''Creating empty arrays for plots'''
T = np.zeros(N) #temperature in tank
P = np.zeros(N) #pressure in tank
m = np.zeros(N) #hydrogen mass in tank
m_dot = np.zeros(N) #mass flow
U = np.zeros(N) #hydrogen internal energy

'''Initial conditions of tank'''
T[0] = T_cooling # Initial tank temperature
P_start = 17e6 # 17 MPa - pure hydrogen (remains after drive) 
P[0] = P_start

U[0] = PropsSI('U', 'P', P[0], 'T', T[0], fluid)
m[0] = V_t * PropsSI('D', 'P', P[0], 'T', T[0], fluid) #m = rho*volume
m_start = m[0] #high starting mass means slow initial temperature rise



'''Heat exchanger (UA can be found probably in Mills-Coimbra)'''
T_ex = -39 + K
P_ex = 70e6 #full range of 70 MPa is needed
Q_ex = 45e3 # Racoon 45 cooling system


'''Resevoir information (constants)'''
T_resv = 15 + K # resevoir temperature (15 deg Celsius)
P_resv = (T_resv/T_ex)*P_ex
rho_resv = PropsSI('D', 'P', P_resv, 'T', T_resv, fluid) #kg/m3 - constant hydrogen supply density assumption
h_resv = PropsSI('H', 'P', P_resv, 'T', T_resv, fluid)


UA = round(Q_ex/(T_resv-T_ex), 1) # of HEX


'''Ideal gas - heat capacities (MS table A-20 at T = 80 C)'''
'''...e3 because we work with KILOgrams!'''
cp_hydrogen = 14.427e3 #cp in tank (constant for ideal gas)
cv_hydrogen = 10.302e3 #cv [J/(kg*K)] in tank
# values at T_avg = 340 K; result is almost exactly the same as iterative use of CoolProp

M_hydrogen = 2.01588e-3 #molar mass (g/mol)
R = 8.314472 #universal gas constant
R_hydrogen = R/M_hydrogen #since we will work with mass instead of moles
k = cp_hydrogen/cv_hydrogen #is 1.4


def derivatives(m, U, T, P, P_valve):

    zet = 1 + (P/70e6)*0.4 #linear

    T_valve = T_ex + (P_ex-P_valve)/(rho_resv*cp_hydrogen)  
    h_valve = h_resv + cp_hydrogen*(T_valve - T_resv) #ideal gas
    rho_valve = P_valve / ((1 + (P_valve/70e6)*0.4)*R_hydrogen*T_valve)
    
    rho_in = rho_valve * (P/P_valve)**(1/k)
    #http://www.therebreathersite.nl/01_Informative/Gasflowthroughorifice/Gasflowthroughorifice.html
    '''the density in inlet throat'''
    

    if (P/P_valve <= (2/(k+1))**(k/(k-1))): #supersonic
        w_squared = k*(P_valve/rho_valve)*(P_valve/P_resv)**(2*k) * (2/(k+1))**((k+1)/(k-1)) # see paper
    else: # Ma < 1
        w_squared = (2*k/(k-1))*(P_valve/rho_valve)*(1-(P/P_valve)**((k-1)/k)) #kick off meeting formula

    if (w_squared < 0): # needed because of the Euler steps inaccuracy - this causes m_dot to stutter though...
        w_squared = 0

      
    v_in = np.sqrt(w_squared) #through an ideal nozzle
    m_dot = v_in*rho_in*A_inlet

    '''
    T_in = T_valve - ((k-1)*v_in**2)/(2*k*R_hydrogen) # see paint - isentropic nozzle flow

    a = np.sqrt(k * R_hydrogen * T_in) #speed of sound at nozzle
    M = v_in/a # mach number
    '''

    v_valve = m_dot/(A_pipe*rho_resv)
    h_in = h_valve + 0.5*v_valve**2 - 0.5*v_in**2

    dUdt = m_dot*((v_in**2)/2 + h_in) - (T - T_cooling)/R_tot
    dTdt = dUdt/(m*cv_hydrogen + 0.5*m_tank*cv_tank)

    rho = m/V_t
    dPdt = zet*(rho*R_hydrogen*dTdt + (m_dot/V_t)*R_hydrogen*T) #chain rule for ideal gas (m/V = rho)
    
    return m_dot, dUdt, dTdt, dPdt


def P_profile (t, T, P, m): #Control valve determines inlet pressure with pressure loss

    '''massa die gebruikt kan worden tijdens rijden is 40 kg in 7 tanks'''
    if (m - m_start >= mass_tot/amount):
        P_valve = 1 #pompslang wordt uitgeschakeld (P=0 geeft error, dus ik doe het zo met P=1)
    elif (t <= 10):
        P_valve = P_start + (13e6/10**2)*t**2
    else:
        P_valve = min(31e6 + (27e6/600**1.1)*(t-10)**1.1, 70e6)

    return P_valve
    


'''Euler differential loop using the kick-off meeting'''
for i in range(N-1): #indices from 0
    t[i+1] = t[i] + dt

    P_valve = P_profile(t[i], T[i], P[i], m[i])

    [dmdt, dUdt, dTdt, dPdt] = derivatives(m[i], U[i], T[i], P[i], P_valve)

    m_dot[i] = dmdt
    m[i+1] = m[i] + dmdt*dt
    U[i+1] = U[i] + dUdt*dt
    T[i+1] = T[i] + dTdt*dt
    P[i+1] = P[i] + dPdt*dt



hyd_price = 11 #euro per kg
t_tanking = np.argmax(m)*dt # index * timestep per index
print('Time to fill to desired hydrogen mass:', round(t_tanking, 1), 'seconds. This is', round(t_tanking/60, 1), 'minutes')

print('Maximum temperature:', round(max(T-K), 1), 'Celsius')
print('Maximum pressure while filling:', round(max(P)/1e6, 2), 'MPa')

print('\nAfter the tank has cooled down completely, a tank pressure of', 55, 'MPa remains')
# checked with asymptote


E_ex = amount*Q_ex*t_tanking # single use for 7 tanks combined

total_price = round(hyd_price*mass_tot + (E_ex/3600000)*0.2525, 2) # 0.2525 euro per kWh energiekosten [https://www.anwb.nl/huis/energie/wat-kost-1-kwh]
print('\nOPEX van 1 keer tanken (inclusief waterstofprijs):', round(total_price, 2), 'euro') 


capex = 20*m_tank*1.03
print('\nCapex costs:', round(amount*capex, 2), 'euro')


#3 seperate plots, used in Overleaf
fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=150)
ax[0].plot(t, m)
ax[1].plot(t[10:], m_dot[10:])

ax[0].set_xlabel('time/s')
ax[0].set_ylabel('hydrogen mass/kg')
ax[1].set_xlabel('time/s')
ax[1].set_ylabel('mass flow (kg/s)')

ax[0].set_title('Hydrogen mass in tank while filling')
ax[1].set_title('Mass flow variation while filling')

plt.xscale('log')

fig.tight_layout(pad=1.2)
plt.show()



fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
ax.plot(t, T-K)

ax.set_xlabel('time/s')
ax.set_ylabel('Temperature/Celsius')

ax.set_title('Average temperature inside tank')
fig.tight_layout(pad=1.0)
plt.show()



fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
ax.plot(t, P/1e6)

ax.set_xlabel('time/s')
ax.set_ylabel('Pressure/MPa')

ax.set_title('Pressure inside of the tank')
fig.tight_layout(pad=1.0)
plt.show()
