# ==============================================================
# Dual‑Debt SFC Model (Isaac & Kim core, Germany)
#   • Keen endogenous markup τ(t)  (speed κ_τ, target u_norm)
#   • Buffer‑stock DSR rule for households (flat dW when DSR in band)
#   • Six macro series: dF, dW, u, gK, omega, S  
#   • Scenarios: baseline, +100 bp @4, θ+10 % @7, κ₀+0.2 @10
#   • Flex (Keen) vs Rigid (τ frozen) markup comparison
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------ #
# 1. PARAMETERS            #
# ------------------------ #
# Keen price‑adjustment (annual ⇒ ÷4 to quarterly inside RHS)
kappa_tau   = 0.6           # speed of τ reaction
lambda_tau  = 0.02          # damping toward τ0
u_norm      = 0.80           # normal utilisation

# Core structural params (DE)
kappa0, kappa_r = 0.02, 0.65
sR, sf          = 0.55, 0.20
tau0            = 0.20
theta, x        = 0.05, 0.45
T_END           = 24               # quarters (3 years)

bund_yield   = 0.03
energy_price = 0.1424
gamma1, gamma2 = 0.5, 0.3

# Household‑credit
s_w   = 0.08
K     = 0.20
alpha = 0.50

#counter-cyclle
zeta = 0.1
i0 = 0.02

# Buffer‑stock DSR band (DE average 5–7.2 %)
DSR_low, DSR_high = 0.05, 0.07

# Mark‑up regimes (flex vs rigid)
kappa_tau_flex = kappa_tau
kappa_tau_rigid = 0.1

# Wage-growth reaction:
phi = 0.02

# ------------------------ #
# 2. POLICY RATE PATHS     #
# ------------------------ #

def baseline_rate(t):
    if 1 <= t <= 3:   return 0.02
    if 4 <= t <= 6:   return 0.03
    if 7 <= t <= 9:   return 0.04
    if 10 <= t <= 12: return 0.05
    if 13 <= t <= 15: return 0.06
    if 16 <= t <= 18: return 0.07
    if 19 <= t <= 21: return 0.08
    if 22 <= t <= 23: return 0.09
    return 0.02


def keen_line(t):

    if 1 <= t <= 3:   return 0.03  # +1%
    if 4 <= t <= 6:   return 0.05  # +1%
    if 7 <= t <= 9:   return 0.07
    if 10 <= t <= 12: return 0.09
    if 13 <= t <= 15: return 0.11
    if 16 <= t <= 18: return 0.13
    if 19 <= t <= 21: return 0.15
    if 22 <= t <= 23: return 0.17
    return 0.03

keen_rate = lambda t: keen_line(t) + (0.02 if t >=4 else 0.0)
rate_IR  = lambda t: baseline_rate(t) + (0.01 if t >= 4 else 0.0) #shock hits at t = 4


# ------------------------ #
# 3. SUPPORT FUNCTIONS     #
# ------------------------ #

def CQD(i):
    return (gamma1*bund_yield + gamma2*energy_price) * (1 + alpha*i)

def D_W_star(dW, W, i):
    return (1-s_w)*W - i*dW - (1-K)*CQD(i)

def util(dF, dW, i, tau):
    pi = tau/(1+tau)
    W  = 1 - pi
    num = (kappa0 + theta*dW - (theta+sR*i)*dW + (1-sR-kappa_r)*sf*i*dF)
    den = pi * (sR + (1-sR-kappa_r)*sf)
    return x * num / den

# ------------------------ #
# 4. RHS GENERATOR         #
# ------------------------ #

def make_rhs(kappa_tau_q,zeta_pass):
    def RHS(t, y, rate_f, θ_f, κ0_f):
        dF, dW, tau,w = y
        i   = rate_f(t)
        θ   = θ_f(t)
        κ0  = κ0_f(t)

        # shares
        pi = tau/(1+tau)
        W  = 1 - pi

        # targets
        Df_star = 0.8
        Dw_star = D_W_star(dW, W, i)

        # flows
        fg  = sR*i*(dW+dF) - θ*(Df_star-dF)
        hhf = θ*(Dw_star-dW)

        dF_dot = (((1-kappa_r+kappa_r*dF)*fg - κ0*dF) - hhf) / (1-kappa_r)

        DSR = i*dW/W
        if DSR_low <= DSR <= DSR_high:
            dW_dot = 0.0
        else:
            num_dW = ((1-kappa_r-kappa_r*dW)*hhf + kappa_r*sR*i*dW*(dW+dF) - κ0*dW)
            dW_dot = num_dW / (1-kappa_r)

        u   = util(dF, dW, i, tau)              # utilisation as before
        tau_dot = (          # Keen markup evolution
          -(kappa_tau_q)*(u - u_norm)   # counter-cyclical demand term
          + zeta_pass*(i - i0)               # cost pass-through term
        - (lambda_tau/4)*(tau - tau0) # damping toward τ0
)

        #κτ_q*(u-u_norm) - (lambda_tau/4)*(tau-tau0)
        phi_q = phi/4
        w_dot = phi_q* (u- u_norm)
        return [dF_dot, dW_dot, tau_dot, w_dot]
    return RHS

rhs_flex  = make_rhs(kappa_tau_flex / 4, zeta)   # Keen active
rhs_rigid = make_rhs(kappa_tau_rigid / 4, 0.0)  # τ frozen

# ------------------------ #
# 5. SIMULATION WRAPPER    #
# ------------------------ #

def simulate(rate_f = baseline_rate,
             rhs      = rhs_flex,
             θ_f      = lambda t: theta,
             κ0_f     = lambda t: kappa0,
             y0       = (0.30, 0.55, tau0, 1.0),
             dense    = 201):

    sol = solve_ivp(lambda t,y: rhs(t,y,rate_f,θ_f,κ0_f),
                    (0, T_END), y0,
                    t_eval=np.linspace(0, T_END, dense), rtol=1e-8)

    cols=[]
    for t_q,dF,dW,tau,w in zip(sol.t,*sol.y):


        i   = rate_f(t_q)
        omega = w / (1+tau)
        pi  = tau/(1+tau); W = 1-pi
        u   = util(dF,dW,i,tau)
        rF  = pi*u - i*dF
        gK  = kappa0 + kappa_r*sf*rF
        #Targets
        Df_star = 0.8
        Dw_star = D_W_star(dW, W, i)
        #Ineqality-penalized macro stability index
        S_weight = 0.1
        gap_penalty = np.exp(-S_weight * (abs(Df_star - dF) + abs(Dw_star - dW)))
        S   = ((0.005+1.8*i)/(0.005+1.8*i + i)) * gap_penalty
        cols.append([t_q,dF,dW,u,gK,W,S,omega])
    return pd.DataFrame(cols, columns=['t','dF','dW','u','gK','W','S','omega'])

# ------------------------ #
# 6. RUN SCENARIOS         #
# ------------------------ #

baseline_df = simulate()
ir_df       = simulate(rate_IR)

θ_loose = lambda t: theta*1.10 if t>=7 else theta
κ0_boost = lambda t: kappa0+0.05 if t>=10 else kappa0

theta_df = simulate(baseline_rate, rhs_flex, θ_f=θ_loose)
kap_df   = simulate(baseline_rate, rhs_flex, κ0_f=κ0_boost)

flex_df  = baseline_df
rigid_df = simulate(baseline_rate, rhs_rigid)

flex_shock_df  = simulate(keen_rate, rhs_flex)
flex_noshock_df = simulate(keen_line, rhs_flex)


# ------------------------ #
# 7. PLOTTING              #
# ------------------------ #

def vline():
    for q in (4,7,10,13,16,19,21,24): plt.axvline(q, ls=':', lw=0.8)

plots = [
    ('Corporate leverage dF', 'dF', baseline_df, ir_df, theta_df, kap_df),
    ('Household leverage dW', 'dW', baseline_df, ir_df, theta_df, kap_df),
    ('Capacity utilisation u', 'u',  baseline_df, ir_df, theta_df, kap_df),
    ('Capital accumulation gK','gK', baseline_df, ir_df, theta_df, kap_df),
    ('Macro‑stability S(t)',   'S',  baseline_df, ir_df, theta_df, kap_df)
]

for title, col, base, shock, th, kp in plots:
    plt.figure(figsize=(7,4))
    plt.plot(base.t,  base[col], label='baseline')
    plt.plot(shock.t, shock[col],'--', label='+100 bp @4')
    plt.plot(th.t,    th[col],   ':', label='θ +10 % @7')
    plt.plot(kp.t,    kp[col],  '-.', label='κ₀ +0.2 @10')
    vline(); plt.title(title); plt.xlabel('quarters'); plt.ylabel(col); plt.legend(); plt.tight_layout()

# Wage share flex vs rigid
plt.figure(figsize=(8,5))
plt.plot(flex_shock_df.t, flex_shock_df.u, color='black', lw=2, label='Capacity utilisation u(t)')
plt.plot(flex_noshock_df.t, flex_noshock_df.omega, color='blue', lw=2, label='Real wage share ω (flex markup, no policy shock)')
plt.plot(flex_shock_df.t, flex_shock_df.omega, color='orange', lw=2, linestyle='--', label='Real wage share ω (flex markup, +100 bp shock)')
vline()
plt.title('Wage Share Response under Flexible Markup: Policy Shock vs Baseline')
plt.axvspan(4, 6, color='gray', alpha=0.2, label='Policy shock starts')
plt.xlabel('Quarters')
plt.ylabel('ω and u')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

