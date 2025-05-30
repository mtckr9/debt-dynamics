# -*- coding: utf-8 -*-
"""Road1-0..

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t9yXoxrXJzrhYMO2Ary60MFAI0HV3X9g
"""

# @title Standardtext für Titel
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ------------------------ #
# Core settings           #
# ------------------------ #
kappa0, kappa_r = 0.5, 0.2
sR = 0.6 
tau0 = 0.2
sf = 0.2
theta = 0.3
x = 0.45
T_END = 12
markup_flex = 0.15
markup_rigid = 0.0
bund_yield    = 0.03
energy_price  = 0.1424
gamma1, gamma2 = 0.5, 0.5
# Semi-endogenous credit parameters
s_w = 0.02
K = 0.7
alpha = 2.0
base_CQD = 1.0

# Policy-rate schedule
def baseline_rate(t):
    if 1 <= t <= 3:   return 0.02
    if 4 <= t <= 6:   return 0.04
    if 7 <= t <= 9:   return 0.06
    if 10 <= t <= 12: return 0.08
    return 0.02

# CQD index
def CQD_C(t):
    """
    Credit-quality deterioration index:
    static average adjusted by current rate
    """
    i = baseline_rate(t)
    base_index = gamma1 * bund_yield + gamma2 * energy_price
    return base_index * (1 + alpha * i)
# Corrected semi-endogenous D_W*
def D_W_star(D_W, W, i, t):
    net_wage     = (1 - s_w) * W
    debt_service = i * D_W
    credit_tight = (1 - K) * CQD_C(t)
    return net_wage - debt_service - credit_tight


def rhs(t, y, rate_func, theta_func, kappa0_func, markup_sens):
    dF, dW = y                                   """ Corporate Leverage Ratıo and Household Leverage Ratıo"""
    i = rate_func(t)                             """Interest Rate""""
    th = theta_func(t)                            """" Speed of debt-adjustment""""
    kap0 = kappa0_func(t)                         """"Animal Spirit""""
    tau = tau0 + markup_sens * i
    pi = tau / (1 + tau)
    W = 1 - pi
    Df_star = 0.2                                    """"Target corporate debt (0.2 fixed)""""
    Dw_star = D_W_star(dW, W, i, t)                 """"Target household debt, calculated semi-endogenously (from Eq. 31) in Isaac & Kim)""""
    fg = sR * i * (dW + dF) - th * (Df_star - dF)   """"Rentier Saving""""
    household_flow = th * (Dw_star - dW)
    dF_dot = (((1 - kappa_r + kappa_r * dF) * fg - kap0 * dF)       """"Corporate Debt Dynamics""""
              - household_flow) / (1 - kappa_r)
    dW_dot = household_flow                                         """" Households Debt Dynamics""""
    return [dF_dot, dW_dot]

# Simulation function
def simulate(rate_func, theta_func=lambda t: theta,
             kappa0_func=lambda t: kappa0,
             markup_sens=1.0, y0=(0.3, 0.1), dense=201):
    sol = solve_ivp(lambda t, y: rhs(t, y, rate_func, theta_func, kappa0_func, markup_sens),
                    (0, T_END), y0, t_eval=np.linspace(0, T_END, dense))
    rows = []
    for t, dF, dW in zip(sol.t, sol.y[0], sol.y[1]):
        i = rate_func(t)
        tau = tau0 + markup_sens * i
        pi = tau / (1 + tau)
        omega = 1 - pi
        num = (kappa0 + theta * dW
               - (theta + sR * i) * dW
               + (1 - sR - kappa_r) * sf * i * dF)
        den = pi * (sR + (1 - sR - kappa_r) * sf)
        u = x * num / den
        rF = pi * u - i * dF
        gK = kappa0 + kappa_r * sf * rF
        by = 0.005 + 1.8 * i
        S = by / (by + i)
        rows.append([t, dF, dW, u, gK, omega, S])
    return pd.DataFrame(rows, columns=['t', 'dF', 'dW', 'u', 'gK', 'omega', 'S'])

# Steady-state initial conditions
i0 = baseline_rate(0)
dF0, dW0 = 0.2, D_W_star(0, 1 - (tau0/(1+tau0)), i0, 0)

""""Shocks""""
rate_IR    = lambda t: baseline_rate(t) + (0.01 if t >= 4 else 0.0)
theta_loose = lambda t: theta * 1.10 if t >= 7 else theta
kap0_boost  = lambda t: kappa0 + 0.5 if t >= 10 else kappa0

""""Run Simulation""""
baseline_df = simulate(baseline_rate, y0=(dF0, dW0))
ir_df       = simulate(rate_IR, y0=(dF0, dW0))
theta_df    = simulate(baseline_rate, theta_func=theta_loose, y0=(dF0, dW0))
kap_df      = simulate(baseline_rate, kappa0_func=kap0_boost, y0=(dF0, dW0))
flex_df     = simulate(baseline_rate, markup_sens=markup_flex, y0=(dF0, dW0))
rigid_df    = simulate(baseline_rate, markup_sens=markup_rigid, y0=(dF0, dW0))


# ------------------------ #
# 8. Plotting              #
# ------------------------ #
# 1. Corporate leverage
plt.figure(figsize=(7,4))
plt.plot(baseline_df['t'], baseline_df['dF'], label='dF baseline')
plt.plot(ir_df['t'],       ir_df['dF'],       '--', label='dF +100 bp at t=4')
plt.plot(theta_df['t'],    theta_df['dF'],    ':', label='dF θ↑10% at t=7')
plt.plot(kap_df['t'],      kap_df['dF'],      '-.', label='dF κ₀+0.5 at t=10')
plt.axvline(4, linestyle=':', lw=0.8); plt.axvline(7, linestyle=':', lw=0.8); plt.axvline(10, linestyle=':', lw=0.8)
plt.title("Corporate leverage dF"); plt.xlabel("Time (quarters)"); plt.ylabel("Leverage ratio")
plt.legend(); plt.tight_layout(); plt.show()

# 2. Household leverage
plt.figure(figsize=(7,4))
plt.plot(baseline_df['t'], baseline_df['dW'], label='dW baseline')
plt.plot(ir_df['t'],       ir_df['dW'],       '--', label='dW +100 bp at t=4')
plt.plot(theta_df['t'],    theta_df['dW'],    ':', label='dW θ↑10% at t=7')
plt.plot(kap_df['t'],      kap_df['dW'],      '-.', label='dW κ₀+0.5 at t=10')
plt.axvline(4, linestyle=':', lw=0.8); plt.axvline(7, linestyle=':', lw=0.8); plt.axvline(10, linestyle=':', lw=0.8)
plt.title("Household leverage dW"); plt.xlabel("Time (quarters)"); plt.ylabel("Leverage ratio")
plt.legend(); plt.tight_layout(); plt.show()

# 3. Capacity utilisation
plt.figure(figsize=(7,4))
plt.plot(baseline_df['t'], baseline_df['u'], label='u baseline')
plt.plot(ir_df['t'],       ir_df['u'],       '--', label='u +100 bp at t=4')
plt.plot(theta_df['t'],    theta_df['u'],    ':', label='u θ↑10% at t=7')
plt.plot(kap_df['t'],      kap_df['u'],      '-.', label='u κ₀+0.5 at t=10')
plt.axvline(4, linestyle=':', lw=0.8); plt.axvline(7, linestyle=':', lw=0.8); plt.axvline(10, linestyle=':', lw=0.8)
plt.title("Capacity utilisation u"); plt.xlabel("Time (quarters)"); plt.ylabel("u")
plt.legend(); plt.tight_layout(); plt.show()

# 4. Capital accumulation
plt.figure(figsize=(7,4))
plt.plot(baseline_df['t'], baseline_df['gK'], label='gK baseline')
plt.plot(ir_df['t'],       ir_df['gK'],       '--', label='gK +100 bp at t=4')
plt.plot(theta_df['t'],    theta_df['gK'],    ':', label='gK θ↑10% at t=7')
plt.plot(kap_df['t'],      kap_df['gK'],      '-.', label='gK κ₀+0.5 at t=10')
plt.axvline(4, linestyle=':', lw=0.8); plt.axvline(7, linestyle=':', lw=0.8); plt.axvline(10, linestyle=':', lw=0.8)
plt.title("Capital accumulation gK"); plt.xlabel("Time (quarters)"); plt.ylabel("gK")
plt.legend(); plt.tight_layout(); plt.show()

# 5. Macro-stability
plt.figure(figsize=(7,4))
plt.plot(baseline_df['t'], baseline_df['S'], label='S baseline')
plt.plot(ir_df['t'],       ir_df['S'],       '--', label='S +100 bp at t=4')
plt.plot(theta_df['t'],    theta_df['S'],    ':', label='S θ↑10% at t=7')
plt.plot(kap_df['t'],      kap_df['S'],      '-.', label='S κ₀+0.5 at t=10')
plt.axvline(4, linestyle=':', lw=0.8); plt.axvline(7, linestyle=':', lw=0.8); plt.axvline(10, linestyle=':', lw=0.8)
plt.title("Macro-stability S(t)"); plt.xlabel("Time (quarters)"); plt.ylabel("S(t)")
plt.legend(); plt.tight_layout(); plt.show()

# 6. Wage share: flexible vs. rigid
plt.figure(figsize=(7,4))
plt.plot(flex_df['t'],  flex_df['omega'],  label="ω flexible", linewidth=2)
plt.plot(rigid_df['t'], rigid_df['omega'], '--', label="ω rigid", linewidth=2)
plt.title("Real wage share under flexible vs. rigid mark-up")
plt.xlabel("Time (quarters)"); plt.ylabel("ω (wage share)")
plt.legend(); plt.tight_layout(); plt.show()
