import json, numpy as np
R = json.load(open('results/results.json'))
L = {'euc': r'$\mathbb{R}^3$', 'nil': r'$\Nil$', 'sol': r'$\Sol$', 'sl2': r'$\SL$'}
DN = {'karate': 'Karate club', 'lesmis': 'Les Mis\\\'erables', 'florentine': 'Florentine families'}
out = []
def mac(name, val):
    out.append(f'\\newcommand{{\\{name}}}{{{val}}}')
def f(x, k=3):
    return f'{x:.{k}f}'

# ---- simulation table ----
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R[f'sim_{g}']
    rows.append(' & '.join([L[g], f(v['density'], 2), f(v['oracle_auc'], 2), f'{v["alpha_true"]:.1f}',
                            f"{f(v['mcmc_alpha'],2)} ({f(v['mcmc_alpha_sd'],2)})", f"{f(v['vi_alpha'],2)} ({f(v['vi_alpha_sd'],2)})",
                            f(v['mcmc_auc'], 2), f(v['vi_auc'], 2), f(v['mcmc_logloss_rep'], 3), f(v['vi_logloss_rep'], 3), f(v['oracle_logloss_rep'], 3),
                            f(v['mcmc_Dcorr'], 2), f(v['vi_Dcorr'], 2), f(v['mcmc_vi_Dcorr'], 2),
                            f'{v["mcmc_time"]:.0f}', f'{v["vi_time"]:.0f}']) + r' \\')
tab = r'''\begin{table}[H]\centering\scriptsize
\caption{Simulation study ($N=40$, one network per geometry, true positions $\LWN(\eb,\diag(\sigma_h^2,\sigma_h^2,\sigma_v^2))$ with $(\sigma_h,\sigma_v)=$ \simsig, $\alpha=2.5$). Columns: edge density; AUC of the true probabilities; $\alpha$ (true, posterior mean (s.d.) from MCMC, variational mean (s.d.)); in-sample AUC; log-loss of the posterior predictive on an independent replicate network from the same latent positions (oracle: true probabilities); correlation between the true distance matrix and the posterior mean distance matrix $\mathbb{E}[d_{ij}\mid\mathcal{Y}]$ (MCMC), the mean distance under $q$ (BBVI), and between the two; wall-clock seconds (MCMC: \simiters\ iterations; BBVI: 2000 iterations, $S=20$).}\label{tab:sims}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccccccccc cc}\toprule
 & & & \multicolumn{3}{c}{$\alpha$} & \multicolumn{2}{c}{AUC} & \multicolumn{3}{c}{replicate log-loss} & \multicolumn{3}{c}{distance corr.} & \multicolumn{2}{c}{time (s)}\\
\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(lr){15-16}
$\M$ & dens. & AUC$_0$ & true & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & oracle & MCMC & BBVI & MCMC--BBVI & MCMC & BBVI\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}'''
open('paper/tab_sims.tex', 'w').write(tab)
v = R['sim_nil']
mac('simsig', f"$({R['sim_nil']['sig'][0]},{R['sim_nil']['sig'][1]})$ for $\\Nil$, $({R['sim_sol']['sig'][0]},{R['sim_sol']['sig'][1]})$ for $\\Sol$, $({R['sim_sl2']['sig'][0]},{R['sim_sl2']['sig'][1]})$ for $\\SL$")
mac('simiters', f"{R['sim_nil']['mcmc_iters']}")
mac('simburn', f"{R['sim_nil']['mcmc_burn']}")

# ---- cross-geometry table ----
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R[f'sim_{g}']['cross']
    rows.append(' & '.join([L[g]] + [f"{f(v[h]['logloss_rep'],3)} / {f(v[h]['auc_rep'],2)}" for h in ('euc', 'nil', 'sol', 'sl2')]) + r' \\')
tab = r'''\begin{table}[H]\centering\small
\caption{Geometry recovery in the simulations: each simulated network (rows: true geometry) is fitted by BBVI (1500 iterations) under each of the four geometries (columns); entries are log-loss / AUC of the variational predictive on the replicate network. The replicate log-loss of the true probabilities is \oracleNil\ ($\Nil$), \oracleSol\ ($\Sol$), \oracleSL\ ($\SL$).}\label{tab:cross}
\begin{tabular}{lcccc}\toprule
true $\backslash$ fitted & $\mathbb{R}^3$ & $\Nil$ & $\Sol$ & $\SL$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}'''
open('paper/tab_cross.tex', 'w').write(tab)
mac('oracleNil', f(R['sim_nil']['oracle_logloss_rep'], 3)); mac('oracleSol', f(R['sim_sol']['oracle_logloss_rep'], 3)); mac('oracleSL', f(R['sim_sl2']['oracle_logloss_rep'], 3))

# ---- real data comparison ----
rows = []
for d in ('karate', 'lesmis', 'florentine'):
    v0 = R[f'cmp_{d}_euc']
    cells = []
    for g in ('euc', 'nil', 'sol', 'sl2'):
        v = R[f'cmp_{d}_{g}']
        cells.append(f"{f(v['out_logloss'],3)} / {f(v['out_auc'],2)}")
    rows.append(' & '.join([DN[d], str(v0['N']), f(v0['density'], 3), f"{int(100*v0.get('holdout_frac',0.1))}\\%"] + cells) + r' \\')
tab = r'''\begin{table}[H]\centering\small
\caption{Real networks: held-out log-loss / AUC of the variational predictive for each geometry (BBVI, 1500 iterations, $S=20$), averaged over two random splits in which the stated fraction of dyads is hidden (AUC pooled over the two splits). Lower log-loss and higher AUC are better.}\label{tab:compare}
\begin{tabular}{lcccccc c}\toprule
network & $N$ & density & held out & $\mathbb{R}^3$ & $\Nil$ & $\Sol$ & $\SL$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}'''
open('paper/tab_compare.tex', 'w').write(tab)
mac('karateN', str(R['cmp_karate_euc']['N'])); mac('lesmisN', str(R['cmp_lesmis_euc']['N'])); mac('florN', str(R['cmp_florentine_euc']['N']))
mac('florHold', str(R['cmp_florentine_euc']['n_holdout']))

# ---- showcase table ----
rows = []
for d, g in (('karate', 'sl2'), ('lesmis', 'nil'), ('florentine', 'sol')):
    v = R[f'show_{d}_{g}']
    rows.append(' & '.join([DN[d], L[g], str(v['N']), f"{f(v['mcmc_alpha'],2)} ({f(v['mcmc_alpha_sd'],2)})", f"{f(v['vi_alpha'],2)} ({f(v['vi_alpha_sd'],2)})",
                            f(v['mcmc_auc'], 3), f(v['vi_auc'], 3), f(v['mcmc_logloss'], 3), f(v['vi_logloss'], 3),
                            f'{v["ll_mcmc"]:.1f}', f'{v["ll_vi"]:.1f}', f(v['mcmc_vi_Dcorr'], 2), f'{v["mcmc_time"]:.0f}', f'{v["vi_time"]:.0f}']) + r' \\')
tab = r'''\begin{table}[H]\centering\scriptsize
\caption{Real networks: MCMC (\showiters\ iterations, first \showburn\ discarded, thinned by 10) versus BBVI (2000 iterations, $S=20$). $\alpha$: posterior mean (s.d.) and variational mean (s.d.); in-sample AUC and log-loss of the posterior predictive; average log-likelihood over the retained draws (MCMC) and over the last 100 iterations (BBVI); correlation between the posterior mean distance matrix and the mean distance matrix under $q$; wall-clock seconds.}\label{tab:show}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccccccccccc}\toprule
 & & & \multicolumn{2}{c}{$\alpha$} & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{log-loss} & \multicolumn{2}{c}{log-lik.} & dist. & \multicolumn{2}{c}{time (s)}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){13-14}
network & $\M$ & $N$ & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & corr. & MCMC & BBVI\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}'''
open('paper/tab_show.tex', 'w').write(tab)
mac('showiters', str(R['show_karate_sl2']['mcmc_iters'])); mac('showburn', str(R['show_karate_sl2']['mcmc_burn']))
for d, g in (('karate', 'sl2'), ('lesmis', 'nil'), ('florentine', 'sol')):
    v = R[f'show_{d}_{g}']
    mac(f'{d}Anchors', ', '.join(str(a + 1) for a in v['anchors']))
    mac(f'{d}AccZ', f"{v['mcmc_acc']['z']:.2f}"); mac(f'{d}AccA', f"{v['mcmc_acc']['alpha']:.2f}")
    mac(f'{d}Dcorr', f(v['mcmc_vi_Dcorr'], 2)); mac(f'{d}Sig', f"({f(v['mcmc_sig'][0],2)},{f(v['mcmc_sig'][1],2)})")
    mac(f'{d}MCMCalpha', f(v['mcmc_alpha'], 2)); mac(f'{d}VIalpha', f(v['vi_alpha'], 2))
    mac(f'{d}MCMCauc', f(v['mcmc_auc'], 3)); mac(f'{d}VIauc', f(v['vi_auc'], 3))

# ---- rigidity table ----
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R['rigidity'][g]
    for rep in range(2):
        s = np.array(v['singular_values'][rep])
        k = v['expected_rank']
        rows.append(' & '.join([L[g] if rep == 0 else '', str(rep + 1), str(k), f'{s[0]:.2f}', f'{s[k-2]:.3f}', f'{s[k-1]:.3f}', f'{s[k]:.1e}', f'{s[-1]:.1e}', f'{v["h"]:g}']) + r' \\')
tab = r'''\begin{table}[H]\centering\small
\caption{Numerical rank of the differential of the distance map at random configurations with $N=8$ (Theorem~\ref{thm:generic}). $k=3N-\dim\Isom(\M)$ is the expected rank; $s_1\ge s_2\ge\cdots$ are the singular values of the $28\times24$ Jacobian computed by central differences with step $h$ (exact distance for $\Nil$; tables for $\Sol$ and $\SL$, whose interpolation error of order $10^{-3}$ yields a noise level of order $10^{-3}/h$ in the Jacobian entries).}\label{tab:rigidity}
\begin{tabular}{lcccccccc}\toprule
$\M$ & config. & $k$ & $s_1$ & $s_{k-1}$ & $s_k$ & $s_{k+1}$ & $s_{24}$ & $h$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}'''
open('paper/tab_rigidity.tex', 'w').write(tab)
open('paper/numbers.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
