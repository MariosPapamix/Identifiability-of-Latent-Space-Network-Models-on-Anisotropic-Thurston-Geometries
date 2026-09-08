import json, numpy as np
R = json.load(open('results/results.json')); R2 = json.load(open('results2/results.json'))
L = {'euc': r'$\mathbb{R}^3$', 'h3': r'$\mathbb{H}^3$', 's3': r'$\mathbb{S}^3$', 'h2r': r'$\mathbb{H}^2\!\times\!\mathbb{R}$', 's2r': r'$\mathbb{S}^2\!\times\!\mathbb{R}$',
     'nil': r'$\Nil$', 'sol': r'$\Sol$', 'sl2': r'$\SL$', 'ase': 'ASE', 'dcsbm': 'DC-SBM', 'cl': 'Chung--Lu'}
DN = {'karate': 'Karate club', 'lesmis': "Les Mis\\'erables", 'cora': 'Cora ball', 'pancreas': 'Pancreas kNN'}
out = []
def mac(n, v): out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
def f(x, k=3): return f'{x:.{k}f}'

# ---------- simulations ----------
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R[f'sim_{g}']
    rows.append(' & '.join([L[g], f(v['density'], 2), f(v['oracle_auc'], 2), f'{v["alpha_true"]:.1f}', f"{f(v['mcmc_alpha'],2)} ({f(v['mcmc_alpha_sd'],2)})", f"{f(v['vi_alpha'],2)} ({f(v['vi_alpha_sd'],2)})",
                            f"({f(v['mcmc_sig'][0],1)},{f(v['mcmc_sig'][1],1)})", f"({f(v['vi_sig'][0],1)},{f(v['vi_sig'][1],1)})", f(v['mcmc_auc'], 2), f(v['vi_auc'], 2),
                            f(v['mcmc_logloss_rep'], 3), f(v['vi_logloss_rep'], 3), f(v['oracle_logloss_rep'], 3), f(v['mcmc_Dcorr'], 2), f(v['vi_Dcorr'], 2), f(v['mcmc_vi_Dcorr'], 2),
                            f'{v["mcmc_time"]:.0f}', f'{v["vi_time"]:.0f}']) + r' \\')
open('paper/tab_sims.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Simulation study ($N=40$, one network per geometry; true positions $\LWN(\eb,\diag(\sigma_h^2,\sigma_h^2,\sigma_v^2))$ with $(\sigma_h,\sigma_v)=$ \simsig, $\alpha=2.5$; scales estimated under the inverse-Gamma prior in both schemes). Columns: edge density; AUC of the true probabilities; $\alpha$ (true; posterior mean (s.d.); variational mean (s.d.)); posterior mean / variational estimate of $(\sigma_h,\sigma_v)$; in-sample AUC; replicate log-loss (oracle: true probabilities); correlation of the posterior mean distance matrix with the truth (MCMC, BBVI) and between the two schemes; wall-clock seconds (MCMC \simiters\ iterations, BBVI 2000 iterations with $S=20$).}\label{tab:sims}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccccccccccccc}\toprule
 & & & \multicolumn{3}{c}{$\alpha$} & \multicolumn{2}{c}{$(\hat\sigma_h,\hat\sigma_v)$} & \multicolumn{2}{c}{AUC} & \multicolumn{3}{c}{replicate log-loss} & \multicolumn{3}{c}{distance corr.} & \multicolumn{2}{c}{time (s)}\\
\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}\cmidrule(lr){17-18}
$\M$ & dens. & AUC$_0$ & true & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & oracle & MCMC & BBVI & MCMC--BBVI & MCMC & BBVI\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
mac('simsig', f"$({R['sim_nil']['sig'][0]},{R['sim_nil']['sig'][1]})$ for $\\Nil$, $({R['sim_sol']['sig'][0]},{R['sim_sol']['sig'][1]})$ for $\\Sol$, $({R['sim_sl2']['sig'][0]},{R['sim_sl2']['sig'][1]})$ for $\\SL$")
mac('simiters', str(R['sim_nil']['mcmc_iters']))

# ---------- identifiability: inverse problem ----------
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R2[f'inv_{g}']
    cells = []
    for N in ('3', '4', '5', '6', '7', '8', '10', '14'):
        x = v[N]; med = x['local_err_median']
        cells.append((f'{med:.2f}' if med is not None and med > 1e-6 else ('$<10^{-6}$' if med is not None else '--')) + f" / {x['global_distinct']}")
    rows.append(' & '.join([L[g]] + cells) + r' \\')
open('paper/tab_inverse.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Noise-free inverse problem: for $12$ random configurations per $N$ the distance matrix is computed and the gauge-fixed positions are re-estimated by least squares. Entries: median recovery error (root mean square over coordinates, after gauge fixing) from a start perturbed by $\mathcal{N}(0,0.3^2)$ noise / number of random starts (out of 12) that reached an exact solution ($\text{residual}<10^{-4}$, $2\times10^{-3}$ for the tabulated distances) that is not congruent to the truth (error $>0.05$). Exact distances for $\Nil$, tables for $\Sol$ and $\SL$.}\label{tab:inverse}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}\toprule
$\M\ \backslash\ N$ & 3 & 4 & 5 & 6 & 7 & 8 & 10 & 14\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
rows = []
for g in ('nil', 'sol', 'sl2'):
    v = R2[f'multi_{g}']; rows.append(' & '.join([L[g]] + [f"{v[k]['mean_solutions']:.1f} ({int(100*v[k]['frac_multiple'])}\\%)" for k in ('3', '4', '5')]) + r' \\')
open('paper/tab_multi.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Number of positions of a further node compatible with its distances to $k$ fixed nodes (mean over 30 random configurations; in parentheses the percentage of configurations with more than one solution), found by 40 random restarts of a least-squares solver.}\label{tab:multi}
\begin{tabular}{lccc}\toprule
$\M\ \backslash\ k$ & 3 & 4 & 5\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
j = R2['jacobian']
for N in ('8', '40'):
    for lab in ('with', 'without'):
        mac(f'jac{ {"8":"viii","40":"xl"}[N] }{lab}', f"{j[N][lab]['r_mean']:.2f} ({j[N][lab]['r_sd']:.2f})")
# ---------- coverage ----------
rows = []
for g in ('nil', 'sol', 'sl2'):
    for lab, key in (('fixed', f'coverage_{g}'), ('inverse-Gamma', f'coverage_{g}_ig')):
        if key not in R2:
            continue
        v = R2[key]
        first = L[g] if (lab == 'fixed' or f'coverage_{g}' not in R2) else ''
        rows.append(' & '.join([first, lab, f"{100*v['cov90_dist']:.0f}\\%", f"{100*v['cov50_dist']:.0f}\\%", f"{100*v['cov90_prob']:.0f}\\%", f(v['width90'], 2),
                                'yes' if v['alpha_cov'] else 'no', (f"({f(v['sig_mean'][0],2)},{f(v['sig_mean'][1],2)})" if 'sig_mean' in v else '--'), f"({v['sig_true'][0]},{v['sig_true'][1]})" if 'sig_true' in v else '--']) + r' \\')
open('paper/tab_coverage.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Frequentist coverage of MCMC credible intervals in the simulations ($N=40$, 10\,000 iterations): fraction of the $780$ true pairwise distances inside the $90\%$ and $50\%$ posterior intervals, fraction of true tie probabilities inside the $90\%$ intervals, mean width of the $90\%$ distance intervals, whether the true $\alpha$ is inside its $90\%$ interval, and the posterior mean of the scales against their true values, for the scales held fixed at their initial values and for the scales sampled under the $\mathrm{InvGamma}(2,2)$ prior.}\label{tab:coverage}
\resizebox{\textwidth}{!}{\begin{tabular}{llccccccc}\toprule
$\M$ & scales & 90\% dist. & 50\% dist. & 90\% prob. & width & $\alpha$ covered & $(\hat\sigma_h,\hat\sigma_v)$ & true\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
# ---------- twist diagnostics ----------
h = R2['holonomy']
mac('holNil', f"{h['nil']['fitted_share_mean']:.2f} ({h['nil']['fitted_share_sd']:.2f})"); mac('holNilTrue', f"{h['nil']['true_share']:.2f}")
mac('holEuc', f"{h['euc']['fitted_share_mean']:.2f} ({h['euc']['fitted_share_sd']:.2f})")
tp = R2['tau_profile']
rows = []
for truth in ('1.0', '0.0'):
    rows.append(' & '.join([('$\\SL$ ($\\tau=1$)' if truth == '1.0' else '$\\mathbb{H}^2\\times\\mathbb{R}$ ($\\tau=0$)')] + [f"{tp[truth][t]['elbo']:.1f} / {tp[truth][t]['logloss_rep']:.3f}" for t in ('0.0', '0.5', '1.0', '2.0')]) + r' \\')
open('paper/tab_tau.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Profile over the twist of the family $ds^2_{\Hb^2}+(d\zeta-\tau A_{-1})^2$: ELBO / replicate log-loss of BBVI fits with $\tau\in\{0,0.5,1,2\}$ to a network of $N=60$ nodes generated with $\tau=1$ ($\SL$) and with $\tau=0$ ($\Hb^2\times\R$).}\label{tab:tau}
\begin{tabular}{lcccc}\toprule
data $\backslash$ fitted $\tau$ & $0$ & $0.5$ & $1$ & $2$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
# ---------- win maps ----------
rows = []
for truth, fits in (('nil', ('nil', 'euc', 'h3', 'h2r')), ('sol', ('sol', 'euc', 'h3', 'h2r')), ('sl2', ('sl2', 'h3', 'h2r', 'nil'))):
    keys = sorted([k for k in R2 if k.startswith(f'win_{truth}_')], key=lambda k: (int(k.split('_N')[1].split('_')[0]), float(k.split('_')[-1])))
    for k in keys:
        v = R2[k]; N = k.split('_N')[1].split('_')[0]; val = k.split('_')[-1]
        best = min(v['fits'][g]['logloss_rep'] for g in fits)
        cells = [(r'\textbf{' + f(v['fits'][g]['logloss_rep'], 3) + '}' if v['fits'][g]['logloss_rep'] == best else f(v['fits'][g]['logloss_rep'], 3)) for g in fits]
        rows.append(' & '.join([L[truth], N, val, f(v['density'], 2), f(v['oracle'], 3)] + [L[g] + ': ' + c for g, c in zip(fits, cells)]) + r' \\')
open('paper/tab_winmap.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Win maps: replicate log-loss of BBVI fits (800 iterations, $S=10$) under the true geometry and under competing geometries, as the anisotropy of the truth grows. Knob: horizontal spread $\sigma_h$ for $\Nil$ (vertical spread $1.5$) and $\SL$ (vertical spread $2$), vertical spread $\sigma_v$ for $\Sol$ (horizontal spread $1.5$); $\alpha$ chosen for an expected density of $0.15$. The best fit in each row is in bold; ``oracle'' is the log-loss of the true probabilities.}\label{tab:winmap}
\begin{tabular}{llcccllll}\toprule
truth & $N$ & knob & dens. & oracle & \multicolumn{4}{c}{replicate log-loss by fitted geometry}\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
# ---------- competitor table with paired CIs ----------
methods = ['euc', 'h3', 's3', 'h2r', 's2r', 'nil', 'sol', 'sl2', 'ase', 'dcsbm', 'cl']
rows = []
for d in ('karate', 'lesmis', 'cora', 'pancreas'):
    ns = max(len([k for k in R2 if k.startswith(f'cmp2_{d}_euc_')]), 1)
    ref = [R2[f'cmp2_{d}_h3_{s}']['logloss'] for s in range(ns) if f'cmp2_{d}_h3_{s}' in R2]
    cells = []
    lls = {}
    for m in methods:
        vals = [R2[f'cmp2_{d}_{m}_{s}'] for s in range(ns) if f'cmp2_{d}_{m}_{s}' in R2]
        if not vals:
            cells.append('--'); continue
        ll = np.array([x['logloss'] for x in vals]); au = np.array([x['auc'] for x in vals]); lls[m] = ll.mean()
        k = 4 if ll.mean() < 0.1 else 3
        if m != 'h3' and len(ref) == len(ll) and len(ll) > 1:
            diff = ll - np.array(ref); ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            cells.append(f"{ll.mean():.{k}f} [{diff.mean():+.{k}f}$\\pm${ci:.{k}f}] / {au.mean():.2f}")
        else:
            cells.append(f"{ll.mean():.{k}f} / {au.mean():.2f}")
    best = min(lls.values())
    cells = [(r'\textbf{' + c + '}' if (m in lls and lls[m] == best) else c) for m, c in zip(methods, cells)]
    rows.append(' & '.join([DN[d], str(ns)] + cells) + r' \\')
open('paper/tab_compare2.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Real networks: held-out log-loss / AUC of each method, averaged over random splits hiding $10\%$ of the dyads (number of splits in the second column). For the latent space models (fitted by BBVI without anchors, scales estimated, 800 iterations with $S=10$; 600 iterations with $20\,000$ subsampled dyads per sample for the two larger networks) the bracket gives the paired difference in log-loss to $\Hb^3$ with a $95\%$ confidence interval over splits. Best log-loss per row in bold. ASE: adjacency spectral embedding of rank 3 with iterative imputation; DC-SBM: degree-corrected block model with $K$ chosen by BIC; Chung--Lu: degree-only model.}\label{tab:compare2}
\resizebox{\textwidth}{!}{\begin{tabular}{lc''' + 'c' * len(methods) + r'''}\toprule
network & splits & ''' + ' & '.join(L[m] for m in methods) + r'''\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
# per-network best and margins as macros
for d in ('karate', 'lesmis', 'cora', 'pancreas'):
    ns = max(len([k for k in R2 if k.startswith(f'cmp2_{d}_euc_')]), 1)
    lls = {m: np.mean([R2[f'cmp2_{d}_{m}_{s}']['logloss'] for s in range(ns) if f'cmp2_{d}_{m}_{s}' in R2]) for m in methods if any(f'cmp2_{d}_{m}_{s}' in R2 for s in range(ns))}
    mac(f'{d}Best', L[min(lls, key=lls.get)].replace('$', '$'))
    mac(f'{d}Splits', str(ns))
# ---------- showcases: fixed vs inverse-gamma ----------
rows = []
for d, g in (('karate', 'sl2'), ('lesmis', 'nil')):
    for lab, key in (('fixed', f'show_{d}_{g}_fix'), ('InvGamma', f'show_{d}_{g}')):
        v = R[key]
        rows.append(' & '.join([DN[d] if lab == 'fixed' else '', L[g] if lab == 'fixed' else '', lab, f"{f(v['mcmc_alpha'],2)} ({f(v['mcmc_alpha_sd'],2)})", f"{f(v['vi_alpha'],2)} ({f(v['vi_alpha_sd'],2)})",
                                f"({f(v['mcmc_sig'][0],1)},{f(v['mcmc_sig'][1],1)})", f"({f(v['vi_sig'][0],1)},{f(v['vi_sig'][1],1)})", f(v['mcmc_auc'], 3), f(v['vi_auc'], 3), f(v['mcmc_logloss'], 3), f(v['vi_logloss'], 3),
                                f'{v["ll_mcmc"]:.0f}', f'{v["ll_vi"]:.0f}', f(v['mcmc_vi_Dcorr'], 2), f'{v["mcmc_time"]:.0f}', f'{v["vi_time"]:.0f}']) + r' \\')
open('paper/tab_show.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Real networks with anchors: MCMC (12\,000 iterations, 4\,000 discarded, thinned by 10) versus BBVI (2000 iterations, $S=20$), with the scales held fixed at their initial values (``fixed'') or estimated under the inverse-Gamma prior (``InvGamma''). $\alpha$: posterior mean (s.d.), variational mean (s.d.); estimated scales; in-sample AUC and log-loss; mean log-likelihood over retained draws / last 100 iterations; correlation of the posterior mean distance matrices of the two schemes; seconds.}\label{tab:show}
\resizebox{\textwidth}{!}{\begin{tabular}{lllccccccccccccc}\toprule
 & & & \multicolumn{2}{c}{$\alpha$} & \multicolumn{2}{c}{$(\hat\sigma_h,\hat\sigma_v)$} & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{log-loss} & \multicolumn{2}{c}{log-lik.} & dist. & \multicolumn{2}{c}{time (s)}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){15-16}
network & $\M$ & scales & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & MCMC & BBVI & corr. & MCMC & BBVI\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
for d, g in (('karate', 'sl2'), ('lesmis', 'nil')):
    v = R[f'show_{d}_{g}_fix']; mac(f'{d}Anchors', ', '.join(str(a + 1) for a in v['anchors'])); mac(f'{d}DcorrFix', f(v['mcmc_vi_Dcorr'], 2))
    w = R[f'show_{d}_{g}']; mac(f'{d}DcorrIG', f(w['mcmc_vi_Dcorr'], 2)); mac(f'{d}MCMCalphaIG', f(w['mcmc_alpha'], 1)); mac(f'{d}MCMCsigIG', f"({f(w['mcmc_sig'][0],1)},{f(w['mcmc_sig'][1],1)})")
    mac(f'{d}VIalphaIG', f(w['vi_alpha'], 1)); mac(f'{d}MCMCaucIG', f(w['mcmc_auc'], 3)); mac(f'{d}VIaucIG', f(w['vi_auc'], 3))
    mac(f'{d}MCMCaucFix', f(v['mcmc_auc'], 3)); mac(f'{d}VIaucFix', f(v['vi_auc'], 3)); mac(f'{d}AccZ', f"{v['mcmc_acc']['z']:.2f}")
# ---------- rigidity (exact certificates) ----------
rows = []
for g in ('nil', 'sol', 'sl2'):
    if g == 'nil':
        s = np.array(R['rigidity']['nil']['singular_values'][0]); k = R['rigidity']['nil']['expected_rank']; src = 'exact distance, central differences'
    else:
        s = np.array(R2[f'rigidity2_{g}']['singular_values']); k = R2[f'rigidity2_{g}']['expected_rank']; src = 'first-variation formula, BVP geodesics'
    rows.append(' & '.join([L[g], str(k), f'{s[0]:.2f}', f'{s[k-2]:.3f}', f'{s[k-1]:.3f}', f'{s[k]:.1e}', f'{s[-1]:.1e}', src]) + r' \\')
open('paper/tab_rigidity.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Rank certificate for Theorem~\ref{thm:generic}: singular values $s_1\ge\dots\ge s_{24}$ of the $28\times24$ differential of the distance map at a random configuration with $N=8$; $k=3N-\dim\Isom(\M)$ is the expected rank. For $\Nil$ the differential is computed by central differences of the exact distance; for $\Sol$ and $\SL$ from the first-variation formula $\nabla_q d(p,q)=\dot\gamma(1)^\flat$, with the minimising geodesic of each pair found by a boundary-value solver and checked against the distance table.}\label{tab:rigidity}
\begin{tabular}{lcccccccl}\toprule
$\M$ & $k$ & $s_1$ & $s_{k-1}$ & $s_k$ & $s_{k+1}$ & $s_{24}$ & method\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
# ---------- pancreas ----------
mac('pancNilR', f"{R2['panc_nil']['circ_R2_fibre']:.2f}"); mac('pancEucR', f"{R2['panc_euc']['circ_R2_fibre']:.2f}")
mac('pancNilCl', f"{100*R2['panc_nil']['cluster_between_share']:.0f}\\%"); mac('pancEucCl', f"{100*R2['panc_euc']['cluster_between_share']:.0f}\\%")
mac('pancN', str(R2['panc_nil']['N']))
open('paper/numbers.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
