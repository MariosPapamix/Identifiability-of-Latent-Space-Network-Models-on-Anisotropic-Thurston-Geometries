import json, numpy as np
R2 = json.load(open('results2/results.json')); R3 = json.load(open('results3/results.json'))
L = {'euc': r'$\mathbb{R}^3$', 'h3': r'$\mathbb{H}^3$', 'h2r': r'$\mathbb{H}^2\!\times\!\mathbb{R}$', 'nil': r'$\Nil$', 'sol': r'$\Sol$', 'sl2': r'$\SL$'}
out = []
def mac(n, v): out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
# ---- replicated win maps: mean (sd) over the original network + 2 replicates, N=40/60 cells ----
rows = []
for truth, fits, N in (('nil', ('nil', 'euc', 'h3', 'h2r'), 40), ('nil', ('nil', 'euc', 'h3', 'h2r'), 100), ('sol', ('sol', 'euc', 'h3', 'h2r'), 60), ('sl2', ('sl2', 'h3', 'h2r', 'nil'), 60)):
    vals = sorted({float(k.split('_')[-2]) for k in R3 if k.startswith(f'winrep_{truth}_N{N}_')})
    for val in vals:
        runs = [R2[f'win_{truth}_N{N}_{val}']] + [R3[k] for k in R3 if k.startswith(f'winrep_{truth}_N{N}_{val}_') and all(g in R3[k]['fits'] for g in fits)]
        cells = []
        for g in fits:
            if g == truth:
                cells.append(f"{np.mean([r['fits'][g]['logloss_rep'] for r in runs]):.3f}")
            else:
                d = np.array([r['fits'][truth]['logloss_rep'] - r['fits'][g]['logloss_rep'] for r in runs])
                cells.append(f"{d.mean():+.4f} ({d.std(ddof=1):.4f})")
        nwin = sum(1 for r in runs if r['fits'][truth]['logloss_rep'] <= min(r['fits'][g]['logloss_rep'] for g in fits) + 1e-9)
        rows.append(' & '.join([L[truth], str(N), str(val), str(len(runs)), f"{np.mean([r['oracle'] for r in runs]):.3f}"] + [(L[g] + ': ' if g != truth else 'own: ') + c for g, c in zip(fits, cells)] + [f"{nwin}/{len(runs)}"]) + r' \\')
open('paper/tab_winrep.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Replicated win maps: five independently generated networks per cell (three for the $N=100$ cell; knob as in Table~\ref{tab:winmap}). ``own'': mean replicate log-loss of the fit under the true geometry; the other columns give the \emph{paired} difference (true geometry minus competitor, negative favours the truth) with its standard deviation over networks; ``wins'': networks on which the true geometry had the lowest (or tied lowest) replicate log-loss.}\label{tab:winrep}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccllllc}\toprule
truth & $N$ & knob & networks & oracle & \multicolumn{4}{c}{own log-loss; paired differences to competitors, mean (s.d.)} & wins\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
# ---- exact KL / Fisher / per-dyad / Le Cam table ----
rows = []
for k, v in R3['deficit_kl'].items():
    truth = k.split('_')[0]; N = k.split('_N')[1].split('_')[0]; val = k.split('_')[-1]; nd = v['ndyads']
    def fis(t):
        a = v['targets'][t]['fisher_deficit']; b = R3['deficit'][k]['deficits'][t]['fisher_deficit'] if k in R3['deficit'] and t in R3['deficit'][k]['deficits'] else a
        return min(a, b)
    cells = [(f"{fis(t):.1f} / {v['targets'][t]['kl_min']:.1f} / {v['targets'][t]['kl_min']/nd:.4f} / {v['targets'][t]['power_bound']:.2f}" if t in v['targets'] else '--') for t in ('euc', 'h3', 'h2r')]
    rows.append(' & '.join([L[truth], N, val, str(nd)] + cells) + r' \\')
open('paper/tab_lecam.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Representation deficit of the win-map configurations against each competitor geometry $T$. Each cell gives: the Fisher deficit $\Delta_T^2$ (nats); the minimum exact dyad divergence $\mathrm{KL}_T$ found by direct optimisation (an upper bound on the infimum; nats); $\mathrm{KL}_T$ per dyad, which is the expected held-out log-loss advantage of the true model over the best competitor to first order; and the bound $0.05+\sqrt{\mathrm{KL}_T/2}$ of Theorem~\ref{thm:lecam} on the power of any level-$0.05$ test from one network ($1$ is vacuous).}\label{tab:lecam}
\resizebox{\textwidth}{!}{\begin{tabular}{lllcccc}\toprule
truth & $N$ & knob & dyads & $T=\mathbb{R}^3$ & $T=\mathbb{H}^3$ & $T=\mathbb{H}^2\!\times\!\mathbb{R}$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')
# ---- annealed MLE table ----
rows = []
for d in ('karate', 'lesmis', 'cora'):
    for m in ('euc', 'h3', 'h2r', 'nil', 'sl2', 'sol'):
        v = [R3[f'anneal_{d}_{m}_{s}'] for s in range(3) if f'anneal_{d}_{m}_{s}' in R3]
        if not v:
            continue
        bay = [R2[f'cmp2_{d}_{m}_{s}']['logloss'] for s in range(3) if f'cmp2_{d}_{m}_{s}' in R2]
        rows.append(' & '.join([{'karate': 'Karate club', 'lesmis': "Les Mis\\'erables", 'cora': 'Cora ball'}[d] if m == 'euc' else '', L[m], f"{np.mean([x['logloss'] for x in v]):.4f}", f"{np.mean([x['auc'] for x in v]):.3f}", f"{np.mean([x['beta'] for x in v]):.1f}", f"{np.mean([x['alpha'] for x in v]):.1f}", (f"{np.mean(bay):.4f}" if bay else '--')]) + r' \\')
open('paper/tab_anneal.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Annealed maximum likelihood with learned $(R,T)$, in the style of \citet{celinska2024thurston} (our implementation: 60 annealing sweeps over positions with $(\alpha,\beta)=(R/T,1/T)$ re-fitted by logistic regression after each sweep), on the first three held-out splits: held-out log-loss, AUC, fitted $\beta$ and $\alpha$, and the held-out log-loss of the Bayesian unit-slope fit on the same splits (not available for the Cora ball, whose Bayesian fits used other splits).}\label{tab:anneal}
\begin{tabular}{llccccc}\toprule
network & $\M$ & log-loss & AUC & $\hat\beta$ & $\hat\alpha$ & Bayesian, $\beta=1$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
# ---- certificate macros ----
c = R3['rig_nil_interval']
mac('certSixLower', f"{c['6']['certified_s_min_lower']:.5f}"); mac('certSixFloat', f"{c['6']['s_min_float']:.5f}"); mac('certSixAug', f"{c['6']['aug_certified_s_min_lower']:.5f}")
mac('certSevenLower', f"{c['7']['certified_s_min_lower']:.5f}"); mac('certSevenAug', f"{c['7']['aug_certified_s_min_lower']:.5f}"); mac('certWidth', f"{c['6']['max_interval_width']:.0e}")
# ---- coverage macros for all geometries ----
names = {'nil': 'Nil', 'sol': 'Sol', 'sl2': 'SLt', 'euc': 'Euc', 'h3': 'Hthree'}
for g in ('nil', 'sol', 'sl2', 'euc', 'h3'):
    for lab in ('ig', 'true'):
        v = R3[f'covrep_{g}_{lab}']['reps']; mac(f'cov{names[g]}{lab}', f"{100*np.mean([x['cov90'] for x in v]):.0f}\\%")
c2 = R3['covrep_nil_ig_matched']['reps']
mac('covNilMatched', f"{100*np.mean([x['cov90'] for x in c2]):.0f}\\%"); mac('alphaCovMatched', f"{100*np.mean([x['alpha_cov'] for x in c2]):.0f}\\%"); mac('biasMatched', f"{np.mean([x['bias'] for x in c2]):.2f}")
dg = R3['diag_nil']['stats']
mac('rhatAlpha', f"{dg['alpha']['rhat']:.2f}"); mac('rhatSig', f"{dg['sigh']['rhat']:.2f}"); mac('rhatDmax', f"{max(dg[k]['rhat'] for k in ('d1','d2','d3')):.2f}")
mac('essAlpha', f"{dg['alpha']['ess']:.0f}"); mac('essDmin', f"{min(dg[k]['ess'] for k in ('d1','d2','d3')):.0f}")
kk = R3['deficit_kl']
mac('klBigH', f"{kk['nil_N100_3.0']['targets']['h3']['kl_min']:.0f}"); mac('klBigHtwoR', f"{kk['nil_N100_3.0']['targets']['h2r']['kl_min']:.0f}")
mac('klBigPerDyadH', f"{kk['nil_N100_3.0']['targets']['h3']['kl_min']/4950:.4f}"); mac('klBigPerDyadHtwoR', f"{kk['nil_N100_3.0']['targets']['h2r']['kl_min']/4950:.4f}")
# lesmis anneal ranking of Sol
v = {m: np.mean([R3[f'anneal_lesmis_{m}_{s}']['logloss'] for s in range(3)]) for m in ('euc', 'h3', 'h2r', 'nil', 'sl2', 'sol')}
mac('annealSolRank', str(sorted(v, key=v.get).index('sol') + 1)); mac('annealSolLL', f"{v['sol']:.3f}"); mac('annealHLL', f"{v['h3']:.3f}")
open('paper/numbers4.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
