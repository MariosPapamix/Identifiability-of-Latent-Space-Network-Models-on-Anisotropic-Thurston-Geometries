import json, numpy as np
R = json.load(open('results/results.json')); R2 = json.load(open('results2/results.json')); R3 = json.load(open('results3/results.json'))
L = {'euc': r'$\mathbb{R}^3$', 'h3': r'$\mathbb{H}^3$', 's3': r'$\mathbb{S}^3$', 'h2r': r'$\mathbb{H}^2\!\times\!\mathbb{R}$', 's2r': r'$\mathbb{S}^2\!\times\!\mathbb{R}$',
     'nil': r'$\Nil$', 'sol': r'$\Sol$', 'sl2': r'$\SL$', 'sl2p': r'$\mathrm{PSL}(2,\mathbb{R})$',
     'const': 'constant', 'beta': r'$\beta$-model', 'cl': 'Chung--Lu', 'rdpg': 'logistic RDPG', 'eigen': 'eigenmodel', 'dcsbm': 'DC-SBM'}
DN = {'karate': 'Karate club', 'lesmis': "Les Mis\\'erables", 'cora': 'Cora ball', 'pancreas': 'Pancreas kNN'}
out = []
def mac(n, v): out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
def f(x, k=3): return f'{x:.{k}f}'

# ---- augmented rank certificates ----
rows = []
v = R3['rig_nil']
for N in ('6', '7', '8', '10'):
    x = v[N]
    rows.append(' & '.join([r'$\Nil$', N, str(x['ndyads']), str(x['k']), str(x['k'] + 1), f"{x['s_min_slice']['median']:.3f} [{x['s_min_slice']['min']:.0e}, {x['s_min_slice']['max']:.3f}]",
                            f"{x['s_min_aug']['median']:.3f} [{x['s_min_aug']['min']:.0e}]", f"{100*(1-x['frac_rank_deficient']):.0f}\\% / {100*(1-x['frac_aug_deficient']):.0f}\\%", 'exact, 100 configurations']) + r' \\')
for g, Ns in (('sol', (6, 7)), ('sl2', (6, 7))):
    for N in Ns:
        x = R3[f'rig_{g}_N{N}']; s = x['sv_slice']; sa = x['sv_aug']
        aug = ('0 (structural, $15\\times16$)' if x['slice_dim'] + 1 > x['ndyads'] else f"{sa[-1]:.4f}")
        rows.append(' & '.join([L[g], str(N), str(x['ndyads']), str(x['slice_dim']), str(x['slice_dim'] + 1), f"{s[-1]:.4f}", aug, f"{x['rank_slice']} / {x['rank_aug']}", 'BVP geodesics, 1 configuration']) + r' \\')
open('paper/tab_rigidity.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Rank certificates for Theorem~\ref{thm:generic} and for joint identification with $\alpha$. $J$ is the differential of the distance map restricted to the tangent space of the gauge slice ($k=\dim\mathcal{S}_I$ columns, $\binom N2$ rows) and $[\mathbf 1,-J]$ its augmentation by the intercept direction. For $\Nil$ the differential is computed by central differences of the exact distance at $100$ random configurations, and the columns give the median [min, max] of the smallest singular value of $J$, the median [min] for $[\mathbf 1,-J]$, and the percentage of configurations at which $J$, resp.\ $[\mathbf 1,-J]$, has full column rank (tolerance $10^{-8}$). For $\Sol$ and $\SL$ the differential comes from the first-variation formula with minimising geodesics from a boundary-value solver at one configuration, and the last column gives the numerical ranks. For $\Sol$ at $N=6$ the augmented matrix has $16$ columns and $15$ rows, so joint identification with $\alpha$ is impossible.}\label{tab:rigidity}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccl}\toprule
$\M$ & $N$ & dyads & $k$ & $k{+}1$ & $s_{\min}(J)$ & $s_{\min}([\mathbf 1,-J])$ & full rank ($J$ / aug.) & method\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')

# ---- Fisher deficit table ----
d = R3['deficit']; rows = []
order = [k for k in d if k != 'complete']
for k in order:
    v = d[k]; truth = k.split('_')[0]; N = k.split('_N')[1].split('_')[0]; val = k.split('_')[-1]
    cells = [f"{v['deficits'][t]['fisher_deficit']:.1f}" if t in v['deficits'] else '--' for t in ('euc', 'h3', 'h2r')]
    rows.append(' & '.join([L[truth], N, val, f(v['alpha'], 1), str(v['ndyads'])] + cells) + r' \\')
open('paper/tab_deficit.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Fisher deficit $\Delta_T^2(Z,\alpha)$ of Proposition~\ref{prop:deficit} for the win-map configurations of Table~\ref{tab:winmap}: the weighted additive-constant distortion of the true distance matrix into each competitor geometry $T$, in nats (a local minimum of a non-convex weighted least-squares problem, hence an upper bound). Knob as in Table~\ref{tab:winmap}; the exact divergences are in Table~\ref{tab:lecam}.}\label{tab:deficit}
\begin{tabular}{llcccccc}\toprule
truth & $N$ & knob & $\alpha$ & dyads & $T=\mathbb{R}^3$ & $T=\mathbb{H}^3$ & $T=\mathbb{H}^2\!\times\!\mathbb{R}$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')

# ---- replicated coverage ----
rows = []
for g in ('nil', 'sol', 'sl2', 'euc', 'h3'):
    for lab, key in (('InvGamma$(2,2)$', f'covrep_{g}_ig'), ('InvGamma$(3,8)$', f'covrep_{g}_ig_matched'), ('true scales', f'covrep_{g}_true')):
        if key not in R3:
            continue
        v = R3[key]['reps']; n = len(v)
        def ms(field, k=2): a = np.array([x[field] for x in v]); return f"{a.mean():.{k}f} ({a.std(ddof=1)/np.sqrt(n):.{k}f})"
        rows.append(' & '.join([L[g] if lab.startswith('Inv') else '', lab, str(n), ms('cov90'), ms('cov50'), ms('cov90_prob'), f"{100*np.mean([x['alpha_cov'] for x in v]):.0f}\\%", ms('bias'), ms('rmse'), ms('alpha_mean')]) + r' \\')
open('paper/tab_covrep.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Replicated coverage study: independent latent configurations and networks ($N=30$; $(\sigma_h,\sigma_v)=(2,2)$ for $\Nil$ and $\mathbb{R}^3$, $(1.8,1.4)$ for $\Sol$, $(1.5,2)$ for $\SL$, $(1.5,1.5)$ for $\mathbb{H}^3$; $\alpha=2.5$; $4000$ MCMC iterations each; the constant-curvature fits are unanchored). Per replicate we record the fraction of true pairwise distances inside the $90\%$ and $50\%$ posterior intervals, the fraction of true tie probabilities inside their $90\%$ intervals, whether $\alpha$ is inside its $90\%$ interval, and the bias and root-mean-square error of the posterior mean distances; entries are means over replicates with Monte Carlo standard errors in parentheses. ``True scales'': $(\sigma_h,\sigma_v)$ fixed at their true values; ``InvGamma'': scales estimated.}\label{tab:covrep}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccccccc}\toprule
$\M$ & scales & networks & 90\% dist. & 50\% dist. & 90\% prob. & $\alpha$ covered & bias & RMSE & $\hat\alpha$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')

# ---- competitor table: geometries (cmp2) + corrected baselines (cmp3) ----
methods_geo = ['euc', 'h3', 's3', 'h2r', 's2r', 'nil', 'sol', 'sl2']; methods_base = ['const', 'beta', 'rdpg', 'eigen', 'dcsbm']
rows = []
for dname in ('karate', 'lesmis', 'cora', 'pancreas'):
    ns = max(len([k for k in R2 if k.startswith(f'cmp2_{dname}_euc_')]), 1)
    ref = [R2[f'cmp2_{dname}_h3_{s}']['logloss'] for s in range(ns) if f'cmp2_{dname}_h3_{s}' in R2]
    cells = []; lls = {}
    for m in methods_geo + methods_base:
        src = R2 if m in methods_geo else R3; pre = 'cmp2' if m in methods_geo else 'cmp3'
        vals = [src[f'{pre}_{dname}_{m}_{s}'] for s in range(ns) if f'{pre}_{dname}_{m}_{s}' in src]
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
    cells = [(r'\textbf{' + c + '}' if (m in lls and abs(lls[m] - best) < 5e-4) else c) for m, c in zip(methods_geo + methods_base, cells)]
    rows.append(' & '.join([DN[dname], str(ns)] + cells) + r' \\')
    mac(f'{dname}Const', f(np.mean([R3[f'cmp3_{dname}_const_{s}']['logloss'] for s in range(ns)]), 4 if dname in ('cora', 'pancreas') else 3))
    mac(f'{dname}BestBase', f(min(np.mean([R3[f'cmp3_{dname}_{m}_{s}']['logloss'] for s in range(ns)]) for m in ('beta', 'rdpg', 'eigen', 'dcsbm')), 4 if dname in ('cora', 'pancreas') else 3))
    mac(f'{dname}BestGeo', f(min(lls[m] for m in methods_geo if m in lls), 4 if dname in ('cora', 'pancreas') else 3))
open('paper/tab_compare2.tex', 'w').write(r'''\begin{table}[H]\centering\scriptsize
\caption{Real networks: held-out log-loss / AUC, averaged over random splits hiding $10\%$ of the dyads (number of splits in the second column; the intervals describe variability over splits of one network, not over networks). Latent space models are fitted by BBVI without anchors at unit slope, with the scales and the centre given variational factors (800 iterations, $n_s=10$; 600 iterations with $20\,000$ subsampled dyads per sample for the two larger networks); the bracket is the paired difference in log-loss to $\mathbb{H}^3$ with a $95\%$ interval over splits. Baselines (Appendix~\ref{app:details}): constant density; $\beta$-model; logistic random dot-product model on a rank-3 spectral embedding; spectral eigenmodel; degree-corrected block model with $K\le8$ by BIC. Best log-loss per row in bold (ties within $0.0005$). $\mathbb{S}^3$ and $\mathbb{S}^2\times\mathbb{R}$ were not run on the two larger networks.}\label{tab:compare2}
\resizebox{\textwidth}{!}{\begin{tabular}{lc''' + 'c' * (len(methods_geo) + len(methods_base)) + r'''}\toprule
network & splits & ''' + ' & '.join(L[m] for m in methods_geo + methods_base) + r'''\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}}\end{table}''')

# ---- learned slope ----
rows = []
for dname in ('karate', 'lesmis'):
    for m in ('euc', 'h3', 'h2r', 'nil', 'sl2', 'sl2p', 'sol'):
        v = [R3[f'cmpb_{dname}_{m}_{s}'] for s in range(5) if f'cmpb_{dname}_{m}_{s}' in R3]
        if not v:
            continue
        fixed = [R2[f'cmp2_{dname}_{m}_{s}']['logloss'] for s in range(5) if f'cmp2_{dname}_{m}_{s}' in R2]
        rows.append(' & '.join([DN[dname] if m == 'euc' else '', L[m], f"{np.mean([x['logloss'] for x in v]):.3f}", f"{np.mean([x['auc'] for x in v]):.3f}", f"{np.mean([x['beta'] for x in v]):.2f}", f"{np.mean([x['alpha'] for x in v]):.2f}", (f"{np.mean(fixed):.3f}" if fixed else '--')]) + r' \\')
open('paper/tab_beta.tex', 'w').write(r'''\begin{table}[H]\centering\small
\caption{Learned slope: variational fits of $\logit p_{ij}=\alpha-\beta\,d_\M(z_i,z_j)$ with variational factors for $\alpha$, $\log\beta$, the scales and the centre (unanchored, 800 iterations, $n_s=10$), on the first five held-out splits; held-out log-loss, AUC, posterior means of $\beta$ and $\alpha$, and the unit-slope log-loss on the same splits. $\mathrm{PSL}(2,\mathbb{R})$ is the compact-fibre quotient of $\SL$ (fibre period $2\pi$).}\label{tab:beta}
\begin{tabular}{llccccc}\toprule
network & $\M$ & log-loss & AUC & $\hat\beta$ & $\hat\alpha$ & log-loss at $\beta=1$\\ \midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}''')
# macros
c = R3['covrep_nil_ig']['reps']; t = R3['covrep_nil_true']['reps']
mac('covNilIG', f"{100*np.mean([x['cov90'] for x in c]):.0f}\\%"); mac('covNilTrue', f"{100*np.mean([x['cov90'] for x in t]):.0f}\\%")
mac('covEucIG', f"{100*np.mean([x['cov90'] for x in R3['covrep_euc_ig']['reps']]):.0f}\\%"); mac('covEucTrue', f"{100*np.mean([x['cov90'] for x in R3['covrep_euc_true']['reps']]):.0f}\\%")
mac('covSLIG', f"{100*np.mean([x['cov90'] for x in R3['covrep_sl2_ig']['reps']]):.0f}\\%"); mac('covSLTrue', f"{100*np.mean([x['cov90'] for x in R3['covrep_sl2_true']['reps']]):.0f}\\%")
mac('alphaCovIG', f"{100*np.mean([x['alpha_cov'] for x in c]):.0f}\\%"); mac('alphaCovTrue', f"{100*np.mean([x['alpha_cov'] for x in t]):.0f}\\%")
mac('biasIG', f"{np.mean([x['bias'] for x in c]):.2f}"); mac('biasTrue', f"{np.mean([x['bias'] for x in t]):.2f}")
x = R3['rig_nil']['6']; mac('nilSixMedian', f"{x['s_min_slice']['median']:.3f}"); mac('nilSixMin', f"{x['s_min_slice']['min']:.0e}"); mac('nilTenMedian', f"{R3['rig_nil']['10']['s_min_slice']['median']:.3f}")
dd = R3['deficit']; mac('defNilBig', f"{dd['nil_N100_3.0']['deficits']['h3']['fisher_deficit']:.0f}"); mac('defNilBigRatio', f"{dd['nil_N100_3.0']['deficits']['h3']['per_parameter']:.2f}")
open('paper/numbers3.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
