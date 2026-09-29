"""Builds 02_explore.ipynb. Kept beside the notebook so the analysis cells are diffable."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# PISA 2022: 02 Explore, what's going on

Reads the Parquet files made by `01_extract.ipynb` and works through each PISA 2022 file.

**Method (the OECD-prescribed way, not a shortcut):**
- every estimate uses the final student weight `W_FSTUWT`,
- standard errors use the 80 **Fay BRR** replicate weights (`W_FSTURWT1-80`, k = 0.5),
- scores use **all 10 plausible values**, combined with Rubin's rules: total var = sampling var + (1 + 1/10) x imputation var,
- the *OECD average* is the unweighted mean of OECD-country estimates, as in the OECD reports.

Section 0 checks these numbers against the published PISA 2022 results before anything else is trusted.

Set `FOCUS` below to profile any country (3-letter code).
""")

code("""
FOCUS = 'ARE'          # country spotlighted in section 15, change freely

from pathlib import Path
import json, re, warnings
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'requirements.txt').exists())  # repo root
PQ, OUT = ROOT/'data/parquet', ROOT/'00_pre-project/outputs'
FIG = OUT/'figures'; FIG.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 120,
              'display.float_format', '{:,.2f}'.format)

# palette (dataviz reference instance): categorical slots 1-3, ink, OECD reference in gray
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
mpl.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12,
    'font.size': 9, 'figure.dpi': 110, 'savefig.bbox': 'tight'})

def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=150); plt.show()

DD = pd.read_csv(ROOT/'data/data_dictionary.csv')
# each dataset is a folder of Parquet parts (<90 MB each, so it fits on GitHub); pandas reads a folder like one file
def cols(stem): return pq.ParquetDataset(PQ/f'CY08MSP_{stem}').schema.names
def load(stem, columns): return pd.read_parquet(PQ/f'CY08MSP_{stem}', columns=columns)
def batches(stem, columns, size=50_000):
    # stream a dataset in fixed-size row batches across its parts, in file order
    import pyarrow as pa
    buf = None
    for f in sorted((PQ/f'CY08MSP_{stem}').glob('part-*.parquet')):
        for b in pq.ParquetFile(f).iter_batches(batch_size=size, columns=columns):
            t = pa.Table.from_batches([b])
            buf = t if buf is None else pa.concat_tables([buf, t])
            while buf.num_rows >= size:
                yield buf.slice(0, size); buf = buf.slice(size)
    if buf is not None and buf.num_rows: yield buf
""")

md("## Estimation engine (BRR + plausible values)")

code("""
W = ['W_FSTUWT'] + [f'W_FSTURWT{i}' for i in range(1, 81)]
FAY = 1 / (80 * 0.5**2)                      # = 0.05
PV = lambda dom: [f'PV{i}{dom}' for i in range(1, 11)]

def _mean(Y, Wm):
    \"\"\"weighted means for every (weight, column): (n,81) x (n,P) -> (81,P); NaN-aware per column\"\"\"
    M = ~np.isnan(Y)
    return (Wm.T @ np.where(M, Y, 0.0)) / (Wm.T @ M)

def _slope(x, Y, Wm, r2=False):
    \"\"\"weighted OLS slope of each Y column on x (or R^2), per weight -> (81,P)\"\"\"
    Wm = Wm * (~np.isnan(x))[:, None]; x = np.nan_to_num(x)
    sw = Wm.sum(0)[:, None]
    mx, my = (Wm.T @ x)[:, None] / sw, (Wm.T @ Y) / sw
    sxy = (Wm.T @ (x[:, None] * Y)) / sw - mx * my
    sxx = (Wm.T @ (x * x))[:, None] / sw - mx**2
    if not r2: return sxy / sxx
    syy = (Wm.T @ (Y * Y)) / sw - my**2
    return sxy**2 / (sxx * syy)

def _combine(E):
    \"\"\"E (81,P): row 0 = full weight, rows 1-80 = replicates, columns = plausible values\"\"\"
    full = E[0]; P = E.shape[1]
    samp = (FAY * ((E[1:] - full)**2).sum(0)).mean()
    imp = full.var(ddof=1) if P > 1 else 0.0
    return full.mean(), np.sqrt(samp + (1 + 1/P) * imp)

def estimate(df, fn, by='CNT', weights=W):
    out = {}
    for g, d in df.groupby(by, sort=True):
        E = np.asarray(fn(d, d[weights].to_numpy(float)), float)
        out[g] = _combine(E.reshape(81, -1))
    return pd.DataFrame(out, index=['est', 'se']).T

# building blocks
def mean_of(c):  return lambda d, Wm: _mean(d[c].to_numpy(float), Wm)
def pct_of(c, cond):
    def f(d, Wm):
        Y = d[c].to_numpy(float)
        return 100 * _mean(np.where(np.isnan(Y), np.nan, cond(Y).astype(float)), Wm)
    return f
def gap_of(c, grp, a, b):
    def f(d, Wm):
        Y, g = d[c].to_numpy(float), d[grp].to_numpy()
        return _mean(Y, Wm * (g == a)[:, None]) - _mean(Y, Wm * (g == b)[:, None])
    return f
def mean_where(c, grp, val):
    return lambda d, Wm: _mean(d[c].to_numpy(float), Wm * (d[grp].to_numpy() == val)[:, None])
def slope_of(x, c, r2=False):
    return lambda d, Wm: _slope(d[x].to_numpy(float), d[c].to_numpy(float), Wm, r2)
""")

md("## Load the student file (only the columns used)")

code("""
IDX = ['ESCS','HISEI','PAREDINT','HOMEPOS','ICTRES','BELONG','BULLIED','FEELSAFE','SCHRISK','RELATST',
       'ANXMAT','MATHEFF','MATHEF21','MATHPERS','GROSAGR','PERSEVAGR','CURIOAGR','STRESAGR','EMOCOAGR',
       'DISCLIM','TEACHSUP','COGACRCO','COGACMCO','EXPOFA','EXPO21ST','FAMSUP','CREATEFF',
       'SDLEFF','PROBSELF','FAMSUPSL','FEELLAH','SCHSUST','LIFESAT','PSYCHSYM','ICTEFFIC','ICTDISTR']
EXTRA = ['ST004D01T','AGE','GRADE','IMMIG','REPEAT','SKIPPING','WORKPAY','EXPECEDU','EFFORT1','EFFORT2',
         'ST273Q06JA','ST273Q07JA','ST347Q01JA','ST350Q01JA','ST356Q01JA','ST255Q01JA']
SUB = ['MCCR','MCQN','MCSS','MCUD','MPEM','MPFS','MPIN','MPRE']
ID = ['CNT','CNTRYID','CNTSCHID','CNTSTUID','OECD','ADMINMODE',
      'Option_CT','Option_FL','Option_ICTQ','Option_WBQ','Option_PQ','Option_TQ']
have = set(cols('STU_QQQ'))
want = ID + W + PV('MATH') + PV('READ') + PV('SCIE') + sum((PV(s) for s in SUB), []) + IDX + EXTRA
missing = [c for c in want if c not in have]
print('not in file:', missing)
stu = load('STU_QQQ', [c for c in want if c in have])
stu['GENDER'] = stu.ST004D01T.map({1: 'F', 2: 'M'})
print(f'{len(stu):,} students x {stu.shape[1]} cols loaded, {stu.memory_usage().sum()/1e9:.2f} GB')

# country names from the SPSS value labels of CNTRYID
lab = json.loads(DD.query(\"dataset=='student' and variable=='CNTRYID'\").value_labels.iloc[0])
cmap = stu[['CNT','CNTRYID']].drop_duplicates().set_index('CNT').CNTRYID
NAME = {c: lab.get(str(float(i)), c) for c, i in cmap.items()}
OECD = sorted(stu.loc[stu.OECD == 1, 'CNT'].unique())
print(f'{stu.CNT.nunique()} countries/economies, {len(OECD)} OECD')
""")

code("""
def with_oecd(res):
    \"\"\"append the OECD average row (unweighted mean of country estimates)\"\"\"
    r = res.loc[res.index.isin(OECD)]
    return pd.concat([res, pd.DataFrame({'est': [r.est.mean()], 'se': [np.sqrt((r.se**2).sum()) / len(r)]},
                                        index=['OECD'])])

def table(specs, df=None, by='CNT', weights=W):
    \"\"\"specs: {name: fn} -> wide DataFrame with name and name_se columns, plus OECD average row\"\"\"
    df = stu if df is None else df
    parts = {}
    for k, fn in specs.items():
        r = with_oecd(estimate(df, fn, by, weights))
        parts[k], parts[k + '_se'] = r.est, r.se
    t = pd.DataFrame(parts)
    t.insert(0, 'name', [NAME.get(i, 'OECD average') for i in t.index])
    return t
""")

md("""
## 0. Validation against the published PISA 2022 results

OECD, *PISA 2022 Results (Volume I)*: OECD average **472 math, 476 reading, 485 science**; Singapore 575 / 543 / 561.
If these don't match to within a point, nothing further down is trustworthy.
""")

code("""
scores = table({'math': mean_of(PV('MATH')), 'read': mean_of(PV('READ')), 'sci': mean_of(PV('SCIE'))})
published = {'OECD': (472, 476, 485), 'SGP': (575, 543, 561), 'JPN': (536, 516, 547), 'USA': (465, 504, 499)}
chk = pd.DataFrame({k: dict(zip(['math','read','sci'], v)) for k, v in published.items()}).T
ours = scores.loc[chk.index, ['math','read','sci']].round(0)
display(pd.concat({'published': chk, 'this notebook': ours}, axis=1))
assert (ours - chk).abs().max().max() <= 1, 'estimates do not reproduce the published results'
print('OK: estimates reproduce the OECD published means')
""")

md("## 1. Who was tested: coverage of every file")

code("""
def n_by(stem, key='CNT'):
    return load(stem, [key])[key].value_counts()

cov = pd.DataFrame({
    'students':        stu.CNT.value_counts(),
    'schools':         load('SCH_QQQ', ['CNT']).CNT.value_counts(),
    'teachers':        n_by('TCH_QQQ'),
    'creative_think':  n_by('CRT_COG'),
    'fin_literacy':    n_by('FLT_QQQ'),
    'population_15yo': stu.groupby('CNT').W_FSTUWT.sum().round(-3),
}).fillna(0).astype(int)
cov.insert(0, 'name', cov.index.map(NAME)); cov.insert(1, 'OECD', cov.index.isin(OECD))
print(f\"{cov.students.sum():,} students in {len(cov)} systems represent ~{cov.population_15yo.sum()/1e6:.1f}M 15-year-olds; \"
      f\"{cov.schools.sum():,} schools, {cov.teachers.sum():,} teachers\")
print('optional modules taken up by:', {c: int((cov[c] > 0).sum()) for c in ['teachers','creative_think','fin_literacy']}, 'systems')
cov.sort_values('students', ascending=False)
""")

md("## 2. Headline scores: math, reading, science")

code("""
scores = scores.sort_values('math', ascending=False)
scores['rank_math'] = scores.drop('OECD').math.rank(ascending=False).astype('Int64')
display(scores[['name','math','math_se','read','read_se','sci','sci_se','rank_math']].round(1))

def rank_chart(t, col, title, xlabel, name, oecd_val=None):
    t = t.drop('OECD', errors='ignore').sort_values(col)
    fig, ax = plt.subplots(figsize=(8, max(4, .19 * len(t))))
    ax.barh(t.name, t[col], color=[ORANGE if i == FOCUS else BLUE for i in t.index], height=.72)
    if col + '_se' in t:
        ax.errorbar(t[col], t.name, xerr=1.96 * t[col + '_se'], fmt='none', ecolor=INK2, lw=.6)
    if oecd_val is not None:
        ax.axvline(oecd_val, color=INK2, ls='--', lw=1)
        ax.text(oecd_val, len(t) - .3, f' OECD avg {oecd_val:.0f}', color=INK2, va='bottom', fontsize=8)
    ax.set_title(title, loc='left'); ax.set_xlabel(xlabel); ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', labelsize=7); ax.margins(y=.005)
    save(fig, name)

rank_chart(scores, 'math', 'Mathematics, PISA 2022 (mean score, 95% CI)', 'score', '02_math_rank',
           scores.loc['OECD', 'math'])
""")

code("""
# how different are the three domains? correlation across countries + read-vs-math scatter
c = scores.drop('OECD')
print('between-country correlation of domain means:'); display(c[['math','read','sci']].corr().round(3))
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(c.math, c.read, s=28, color=BLUE, edgecolor=SURF, lw=1)
ax.plot([330, 590], [330, 590], color=INK2, lw=.8, ls=':')
for i in c.index:
    if i in ('SGP','IRL','USA','GBR','JPN','KHM','DOM','PHL', FOCUS) or abs(c.math[i] - c.read[i]) > 35:
        ax.annotate(c.name[i], (c.math[i], c.read[i]), fontsize=7, xytext=(3, 2), textcoords='offset points', color=INK2)
ax.set(xlabel='mathematics', ylabel='reading', title='Reading vs mathematics (dotted = equal)')
save(fig, '02_read_vs_math')
""")

md("## 3. Proficiency levels: low performers and top performers")

code("""
# Level-2 and Level-5 lower bounds on each PISA scale (PISA 2022 Technical Report)
CUT = {'MATH': (420.07, 606.99), 'READ': (407.47, 625.61), 'SCIE': (409.54, 633.33)}
prof = table({f'low_{d}': pct_of(PV(d), lambda y, c=c: y < c[0]) for d, c in CUT.items()} |
             {f'top_{d}': pct_of(PV(d), lambda y, c=c: y >= c[1]) for d, c in CUT.items()})
prof = prof.sort_values('low_MATH')
display(prof[['name','low_MATH','top_MATH','low_READ','top_READ','low_SCIE','top_SCIE']].round(1))

# the full math level distribution per country
MLV = [-np.inf, 233.17, 295.47, 357.77, 420.07, 482.38, 544.68, 606.99, 669.30, np.inf]
LVN = ['<1c','1c','1b','1a','2','3','4','5','6']
lev = table({L: pct_of(PV('MATH'), lambda y, a=a, b=b: (y >= a) & (y < b))
             for L, a, b in zip(LVN, MLV[:-1], MLV[1:])})[['name'] + LVN]
lev.round(1).sort_values('2')
""")

code("""
# diverging stacked bar: below Level 2 to the left, Level 2+ to the right
t = lev.sort_values(['<1c','1c','1b','1a'], ascending=True).copy()
t = t.assign(below=t[['<1c','1c','1b','1a']].sum(axis=1)).sort_values('below', ascending=False)
blues = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']        # levels 2..6
reds  = ['#f3c3c2', '#ea8f8e', '#e34948', '#a52f2e']                   # 1a..<1c going outward
fig, ax = plt.subplots(figsize=(9, .19 * len(t) + 1))
left = np.zeros(len(t))
for L, col in zip(['1a','1b','1c','<1c'], reds):
    ax.barh(t.name, -t[L], left=-left, color=col, height=.75, label=f'Level {L}', edgecolor=SURF, lw=.5)
    left += t[L].to_numpy()
right = np.zeros(len(t))
for L, col in zip(['2','3','4','5','6'], blues):
    ax.barh(t.name, t[L], left=right, color=col, height=.75, label=f'Level {L}', edgecolor=SURF, lw=.5)
    right += t[L].to_numpy()
ax.axvline(0, color=INK, lw=.8); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', labelsize=7)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{abs(v):.0f}%'))
ax.set_title('Mathematics proficiency levels (left of 0 = below the Level 2 baseline)', loc='left')
ax.legend(ncol=9, fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.02), frameon=False)
ax.margins(y=.003); save(fig, '03_math_levels')
""")

md("""
## 4. Mathematics sub-scales: where each system is relatively strong or weak

Content (MCCR change & relationships, MCQN quantity, MCSS space & shape, MCUD uncertainty & data)
and process (MPEM employ, MPFS formulate, MPIN interpret, MPRE reason) sub-scales, as a difference from the country's overall math mean.
""")

code("""
subs = table({s: (lambda d, Wm, s=s: _mean(d[PV(s)].to_numpy(float), Wm) - _mean(d[PV('MATH')].to_numpy(float), Wm))
              for s in SUB})
LAB = {'MCCR':'change & relat.','MCQN':'quantity','MCSS':'space & shape','MCUD':'uncertainty & data',
       'MPEM':'employ','MPFS':'formulate','MPIN':'interpret','MPRE':'reason'}
h = subs.set_index('name')[SUB].rename(columns=LAB).loc[lambda x: x.index != 'OECD average']
h = h.loc[scores.drop('OECD').name]          # order by overall math
fig, ax = plt.subplots(figsize=(7, .19 * len(h) + 1))
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
div = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#ea8f8e', '#a52f2e'])
im = ax.imshow(h.to_numpy(), cmap=div.reversed(), norm=TwoSlopeNorm(0, -25, 25), aspect='auto')
ax.set_xticks(range(len(h.columns)), h.columns, rotation=40, ha='right'); ax.set_yticks(range(len(h)), h.index, fontsize=7)
ax.grid(False); plt.colorbar(im, ax=ax, shrink=.3, label='sub-scale minus overall math')
ax.set_title('Relative strengths in math (blue = stronger than own average)', loc='left')
save(fig, '04_math_subscales')
""")

md("## 5. Gender gaps (boys minus girls)")

code("""
gg = table({f'gap_{d}': gap_of(PV(d), 'GENDER', 'M', 'F') for d in ['MATH','READ','SCIE']})
gg = gg.sort_values('gap_MATH')
display(gg.round(1))
o = gg.loc['OECD']
print(f\"OECD average: boys {o.gap_MATH:+.1f} in math, {o.gap_READ:+.1f} in reading, {o.gap_SCIE:+.1f} in science\")
sig = lambda k: ((gg.drop('OECD')[k].abs() > 1.96 * gg.drop('OECD')[k + '_se']) * np.sign(gg.drop('OECD')[k])).value_counts()
print('math gap significant  (+1 boys ahead, -1 girls ahead, 0 n.s.):', sig('gap_MATH').to_dict())
print('reading gap significant:', sig('gap_READ').to_dict())

fig, ax = plt.subplots(figsize=(7, 6))
t = gg.drop('OECD')
ax.axhline(0, color=INK2, lw=.8); ax.axvline(0, color=INK2, lw=.8)
ax.scatter(t.gap_MATH, t.gap_READ, s=26, color=BLUE, edgecolor=SURF)
for i in t.index:
    if abs(t.gap_MATH[i]) > 15 or t.gap_READ[i] < -45 or i == FOCUS:
        ax.annotate(t.name[i], (t.gap_MATH[i], t.gap_READ[i]), fontsize=7, xytext=(3, 2), textcoords='offset points', color=INK2)
ax.set(xlabel='math gap (boys - girls)', ylabel='reading gap (boys - girls)',
       title='Girls lead reading everywhere; math gaps are small and mixed')
save(fig, '05_gender')
""")

md("""
## 6. Equity: how much does socio-economic status (ESCS) predict math?

- **gradient**: score points per 1-unit ESCS increase (within country),
- **strength**: % of math variance explained by ESCS,
- math gap between top- and bottom-quarter ESCS students in each country.
""")

code("""
def wquartile(d):
    # weighted within-country ESCS quartile (NaN where ESCS missing; all-NaN if a country has no ESCS)
    d = d[['ESCS', 'W_FSTUWT']].dropna().sort_values('ESCS')
    share = d.W_FSTUWT.cumsum() / d.W_FSTUWT.sum()
    return np.minimum(np.ceil(share * 4), 4)
stu['ESCS_Q'] = pd.concat([wquartile(d) for _, d in stu.groupby('CNT')]).reindex(stu.index)
print('systems without ESCS:', [NAME[c] for c, v in stu.groupby('CNT').ESCS.count().items() if v == 0])
eq = table({'escs_mean': mean_of(['ESCS']),
            'gradient':  slope_of('ESCS', PV('MATH')),
            'pct_var':   lambda d, Wm: 100 * _slope(d.ESCS.to_numpy(float), d[PV('MATH')].to_numpy(float), Wm, r2=True),
            'q4_minus_q1': gap_of(PV('MATH'), 'ESCS_Q', 4.0, 1.0)})
eq = eq.join(scores[['math']])
display(eq.sort_values('pct_var').round(2))

t = eq.drop('OECD'); o = eq.loc['OECD']
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.axvline(o.pct_var, color=INK2, ls='--', lw=.8); ax.axhline(o.math, color=INK2, ls='--', lw=.8)
ax.scatter(t.pct_var, t.math, s=28, color=[ORANGE if i == FOCUS else BLUE for i in t.index], edgecolor=SURF)
for i in t.index:
    if t.math[i] > 520 or t.pct_var[i] > 20 or t.pct_var[i] < 5.5 and i != 'KSV' or i == FOCUS:
        ax.annotate(t.name[i], (t.pct_var[i], t.math[i]), fontsize=7, xytext=(3, 2), textcoords='offset points', color=INK2)
ax.set(xlabel='% of math variance explained by ESCS (lower = more equitable)', ylabel='mean math',
       title='Performance vs equity (dashed = OECD average)')
ax.text(.01, .99, 'high performance, more equitable', transform=ax.transAxes, va='top', color=INK2, fontsize=8)
save(fig, '06_equity')
""")

md("""
## 7. What's associated with math *within* countries

For each index: the within-country change in math per 1 SD of the index (indices are standardised to OECD SD = 1),
estimated per country with BRR and averaged over OECD countries. These are **associations**, not causal effects,
and many indices are correlated with ESCS.
""")

code("""
assoc = {}
for v in IDX:
    if v not in stu or stu[v].notna().mean() < .2: continue
    r = with_oecd(estimate(stu, slope_of(v, PV('MATH'))))
    sd = stu[v].std()
    assoc[v] = dict(label=DD.query('dataset==\"student\" and variable==@v').label.iloc[0],
                    per_SD=r.loc['OECD','est'] * sd, se=r.loc['OECD','se'] * sd,
                    countries_positive=int((r.drop('OECD').est > 0).sum()), coverage=round(stu[v].notna().mean(), 2))
assoc = pd.DataFrame(assoc).T.astype({'per_SD': float, 'se': float}).sort_values('per_SD')
display(assoc.round(2))

fig, ax = plt.subplots(figsize=(8, .26 * len(assoc) + 1))
ax.barh(assoc.label.str.replace(r' \\(WLE\\)', '', regex=True).map(lambda x: x if len(x) <= 55 else x[:53] + '...'), assoc.per_SD,
        color=[BLUE if v > 0 else RED for v in assoc.per_SD], height=.7)
ax.errorbar(assoc.per_SD, range(len(assoc)), xerr=1.96 * assoc.se, fmt='none', ecolor=INK2, lw=.6)
ax.axvline(0, color=INK, lw=.8); ax.grid(axis='y', visible=False)
ax.set(xlabel='math score points per 1 SD of the index (OECD average of within-country slopes)',
       title='Within-country associations with math')
save(fig, '07_associations')
""")

md("## 8. Phones and distraction in math lessons")

code("""
dist = table({'pct_distracted_own':    pct_of(['ST273Q06JA'], lambda y: y <= 2),
              'pct_distracted_others': pct_of(['ST273Q07JA'], lambda y: y <= 2)}).join(scores[['math']])
dist = dist.sort_values('pct_distracted_own')
display(dist.round(1))
FREQ = {1: 'every lesson', 2: 'most lessons', 3: 'some lessons', 4: 'never/almost never'}
by_freq = pd.Series({lab: with_oecd(estimate(stu, mean_where(PV('MATH'), 'ST273Q06JA', float(k)))).loc['OECD', 'est']
                     for k, lab in FREQ.items()})
print('OECD average math by how often students are distracted by digital devices in math class:')
print(by_freq.round(1).to_string())
rank_chart(dist.rename(columns={'pct_distracted_own': 'v'}), 'v',
           '% of students distracted by digital devices in most or every math lesson', '%',
           '08_distraction', dist.loc['OECD', 'pct_distracted_own'])
""")

md("## 9. COVID-19 school closures")

code("""
cov19 = table({'closed_gt3m':  pct_of(['ST347Q01JA'], lambda y: y >= 4),
               'closed_any':   pct_of(['ST347Q01JA'], lambda y: y >= 2),
               'learnt_less':  pct_of(['ST350Q01JA'], lambda y: y == 1),
               'feel_prepared_next_time': pct_of(['ST356Q01JA'], lambda y: y >= 3),
               'SDLEFF': mean_of(['SDLEFF']), 'PROBSELF': mean_of(['PROBSELF'])}).join(scores[['math']])
display(cov19.sort_values('closed_gt3m').round(1))
t = cov19.drop('OECD').dropna(subset=['closed_gt3m'])
print('between-country correlation, % closed >3 months vs mean math:', round(t.closed_gt3m.corr(t.math), 2))
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(t.closed_gt3m, t.math, s=28, color=[ORANGE if i == FOCUS else BLUE for i in t.index], edgecolor=SURF)
for i in t.index:
    if t.closed_gt3m[i] > 70 or t.closed_gt3m[i] < 10 or t.math[i] > 520 or i == FOCUS:
        ax.annotate(t.name[i], (t.closed_gt3m[i], t.math[i]), fontsize=7, xytext=(3, 2), textcoords='offset points', color=INK2)
ax.set(xlabel='% of students whose school closed > 3 months for COVID-19', ylabel='mean math',
       title='Length of closures vs 2022 math (cross-country)')
save(fig, '09_covid')
""")

md("## 10. Schools: type, resources, and how much schools differ")

code("""
SCHV = ['CNTSCHID','SCHLTYPE','STRATIO','SCHSIZE','CLSIZE','EDUSHORT','STAFFSHORT','STUBEHA','TEACHBEHA',
        'SCHAUTO','DIGDVPOL','SCHSEL','ABGMATH']
sch = load('SCH_QQQ', [c for c in SCHV if c in cols('SCH_QQQ')])
s = stu.merge(sch, on='CNTSCHID', how='left')
print(f'students matched to a school record: {s.SCHLTYPE.notna().mean():.1%}')
s['PRIVATE'] = s.SCHLTYPE.map({1: 1.0, 2: 1.0, 3: 0.0})
schools = table({'pct_private':   pct_of(['PRIVATE'], lambda y: y == 1),
                 'private_minus_public': gap_of(PV('MATH'), 'PRIVATE', 1.0, 0.0),
                 'stratio':       mean_of(['STRATIO']),
                 'class_size':    mean_of(['CLSIZE']),
                 'edu_shortage':  mean_of(['EDUSHORT']),
                 'staff_shortage': mean_of(['STAFFSHORT'])}, df=s)

# descriptive between-school share of math variance (weighted, per PV, averaged), not a multilevel model
def between_share(d):
    out = []
    for p in PV('MATH'):
        w, y = d.W_FSTUWT.to_numpy(), d[p].to_numpy()
        mu = np.average(y, weights=w)
        sm = d.assign(wy=w * y).groupby('CNTSCHID').agg(wy=('wy', 'sum'), w=('W_FSTUWT', 'sum'))
        between = (sm.w * (sm.wy / sm.w - mu)**2).sum() / w.sum()
        out.append(between / np.average((y - mu)**2, weights=w))
    return 100 * np.mean(out)
schools['between_school_pct'] = s.groupby('CNT').apply(between_share)
schools.loc['OECD', 'between_school_pct'] = schools.loc[OECD, 'between_school_pct'].mean()
schools = schools.join(scores[['math']])
display(schools[['name','pct_private','private_minus_public','private_minus_public_se','stratio','class_size',
                 'edu_shortage','staff_shortage','between_school_pct','math']].sort_values('between_school_pct').round(1))
""")

code("""
# does the private-school advantage survive once you account for who attends? (OECD average)
def adj_gap(d, Wm):
    \"\"\"private-public math gap controlling for student ESCS and school mean ESCS, per weight x PV\"\"\"
    d = d.dropna(subset=['PRIVATE', 'ESCS'])
    if d.PRIVATE.nunique() < 2 or len(d) < 200: return np.full((81, 10), np.nan)
    Wm = d[W].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d.PRIVATE, d.ESCS, d.groupby('CNTSCHID').ESCS.transform('mean')])
    Y = d[PV('MATH')].to_numpy(float)
    return np.array([np.linalg.lstsq(X * np.sqrt(w)[:, None], Y * np.sqrt(w)[:, None], rcond=None)[0][1] for w in Wm.T])
adj = with_oecd(estimate(s, adj_gap))
print(f\"OECD average private - public math gap: raw {schools.loc['OECD','private_minus_public']:+.1f}, \"
      f\"after ESCS (student + school mean) {adj.loc['OECD','est']:+.1f} (SE {adj.loc['OECD','se']:.1f})\")
schools['private_gap_adjusted'] = adj.est
""")

md("## 11. Teachers (18 systems ran the teacher questionnaire)")

code("""
TV = ['SATJOB','SATTEACH','OCSTRESS','OVERLOAD','SEFFCM','SEFFREL','SEFFINS','TRUST','AUTONOMY','LEADSHIP',
      'DEVNEED','ICTOTL','CREATPED','NEGSYMPT']
tch = load('TCH_QQQ', ['CNT','SENWT'] + TV)
print(f'{len(tch):,} teachers in {tch.CNT.nunique()} systems')
tmean = tch.groupby('CNT').apply(lambda d: pd.Series({v: np.average(d[v].dropna(), weights=d.loc[d[v].notna(), 'SENWT'])
                                                     if d[v].notna().any() else np.nan for v in TV}))
tmean.index = tmean.index.map(NAME)
fig, ax = plt.subplots(figsize=(9, 6))
z = (tmean - tmean.mean()) / tmean.std()
im = ax.imshow(z.to_numpy(), cmap=div.reversed(), norm=TwoSlopeNorm(0, -2.5, 2.5), aspect='auto')
ax.set_xticks(range(len(TV)), TV, rotation=40, ha='right'); ax.set_yticks(range(len(z)), z.index); ax.grid(False)
for (r, c_), v in np.ndenumerate(tmean.to_numpy()):
    ax.text(c_, r, '-' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=6, color=INK)
plt.colorbar(im, ax=ax, shrink=.5, label='z-score across the 18 systems')
ax.set_title('Teacher indices (cell = mean WLE; blue = higher than others; for OCSTRESS/OVERLOAD/NEGSYMPT higher is worse; - = not fielded)', loc='left')
save(fig, '11_teachers')
labels = DD.query('dataset==\"teacher\" and variable in @TV').set_index('variable').label
display(labels.to_frame())
""")

md("""
## 12. Creative Thinking (new domain in 2022, scale 0-60)

The CRT file has no weights, so they are joined from the student file by student ID.
""")

code("""
crt = load('CRT_COG', ['CNT','CNTSTUID'] + PV('CRTH_NC')).merge(stu[['CNTSTUID'] + W + PV('MATH') + ['GENDER']],
                                                                  on='CNTSTUID', how='inner')
print(f'{len(crt):,} creative-thinking students with weights, {crt.CNT.nunique()} systems')
creat = table({'creative': mean_of(PV('CRTH_NC')),
               'gap_girls_minus_boys': gap_of(PV('CRTH_NC'), 'GENDER', 'F', 'M')}, df=crt).join(scores[['math']])
creat = creat.dropna(subset=['creative']).sort_values('creative', ascending=False)
display(creat.round(1))
t = creat.drop('OECD')
print('between-country correlation, creative thinking vs math:', round(t.creative.corr(t.math), 2))
fig, ax = plt.subplots(figsize=(7.5, 6))
b, a = np.polyfit(t.math, t.creative, 1)
xs = np.linspace(t.math.min(), t.math.max(), 2); ax.plot(xs, a + b * xs, color=INK2, lw=.8, ls=':')
ax.scatter(t.math, t.creative, s=28, color=[ORANGE if i == FOCUS else BLUE for i in t.index], edgecolor=SURF)
for i in t.index:
    if abs(t.creative[i] - (a + b * t.math[i])) > 4 or i == FOCUS or t.creative[i] > 38:
        ax.annotate(t.name[i], (t.math[i], t.creative[i]), fontsize=7, xytext=(3, 2), textcoords='offset points', color=INK2)
ax.set(xlabel='mean math', ylabel='mean creative thinking (0-60)',
       title='Creative thinking vs math (dotted = fitted line; far from line = over/under-performer)')
save(fig, '12_creative')
""")

md("## 13. Financial Literacy (separate sample, own weights and PVs)")

code("""
flt = load('FLT_QQQ', ['CNT','ST004D01T'] + W + PV('FLIT') + PV('MATH') + PV('READ'))
flt['GENDER'] = flt.ST004D01T.map({1: 'F', 2: 'M'})
fin = table({'finlit': mean_of(PV('FLIT')),
             'low_finlit': pct_of(PV('FLIT'), lambda y: y < 400.33),
             'top_finlit': pct_of(PV('FLIT'), lambda y: y >= 624.63),
             'gap_boys_minus_girls': gap_of(PV('FLIT'), 'GENDER', 'M', 'F'),
             'math_same_sample': mean_of(PV('MATH'))}, df=flt)
# OECD average here = only the OECD systems that took part
fin = fin.sort_values('finlit', ascending=False)
display(fin.round(1))
t = fin.drop('OECD')
print('correlation finlit vs math across systems:', round(t.finlit.corr(t.math_same_sample), 2))
""")

md("## 14. Cognitive items: what students got right and how long it took")

code("""
cog_cols = cols('STU_COG')
lab = DD[DD.dataset == 'cognitive'].set_index('variable').label
scored = [c for c in cog_cols if isinstance(lab.get(c), str) and 'Scored Response' in lab[c]]
timing = [c for c in cog_cols if isinstance(lab.get(c), str) and 'Total Timing' in lab[c]]
print(len(scored), 'scored items,', len(timing), 'timing vars')

n = pd.Series(0.0, index=scored); tot = pd.Series(0.0, index=scored); mx = pd.Series(0.0, index=scored)
tsamp = []
for i, batch in enumerate(batches('STU_COG', scored + timing)):
    b = batch.to_pandas()
    S = b[scored]
    n += S.notna().sum(); tot += S.sum(); mx = np.maximum(mx, S.max().fillna(0))
    if i % 3 == 0: tsamp.append(b[timing].sample(frac=.2, random_state=i))
items = pd.DataFrame({'administered': n, 'mean_score': tot / n, 'max_score': mx})
items['pct_correct'] = 100 * items.mean_score / items.max_score
items['domain'] = items.index.str[1].map({'M': 'math', 'R': 'reading', 'S': 'science'}).fillna('other')
items['label'] = lab.reindex(items.index).str.replace(r' \\(Scored Response\\)', '', regex=True)
T = pd.concat(tsamp)
# timing unit check: SPSS labels do not state it, so infer from the median of all items
med = T.median()
unit = 1000 if med.median() > 1000 else 1
items['median_seconds'] = (med / unit).rename(lambda c: c[:-2] + 'S').reindex(items.index)
print('timing unit inferred:', 'milliseconds' if unit == 1000 else 'seconds', '| sampled students:', len(T))
display(items.groupby('domain').agg(items=('label','size'), median_pct_correct=('pct_correct','median'),
                                    median_seconds=('median_seconds','median')).round(1))
print('\\nhardest items:'); display(items.nsmallest(8, 'pct_correct')[['domain','label','pct_correct','median_seconds','administered']].round(1))
print('easiest items:'); display(items.nlargest(8, 'pct_correct')[['domain','label','pct_correct','median_seconds','administered']].round(1))
items.to_csv(OUT/'item_statistics.csv')

fig, ax = plt.subplots(figsize=(8, 4.5))
for dom, col in [('math', BLUE), ('reading', ORANGE), ('science', AQUA)]:
    d = items[items.domain == dom]
    ax.scatter(d.median_seconds, d.pct_correct, s=12, color=col, alpha=.75, label=f'{dom} ({len(d)})', edgecolor='none')
ax.set_xscale('log'); ax.legend(frameon=False)
ax.set(xlabel='median time on item, seconds (log)', ylabel='% of max score (unweighted, pooled)',
       title='Item difficulty vs time spent')
save(fig, '14_items')
""")

code("""
# test effort (1-10 scale, asked right after the test) vs performance
eff = table({'effort': mean_of(['EFFORT1']), 'effort_if_counted': mean_of(['EFFORT2'])}).join(scores[['math']])
t = eff.drop('OECD')
print('between-country correlation effort vs math:', round(t.effort.corr(t.math), 2))
print('within-country: OECD avg math points per effort point:',
      round(with_oecd(estimate(stu, slope_of('EFFORT1', PV('MATH')))).loc['OECD', 'est'], 1))
display(eff.sort_values('effort').round(2).head(10))
""")

md("## 15. Questionnaire timing file")

code("""
tim = load('STU_TIM', ['CNT','CNTSTUID'] + [c for c in cols('STU_TIM') if c.endswith('_TT')])
tt = [c for c in tim if c.endswith('_TT')]
unit = 1000 if tim[tt].median().median() > 1000 else 1
tim['total_min'] = tim[tt].sum(axis=1, min_count=1) / unit / 60
print('timing unit inferred:', 'ms' if unit == 1000 else 's')
tq = tim.groupby('CNT').total_min.median().rename('median_questionnaire_minutes').to_frame()
tq['name'] = tq.index.map(NAME)
display(tq.sort_values('median_questionnaire_minutes').round(1).T)
slow = (tim[tt].median() / unit).sort_values(ascending=False).head(10).rename('median_seconds').to_frame()
slow['question'] = DD[DD.dataset == 'timing'].set_index('variable').label.reindex(slow.index)
print('slowest questionnaire screens (median seconds):'); display(slow.round(1))
""")

md("## 16. ESCS over time (trend file, 2012-2022)")

code("""
et = pd.read_parquet(PQ/'escs_trend')
et['year'] = et.cycle.map({5: 2012, 6: 2015, 7: 2018})
e22 = stu[['CNT','OECD','ESCS']].rename(columns={'ESCS': 'escs_trend', 'OECD': 'oecd', 'CNT': 'cnt'}).assign(year=2022)
allc = pd.concat([et[['cnt','oecd','year','escs_trend']], e22])
# the trend file carries no weights, so these are unweighted sample means (fine for direction, not for exact levels)
tr = allc.groupby(['cnt','year']).escs_trend.mean().unstack()
tr = tr[tr.notna().sum(axis=1) == 4]
tr['change_2012_2022'] = tr[2022] - tr[2012]
tr.insert(0, 'name', tr.index.map(NAME))
display(tr.sort_values('change_2012_2022').round(2))
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in tr.index:
    ax.plot([2012, 2015, 2018, 2022], tr.loc[c, [2012, 2015, 2018, 2022]].astype(float),
            color=ORANGE if c == FOCUS else '#b7d3f6', lw=2 if c == FOCUS else .8, zorder=3 if c == FOCUS else 1)
oe = tr.loc[tr.index.isin(OECD), [2012, 2015, 2018, 2022]].mean()
ax.plot(oe.index, oe.values, color=BLUE, lw=2.2, label='OECD mean'); ax.legend(frameon=False)
ax.set(title='ESCS by cycle (unweighted; each line a system)', ylabel='ESCS (trend scale)'); ax.set_xticks([2012, 2015, 2018, 2022])
save(fig, '16_escs_trend')
""")

md("## 17. Country profile: everything above for `FOCUS`")

code("""
panel = (scores[['name','math','read','sci']]
         .join(prof[['low_MATH','top_MATH','low_READ','low_SCIE']])
         .join(gg[['gap_MATH','gap_READ']])
         .join(eq[['escs_mean','gradient','pct_var']])
         .join(dist[['pct_distracted_own']])
         .join(cov19[['closed_gt3m','learnt_less']])
         .join(schools[['pct_private','private_minus_public','between_school_pct']])
         .join(creat[['creative']]).join(fin[['finlit']]).join(eff[['effort']]))
panel.to_csv(OUT/'country_summary.csv')
countries = panel.drop('OECD')
rank = countries.drop(columns='name').rank(ascending=False, method='min').astype('Int64')
prof_f = pd.DataFrame({FOCUS: panel.loc[FOCUS].drop('name'), 'OECD avg': panel.loc['OECD'].drop('name'),
                       'rank (1 = highest)': rank.loc[FOCUS], 'systems with data': countries.drop(columns='name').count(),
                       'min': countries.drop(columns='name').min(), 'max': countries.drop(columns='name').max()})
print(NAME.get(FOCUS, FOCUS)); display(prof_f.round(1))
print(f'country_summary.csv written: {panel.shape}')
""")

md("## 18. Key findings (computed from the cells above)")

code("""
o, c = scores.loc['OECD'], scores.drop('OECD')
top5 = ', '.join(f'{c.name[i]} {c.math[i]:.0f}' for i in c.math.nlargest(5).index)
bot3 = ', '.join(f'{c.name[i]} {c.math[i]:.0f}' for i in c.math.nsmallest(3).index)
lp = prof.drop('OECD')
F = [
 f\"Coverage: {len(stu):,} students, {stu.CNT.nunique()} systems, representing ~{stu.W_FSTUWT.sum()/1e6:.1f}M 15-year-olds.\",
 f\"OECD averages: math {o.math:.0f}, reading {o.read:.0f}, science {o.sci:.0f} (reproduces the published figures).\",
 f\"Top in math: {top5}. Bottom: {bot3}. Spread top-to-bottom ~{c.math.max()-c.math.min():.0f} points.\",
 f\"{prof.loc['OECD','low_MATH']:.0f}% of students in the average OECD country are below math Level 2; \"
 f\"in {int((lp.low_MATH > 50).sum())} systems it is over half.\",
 f\"Gender: girls ahead in reading by {-gg.loc['OECD','gap_READ']:.0f} points on OECD average; boys ahead in math by \"
 f\"{gg.loc['OECD','gap_MATH']:.0f}.\",
 f\"Equity: ESCS explains {eq.loc['OECD','pct_var']:.0f}% of math variance on OECD average; top vs bottom ESCS quarter gap \"
 f\"{eq.loc['OECD','q4_minus_q1']:.0f} points.\",
 f\"Strongest positive within-country association: {assoc.label.iloc[-1]} ({assoc.per_SD.iloc[-1]:+.0f}/SD); \"
 f\"strongest negative: {assoc.label.iloc[0]} ({assoc.per_SD.iloc[0]:+.0f}/SD).\",
 f\"Phones: {dist.loc['OECD','pct_distracted_own']:.0f}% of OECD students are distracted by digital devices in most/every math lesson; \"
 f\"those distracted every lesson score {by_freq['never/almost never']-by_freq['every lesson']:.0f} points below those who never are.\",
 f\"COVID: {cov19.loc['OECD','closed_gt3m']:.0f}% of OECD students had schools closed >3 months; \"
 f\"{cov19.loc['OECD','learnt_less']:.0f}% say they learnt less.\",
 f\"Schools: on OECD average {schools.loc['OECD','between_school_pct']:.0f}% of math variance lies between schools; \"
 f\"private-public gap {schools.loc['OECD','private_minus_public']:+.0f} raw vs {schools.loc['OECD','private_gap_adjusted']:+.0f} after ESCS.\",
 f\"Creative thinking tracks math closely across countries (r = {creat.drop('OECD').creative.corr(creat.drop('OECD').math):.2f}).\",
]
for f in F: print('-', f)
(OUT/'key_findings.md').write_text('# PISA 2022: key findings\\n\\n' + '\\n'.join('- ' + f for f in F) + '\\n')
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (ipykernel)', 'language': 'python'}
nbf.write(nb, __file__.replace('_build_02.py', '02_explore.ipynb'))
