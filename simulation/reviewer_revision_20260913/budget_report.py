"""Plot independently audited budget evidence; no fitting or array archives."""
import argparse
import json
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'smart-budget-report'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CASES={'reference_exact_n80':'Exact structure, source noise 0.01',
       'containment_both30_n80':'30-degree two-sided leakage',
       'fitted_source_n0100_n80':'Fitted source, total n0 = 100',
       'sparse_stress_exact_n40':'Sparse stress, n = 40 (full caps)'}

def moments(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
                mcse=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None)

def report(root,prefix):
    root,prefix=Path(root),Path(prefix)
    audit=json.loads((root/'budget-audit.json').read_text())
    if not(audit['audit_passed'] and audit['all_planned_tasks_complete'] and audit['full_predeclared_study']):
        raise ValueError('Requires a complete independently audited budget study')
    rows=json.loads((root/'budget-audited-per-replication.json').read_text())
    prefix.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    summaries=[];contrasts=[]
    for ax,(case,title) in zip(axes.flat,CASES.items()):
        for budget in (100,500,1000,2000):
            selected=[r for r in rows if r['case_id']==case and r['budget']==budget and r['method']=='v2_reduced_grid']
            raw=[r for r in rows if r['case_id']==case and r['budget']==budget and r['method']=='raw_initializer_selected']
            good=[r for r in selected if r['status']=='success']
            record=dict(case_id=case,budget=budget,n_planned=len(selected),n_success=len(good),
                selected_risk=moments([r['metrics']['population_prediction_excess'] for r in good]),
                terminal_risk=moments([r['terminal_metrics']['population_prediction_excess'] for r in good]),
                initializer_risk=moments([r['metrics']['population_prediction_excess'] for r in raw if r['status']=='success']),
                library_seconds=moments([r['fitting_call_seconds'] for r in selected]),
                selected_iterations=[r['selected_iteration'] for r in good],
                terminal_stops=sum(r.get('optimization_converged',False) for r in good),
                selected_stops=sum(r.get('selected_output_converged',False) for r in good))
            summaries.append(record)
        group=[s for s in summaries if s['case_id']==case]
        for key,label,color,marker in [('selected_risk','Validation-selected','#0072B2','o'),
                                      ('terminal_risk','Terminal iterate of selected candidate','#D55E00','s'),
                                      ('initializer_risk','Separately selected raw initializer','#009E73','^')]:
            x=[s['budget'] for s in group];y=[s[key]['mean'] for s in group]
            error=[1.96*(s[key]['mcse'] or 0) for s in group]
            ax.errorbar(x,y,yerr=error,color=color,marker=marker,capsize=2,lw=1.2,label=label)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([100,500,1000,2000],['100','500','1000','2000'])
        ax.set_title(title);ax.set_xlabel('Maximum accepted updates');ax.set_ylabel('Population excess prediction risk');ax.grid(axis='y',alpha=.25)
        by_budget={b:{r['seed']:r for r in rows if r['case_id']==case and r['budget']==b and r['method']=='v2_reduced_grid' and r['status']=='success'} for b in (500,2000)}
        seeds=sorted(set(by_budget[500])&set(by_budget[2000]))
        differences=[by_budget[2000][s]['metrics']['population_prediction_excess']-by_budget[500][s]['metrics']['population_prediction_excess'] for s in seeds]
        contrasts.append(dict(case_id=case,n_planned=5,paired_2000_minus_500=moments(differences),
            unchanged_selected_risk=sum(d==0 for d in differences)))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.047),fontsize=8)
    fig.suptitle('Iteration-budget sensitivity on a fixed six-candidate library',x=.06,ha='left',fontsize=12)
    fig.text(.06,.015,'Five paired seeds per scenario; means and pointwise 95% Monte Carlo intervals. Every budget uses fresh fits. All caps are full.',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.90,hspace=.46,wspace=.32)
    for ext in ('pdf','png'):fig.savefig(str(prefix)+'.'+ext,dpi=200,bbox_inches='tight')
    plt.close(fig)
    payload=dict(audit_plan_sha256=audit['plan_sha256'],summaries=summaries,contrasts=contrasts,
                 scope='Reduced fixed library; five seeds; no claim of global convergence or full-grid retuning')
    Path(str(prefix)+'.json').write_text(json.dumps(payload,indent=2)+'\n')
    end=' '+chr(92)*2
    table=['\\begin{tabular}{llrrrr}','\\toprule','Scenario & Budget & Valid & Selected risk & Terminal risk & Library seconds'+end,'\\midrule']
    names=['Exact structure','30-degree leakage','Fitted source','Sparse stress']
    for case,name in zip(CASES,names):
        for s in [s for s in summaries if s['case_id']==case]:
            fmt=lambda d:f"{d['mean']:.4g} ({d['mcse']:.2g})" if d['mean'] is not None and d['mcse'] is not None else '--'
            table.append(f"{name} & {s['budget']} & {s['n_success']}/{s['n_planned']} & {fmt(s['selected_risk'])} & {fmt(s['terminal_risk'])} & {fmt(s['library_seconds'])}"+end)
        table.append('\\midrule')
    table[-1]='\\bottomrule';table.append('\\end{tabular}')
    Path(str(prefix)+'-table.tex').write_text('\n'.join(table)+'\n')
    print(json.dumps(contrasts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('prefix');a=p.parse_args();report(a.root,a.prefix)
