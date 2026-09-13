"""Make the standalone dimensional-runtime panel from audited JSON/CSV only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
try:
    from . import evidence_report as report
    from .complete_tables import compact
except ImportError:
    import evidence_report as report
    from complete_tables import compact


def build(root, prefix):
    root, prefix = Path(root).resolve(), Path(prefix)
    plan = report._json(root/'plan.json')
    audit = report._json(root/'audit.json')
    summary = report._json(root/'summary.json')
    digest = hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
    if not (plan.get('runtime_study') and audit.get('audit_passed') and
            audit.get('all_planned_tasks_complete') and audit.get('plan_sha256') == digest and
            summary.get('plan_sha256') == digest):
        raise ValueError('Requires a matching complete passing runtime audit')
    rows = report._read_rows(root/'per_replication.csv')
    if Counter(row['status'] for row in rows) != Counter(summary['status_counts']):
        raise ValueError('Runtime status counts disagree with audited summary')
    _, timing = report.attach_workflow_timing(root, rows, plan, digest)
    statistics = report.method_statistics(rows)
    paired, _, _ = report.paired_evidence(rows)
    report._style()
    figure = report.plot_runtime(statistics, False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(str(prefix)+'.'+extension, bbox_inches='tight', pad_inches=.12, dpi=200)
    report.plt.close(figure)
    payload = dict(plan_sha256=digest, methods=statistics, paired=paired, timing=timing)
    Path(str(prefix)+'.json').write_text(json.dumps(payload, indent=2)+'\n')
    dimensions = sorted({(row['p'],row['q']) for row in statistics})
    newline = ' '+chr(92)*2
    table = [r'\begin{tabular}{l'+'r'*len(dimensions)+'}', r'\toprule',
        'Method & '+' & '.join(f'${p}\\times{q}$' for p,q in dimensions)+newline, r'\midrule']
    for method in ('v2','initializer_only','target_ridge_rrr','source_subspace_ridge_rrr'):
        cells = []
        for p,q in dimensions:
            row = next(row for row in statistics if row['method']==method and row['p']==p and row['q']==q)
            value = row['workflow_tuning_seconds']
            cells.append(f"${compact(value['mean'])}$ (${compact(value['mcse'])}$)" if value['mean'] is not None and value['mcse'] is not None else '--')
        table.append(report._tex(report.LABELS[method])+' & '+' & '.join(cells)+newline)
    table += [r'\bottomrule',r'\end{tabular}']
    Path(str(prefix)+'-table.tex').write_text('\n'.join(table)+'\n')
    return dict(n_tasks=len(plan['tasks']), n_dimensions=len(dimensions), prefix=str(prefix))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('prefix')
    args=parser.parse_args(); print(json.dumps(build(args.root,args.prefix),sort_keys=True))
