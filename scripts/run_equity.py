"""Example: python scripts/run_equity.py --model FF6_BAB_QMJ --universe POOLED_130 --as-of 2026-10-31"""
import argparse
from pathlib import Path
from erp import EquityData, EquityPremiaEngine, export_snapshot

p=argparse.ArgumentParser()
p.add_argument('--model',default='FF6_BAB_QMJ')
p.add_argument('--universe',default='POOLED_130')
p.add_argument('--as-of',default=None)
p.add_argument('--no-aqr',action='store_true')
p.add_argument('--cache',default='data/cache')
p.add_argument('--output',default='outputs/equity_v02')
p.add_argument('--beta-window',type=int,default=60)
p.add_argument('--bab-path',default=None)
p.add_argument('--qmj-path',default=None)
a=p.parse_args()
paths={k:v for k,v in {'BAB':a.bab_path,'QMJ':a.qmj_path}.items() if v}
print('Loading published French and AQR sources (using local cache where available)...')
data=EquityData.from_public_sources(cache_dir=a.cache,aqr=not a.no_aqr,aqr_paths=paths)
engine=EquityPremiaEngine(data)
print('Available test universes:',list(data.universes),'POOLED_130')
snapshot=engine.fit(model=a.model,universe=a.universe,as_of=a.as_of,beta_window=a.beta_window)
folder=export_snapshot(snapshot,a.output)
print('Snapshot:',folder)
print('Sample:',snapshot.metadata['sample_start'],'->',snapshot.metadata['sample_end'])
print('Mean absolute alpha (annual):',round(12*snapshot.alpha.abs().mean(),5))
print('FMB premia and identification:')
print(snapshot.fmb_diagnostics.round(4).to_string())
