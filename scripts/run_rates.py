"""Example: python scripts/run_rates.py --as-of 2026-10-31"""
import argparse
from erp import RatesData,RatesPremiaEngine,export_snapshot
p=argparse.ArgumentParser()
p.add_argument('--as-of',default=None)
p.add_argument('--cache',default='data/cache')
p.add_argument('--funding-source',default='mirror',choices=['mirror','fred','auto','local'])
p.add_argument('--fedfunds-csv',default=None,help='Original FEDFUNDS.csv or mirror CSV; avoids all funding downloads')
p.add_argument('--output',default='outputs/rates_v02')
a=p.parse_args()
data=RatesData.from_fred(cache_dir=a.cache, funding_source=a.funding_source, local_fedfunds_path=a.fedfunds_csv)
engine=RatesPremiaEngine(data)
s=engine.fit(as_of=a.as_of)
folder=export_snapshot(s,a.output)
engine.export_auxiliary(folder)
print('Snapshot:',folder)
print('Funded factor premia (mean monthly):',s.factor_premia)
print('Exact cross-factor purity diagnostics:',engine.latest_mimic_diagnostics)
