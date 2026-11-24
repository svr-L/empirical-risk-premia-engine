"""Run Equity residual discovery from published French + AQR data."""
import argparse
from pathlib import Path
from erp import EquityData,analyze_equity_residuals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='data/cache')
    ap.add_argument('--as-of',default='2026-10-31')
    ap.add_argument('--train-end',default='2005-12-31')
    ap.add_argument('--pcs',type=int,default=5)
    ap.add_argument('--out',default='outputs/equity_residual_v010')
    ap.add_argument('--aqr-bab',default=None)
    ap.add_argument('--aqr-qmj',default=None)
    ap.add_argument('--economic-shocks',default=None)
    ap.add_argument('--lagged-states',default=None)
    ap.add_argument('--state-lag',type=int,default=2)
    args=ap.parse_args()
    paths={k:v for k,v in [('BAB',args.aqr_bab),('QMJ',args.aqr_qmj)] if v}
    data=EquityData.from_public_sources(cache_dir=args.cache,aqr=True,aqr_paths=paths)
    def read(path):
        if path is None:return None
        return __import__('pandas').read_csv(path,index_col=0,parse_dates=True)
    report=analyze_equity_residuals(data,train_end=args.train_end,as_of=args.as_of,
           n_pcs=args.pcs,macro_shocks=read(args.economic_shocks),
           state_predictors=read(args.lagged_states),state_lag_months=args.state_lag)
    out=report.export(Path(args.out));print('Exported:',out)
    print(report.fmp_premia_tests.to_string())
    print(report.holdout_metrics.to_string())
    print(report.sdf_gmm_comparison.to_string())
if __name__=='__main__':main()
