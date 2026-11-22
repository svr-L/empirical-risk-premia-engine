"""Run the integrated v0.8 Rates research grid under ERP v0.2.4."""
import argparse
from pathlib import Path
from erp import RatesData, RatesPremiaEngine, RatesEmpiricalBondPremiaEngine, load_v08_macro_csv, export_snapshot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--as-of',default='2026-10-31')
    p.add_argument('--cache',default='data/cache')
    p.add_argument('--output',default='outputs/rates_v024')
    p.add_argument('--curve-source',default='board',choices=['auto','board','fred'])
    p.add_argument('--curve-csv',default=None)
    p.add_argument('--funding-source',default='mirror',choices=['mirror','fred','auto','local'])
    p.add_argument('--funding-csv',default=None)
    p.add_argument('--macro-csv',default=None,help='Date,CPI,INDPRO,BREAKEVEN10Y; original FRED levels and T10YIE in percent')
    p.add_argument('--require-full-grid',action='store_true')
    p.add_argument('--min-train',type=int,default=60)
    p.add_argument('--horizons',type=int,nargs='+',default=[3,6,12])
    args=p.parse_args()
    data=RatesData.from_fred(cache_dir=args.cache,curve_source=args.curve_source,
        local_curve_path=args.curve_csv,local_fedfunds_path=args.funding_csv,
        funding_source=args.funding_source)
    rates=RatesPremiaEngine(data)
    snapshot=rates.fit(as_of=args.as_of,pca_train_end='2002-12-31')
    raw=load_v08_macro_csv(args.macro_csv) if args.macro_csv else None
    result=RatesEmpiricalBondPremiaEngine(rates).fit(snapshot,macro_raw=raw,
        horizons=tuple(args.horizons),min_train=args.min_train,require_full_grid=args.require_full_grid)
    out=Path(args.output)
    snap=export_snapshot(snapshot,out/'core_factor_snapshot')
    rates.export_auxiliary(snap)
    result.export(out/'empirical_bond_premia')
    print('Version 0.2.4 | as_of:',snapshot.as_of)
    print('Models run:',result.metadata['models_run'])
    print('Skipped missing macro:',result.metadata['models_skipped_for_missing_macro'])
    print('Forecast panel sizes:',{str(k):len(v) for k,v in result.forecasts.items()})
    print('Results in:',out)


if __name__=='__main__': main()
