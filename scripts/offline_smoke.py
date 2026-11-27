"""No vendor access: offline funded-factor and equity pricing integration smoke test."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from erp import EquityData,EquityPremiaEngine,RatesData,RatesPremiaEngine,export_snapshot

rng=np.random.default_rng(2026)
idx=pd.date_range('1990-01-31',periods=240,freq='ME')
factors=pd.DataFrame(rng.normal(.003,.02,(240,8)),index=idx,columns=['MKT','SMB','HML','RMW','CMA','MOM','BAB','QMJ'])
ff3=factors[['MKT','SMB','HML']].copy();ff3['RF']=.001
ff5=factors[['MKT','SMB','HML','RMW','CMA']].copy();ff5['RF']=.001
n=30;B=rng.normal(0,.7,(n,8));B[:,0]+=1
returns=pd.DataFrame(factors.to_numpy()@B.T+.001+rng.normal(0,.01,(240,n)),index=idx,columns=[f'P{i:02d}' for i in range(n)])
data=EquityData(ff3,ff5,factors['MOM'],{'INDUSTRY_30':returns},factors[['BAB','QMJ']])
se=EquityPremiaEngine(data).fit('FF6_BAB_QMJ','INDUSTRY_30',as_of='2007-12-31')
export_snapshot(se,'outputs/offline_equity')
print('Equity synthetic example',se.model,len(se.fmb_lambdas),se.asset_excess_returns.shape)

load=np.column_stack([np.ones(10),np.linspace(-1,1,10),1-2*np.linspace(-1,1,10)**2])
curve=.03+np.linspace(0,.007,10)+np.cumsum(rng.normal(0,.0008,(240,3))@load.T,axis=0)
zd=pd.DataFrame(curve,index=idx,columns=[f'{j}Y' for j in range(1,11)])
re=RatesPremiaEngine(RatesData(zd,pd.Series(.025,index=idx)))
sr=re.fit(pca_train_end='2002-12-31')
export_snapshot(sr,'outputs/offline_rates');re.export_auxiliary('outputs/offline_rates')
print('Rates synthetic example',sr.model, sr.factors.shape,'max cross loading',sr.metadata['mimic_max_cross_violation'])
