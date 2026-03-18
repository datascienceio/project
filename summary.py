from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cfg:
    root: Path
    seed: int = 7

    @property
    def raw(self) -> Path:
        return self.root / "01_raw"

    @property
    def inter(self) -> Path:
        return self.root / "02_intermediate"

    @property
    def primary(self) -> Path:
        return self.root / "03_primary"

    @property
    def feat(self) -> Path:
        return self.root / "04_feature"

    @property
    def mi(self) -> Path:
        return self.root / "05_model_input"

    @property
    def models(self) -> Path:
        return self.root / "06_models"

    @property
    def mo(self) -> Path:
        return self.root / "07_model_output"

    @property
    def rep(self) -> Path:
        return self.root / "08_reporting"


def _pct(x: str | float | int | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s=str(x).strip()
    if not s:
        return None
    return float(s.replace("%",""))


def _money(x: str | float | int | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s=str(x).strip()
    if not s:
        return None
    return float(s.replace("$","").replace(",",""))


def _b(x) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s=str(x).strip().lower()
    if s in {"t","true","1","yes"}:
        return 1
    if s in {"f","false","0","no"}:
        return 0
    return None


class Capstone:
    def __init__(self, root: str | Path, seed: int = 7):
        self.c=Cfg(Path(root),seed)
        self.rng=np.random.default_rng(seed)
        self.companies=None
        self.shuttles=None
        self.reviews=None
        self.df=None
        self.X=None
        self.y=None
        self.ts=None
        self.ts_xy=None
        self.res={}

    def load(self):
        c=self.c
        self.companies=pd.read_csv(c.raw/"companies.csv")
        self.reviews=pd.read_csv(c.raw/"reviews.csv")
        self.shuttles=pd.read_excel(c.raw/"shuttles.xlsx")
        return self

    def clean(self):
        co=self.companies.rename(columns={"id":"company_id"}).copy()
        co["company_rating"]=co["company_rating"].map(_pct).astype("float32")
        co["iata_approved"]=co["iata_approved"].map(_b).astype("float32")
        co["total_fleet_count"]=pd.to_numeric(co["total_fleet_count"],errors="coerce").astype("float32")

        sh=self.shuttles.rename(columns={"id":"shuttle_id"}).copy()
        sh["d_check_complete"]=sh["d_check_complete"].map(_b).astype("float32")
        sh["moon_clearance_complete"]=sh["moon_clearance_complete"].map(_b).astype("float32")
        sh["price"]=sh["price"].map(_money).astype("float32")
        for k in ["engines","passenger_capacity","crew"]:
            sh[k]=pd.to_numeric(sh[k],errors="coerce").astype("float32")

        rv=self.reviews.copy()
        for k in rv.columns:
            if k!="shuttle_id":
                rv[k]=pd.to_numeric(rv[k],errors="coerce")
        rv["reviews_per_month"]=rv["reviews_per_month"].astype("float32")
        rv["number_of_reviews"]=rv["number_of_reviews"].fillna(0).astype("float32")

        df=rv.merge(sh,on="shuttle_id",how="left").merge(co,on="company_id",how="left",suffixes=("","_c"))
        df["months_active"]=np.where(
            (df["reviews_per_month"]>0) & df["reviews_per_month"].notna(),
            (df["number_of_reviews"]/df["reviews_per_month"]).clip(0,1e4),
            np.nan,
        ).astype("float32")
        df["rating"]=df["review_scores_rating"].astype("float32")
        self.df=df
        self.companies,self.shuttles,self.reviews=co,sh,rv
        return self

    def make_features(self, min_rows: int = 5000):
        df=self.df.copy()
        df=df[df["rating"].notna()].copy()
        if len(df)<min_rows:
            raise ValueError(f"too_few_rows:{len(df)}")

        num=[
            "price","engines","passenger_capacity","crew",
            "d_check_complete","moon_clearance_complete",
            "company_rating","total_fleet_count","iata_approved",
            "number_of_reviews","reviews_per_month","months_active",
            "review_scores_comfort","review_scores_amenities","review_scores_trip",
            "review_scores_crew","review_scores_location","review_scores_price",
        ]
        cat=[
            "shuttle_location","shuttle_type","engine_type","engine_vendor","cancellation_policy","company_location",
        ]
        keep=["shuttle_id","company_id","rating"]+num+cat
        z=df[keep].copy()
        for k in num:
            z[k]=pd.to_numeric(z[k],errors="coerce")
        for k in cat:
            z[k]=z[k].astype("string").fillna("∅")
        z=z.dropna(subset=["rating"])
        Xn=z[num].fillna(z[num].median(numeric_only=True))
        Xc=pd.get_dummies(z[cat],drop_first=False,dtype="int8")
        X=pd.concat([Xn.astype("float32"),Xc],axis=1)
        y=z["rating"].astype("float32")
        self.X,self.y=X,y
        self.res["n"]=int(len(z))
        self.res["p"]=int(X.shape[1])
        return self

    def make_ts(self, bins: int = 60):
        df=self.df.copy()
        d=df[df["rating"].notna() & df["months_active"].notna()].copy()
        d=d[d["months_active"].between(0,365)].copy()
        d["t"]=pd.cut(d["months_active"],bins=bins,labels=False,include_lowest=True)
        ts=d.groupby("t",as_index=False).agg(
            y=("rating","mean"),
            n=("rating","size"),
            w=("number_of_reviews","sum"),
        )
        ts=ts.sort_values("t").reset_index(drop=True)
        for k in ["y","n","w"]:
            ts[k]=ts[k].astype("float32")
        for lag in [1,2,3,6]:
            ts[f"y_l{lag}"]=ts["y"].shift(lag)
            ts[f"w_l{lag}"]=ts["w"].shift(lag)
        ts=ts.dropna().reset_index(drop=True)
        feats=[c for c in ts.columns if c.startswith(("y_l","w_l"))]
        X=ts[feats].astype("float32")
        y=ts["y"].astype("float32")
        self.ts,self.ts_xy=ts,(X,y,feats)
        self.res["ts_n"]=int(len(ts))
        return self

    def fit_xgb(self):
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X,y=self.X,self.y
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=self.c.seed)
        m=XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=self.c.seed,
            tree_method="hist",
        )
        m.fit(Xtr,ytr)
        p=m.predict(Xte)
        self.res["xgb"]={
            "mae":float(mean_absolute_error(yte,p)),
            "rmse":float(np.sqrt(mean_squared_error(yte,p))),
            "r2":float(r2_score(yte,p)),
        }
        self.m_xgb=m
        return self

    def fit_xgb_ts(self):
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        X,y,_=self.ts_xy
        n=len(X)
        k=max(8,int(n*0.2))
        Xtr,ytr=X.iloc[:-k],y.iloc[:-k]
        Xte,yte=X.iloc[-k:],y.iloc[-k:]
        m=XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.c.seed,
            tree_method="hist",
        )
        m.fit(Xtr,ytr)
        p=m.predict(Xte)
        self.res["xgb_ts"]={
            "mae":float(mean_absolute_error(yte,p)),
            "rmse":float(np.sqrt(mean_squared_error(yte,p))),
            "k":int(k),
        }
        self.m_xgb_ts=m
        self.ts_pred=pd.DataFrame({"t":self.ts["t"].iloc[-k:].to_numpy(),"y":yte.to_numpy(),"yhat":p})
        return self

    def fit_tf_ts(self, epochs: int = 200):
        try:
            import tensorflow as tf
        except Exception as e:
            self.res["tf_ts"]=str(e)
            return self
        X,y,_=self.ts_xy
        n=len(X)
        k=max(8,int(n*0.2))
        Xtr,ytr=X.iloc[:-k].to_numpy(),y.iloc[:-k].to_numpy()
        Xte,yte=X.iloc[-k:].to_numpy(),y.iloc[-k:].to_numpy()
        z_mu,z_sd=Xtr.mean(0,keepdims=True),Xtr.std(0,keepdims=True)+1e-6
        Xtr=(Xtr-z_mu)/z_sd
        Xte=(Xte-z_mu)/z_sd
        m=tf.keras.Sequential([
            tf.keras.layers.Input(shape=(Xtr.shape[1],)),
            tf.keras.layers.Dense(32,activation="swish"),
            tf.keras.layers.Dense(16,activation="swish"),
            tf.keras.layers.Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(3e-3),loss="mae")
        m.fit(Xtr,ytr,epochs=epochs,verbose=0)
        p=m.predict(Xte,verbose=0).reshape(-1)
        mae=float(np.mean(np.abs(yte-p)))
        rmse=float(np.sqrt(np.mean((yte-p)**2)))
        self.res["tf_ts"]={"mae":mae,"rmse":rmse,"k":int(k)}
        self.m_tf_ts=m
        self.tf_norm={"mu":z_mu.reshape(-1).tolist(),"sd":z_sd.reshape(-1).tolist()}
        return self

    def fit_torch_ts(self, epochs: int = 500):
        try:
            import torch
            from torch import nn
        except Exception as e:
            self.res["torch_ts"]=str(e)
            return self
        X,y,_=self.ts_xy
        n=len(X)
        k=max(8,int(n*0.2))
        Xtr,ytr=X.iloc[:-k].to_numpy(),y.iloc[:-k].to_numpy()
        Xte,yte=X.iloc[-k:].to_numpy(),y.iloc[-k:].to_numpy()
        z_mu,z_sd=Xtr.mean(0,keepdims=True),Xtr.std(0,keepdims=True)+1e-6
        Xtr=(Xtr-z_mu)/z_sd
        Xte=(Xte-z_mu)/z_sd
        dev="cpu"
        Xt=torch.tensor(Xtr,dtype=torch.float32,device=dev)
        yt=torch.tensor(ytr,dtype=torch.float32,device=dev).view(-1,1)
        Xv=torch.tensor(Xte,dtype=torch.float32,device=dev)
        yv=torch.tensor(yte,dtype=torch.float32,device=dev).view(-1,1)
        m=nn.Sequential(nn.Linear(Xt.shape[1],32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1)).to(dev)
        opt=torch.optim.Adam(m.parameters(),lr=3e-3)
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss=(m(Xt)-yt).abs().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p=m(Xv).cpu().numpy().reshape(-1)
        mae=float(np.mean(np.abs(yte-p)))
        rmse=float(np.sqrt(np.mean((yte-p)**2)))
        self.res["torch_ts"]={"mae":mae,"rmse":rmse,"k":int(k)}
        self.m_torch_ts=m
        self.torch_norm={"mu":z_mu.reshape(-1).tolist(),"sd":z_sd.reshape(-1).tolist()}
        return self

  

    def save(self):
        c=self.c
        for p in [c.inter,c.primary,c.feat,c.mi,c.models,c.mo,c.rep]:
            p.mkdir(parents=True,exist_ok=True)

        self.df.to_csv(c.inter/"all.csv",index=False)
        pd.concat([self.X,self.y.rename("y")],axis=1).to_csv(c.mi/"xy.csv",index=False)
        self.ts.to_csv(c.feat/"ts.csv",index=False)
        (c.mo/"metrics.json").write_text(json.dumps(self.res,indent=2),encoding="utf-8")

        if hasattr(self,"m_xgb"):
            self.m_xgb.save_model(c.models/"xgb.json")
        if hasattr(self,"m_xgb_ts"):
            self.m_xgb_ts.save_model(c.models/"xgb_ts.json")
        if hasattr(self,"m_tf_ts"):
            self.m_tf_ts.save(c.models/"tf_ts.keras",include_optimizer=False)
            (c.models/"tf_ts_norm.json").write_text(json.dumps(self.tf_norm),encoding="utf-8")
        if hasattr(self,"m_torch_ts"):
            import torch
            torch.save(self.m_torch_ts.state_dict(),c.models/"torch_ts.pt")
            (c.models/"torch_ts_norm.json").write_text(json.dumps(self.torch_norm),encoding="utf-8")

        if hasattr(self,"ts_pred"):
            self.ts_pred.to_csv(c.mo/"ts_pred.csv",index=False)
        return self


 
      