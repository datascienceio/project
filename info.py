from pathlib import Path
import sys

root=Path(r"C:\Users\prach\Downloads\THE-DATA0SCIENCE-PROJECT")
sys.path.insert(0,str(root/"src"))
from ds0 import summary

(Capstone(root)
 .load()
 .clean()
 .make_features()
 .make_ts()
 .fit_xgb()
 .fit_xgb_ts()
 .fit_tf_ts()
 .fit_torch_ts()
 .save())
