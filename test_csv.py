import pandas as pd
from pathlib import Path
import pickle
import json

save_compar_by_deform = "./Euc_dist/compar_by_deform/"

file_name = "test"
deform_flag = True
deform_name = "wrap"

data = {"g": 393.9690254949356,
        "u": 393.6978474009045,
        }
col_list = ['u', 'g']

# print(data["g\t\t"])
# data["g"].append(448.537189563781)
# print(data["g"])

print(list(data.keys()), list(data.values()))
print()
"""
data = {"g": [93.9690254949356],
        "u": [33.6978474009045],
        }
df = pd.DataFrame(data)
print(df)
fname = save_compar_by_deform + file_name + ".csv"
df.to_csv(fname, sep='\t', index=True, na_rep='nan', mode='w', float_format="%.6f", doublequote=False)
print(Path(fname).read_text())


# df = pd.read_csv(save_compar_by_deform + file_name + ".csv", index_col=0)
"""
"""
print(df)

f = open(save_compar_by_deform + file_name + ".csv", "r+")
col_name = ""
if deform_flag:
    f.write(deform_name)

else:
    f.write("変形なし")
for key, value in dist_dic2:
    f.write(f'{value} との比較 {key}')
"""
