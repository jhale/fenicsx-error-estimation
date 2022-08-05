import pandas as pd
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def main():
    test_case = 'goal-oriented'
    alph = 'abcde'    # Code to define variables in tikz
    tikzfig = open('tex_tables/{}/tikzfig_P1.tex'.format(test_case), 'w')

    for j in [0, 1, 2, 3]:
        for i in [1, 2, 3, 4][j:]:
            df= pd.read_pickle(f'./output_minimal/bw_{j}_{i}/results.pkl')
            print(df)
            est_w = df["error_bw_u"].values[-1] * df["error_bw_z"].values[-1]
            bw_eff = np.round(est_w/df['exact_error'].values[-1], 2)
            print(f"bw_{j}_{i}_eff =", bw_eff)
            tikzfig.write('\\def\\{}{}{{{}}};\n'.format(alph[j], alph[i], bw_eff))
    for i, name in enumerate(['res', 'zz']):
        df= pd.read_pickle('./output_minimal/{}/results.pkl'.format(name))
        print(df)
        est_w = df["error_{}_u".format(name)].values[-1] * \
        df["error_{}_z".format(name)].values[-1]
        est_eff = np.round(est_w/df['exact_error'.format(name)].values[-1], 2)
        print(f"{name}_eff =", est_eff)
        tikzfig.write('\\def\\{}{}{{{}}};\n'.format('o', alph[i], est_eff))
    template = open('P1_template.tex', 'r')
    for line in template:
        tikzfig.write(line)

    tikzfig.close()
    template.close()
    return

if __name__=='__main__':
    main()
