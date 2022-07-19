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
            if not (j == 0 and i == 1):
                df= pd.read_pickle('./output/bw_{}_{}/results.pkl'.format(j,i))
                print(df)
                bw_eff = np.round(df['error_bw_w'.format(i,j)].values[-1]/df['exact_error'.format(i,j)].values[-1], 2)
                print(f"bw_{j}_{i}_eff =", bw_eff)
                tikzfig.write('\\def\\{}{}{{{}}};\n'.format(alph[j], alph[i], bw_eff))
    for i, name in enumerate(['res', 'zz']):
        df= pd.read_pickle('./output/{}/results.pkl'.format(name))
        est_eff = np.round(df['error_{}_w'.format(name)].values[-1]/df['exact_error'.format(name)].values[-1], 2)
        tikzfig.write('\\def\\{}{}{{{}}};\n'.format('o', alph[i], est_eff))
    template = open('P1_template.tex', 'r')
    for line in template:
        tikzfig.write(line)

    tikzfig.close()
    template.close()
    return

if __name__=='__main__':
    main()
