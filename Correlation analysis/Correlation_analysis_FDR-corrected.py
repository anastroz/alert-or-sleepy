# -*- coding: utf-8 -*-
"""

This script was used to calculate the correlations in the experimental study for both the PVT and MCT tasks. It uses the datasets exported from JASP.

Spearman's rho correlations are computed, and the Benjamini-Yekutieli FDR is applied to the p values. 

## PVT

Correlations for the PVT, FDR-corrected
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


pvt_df = pd.read_csv("Statystyki - Badanie 2 - cz. 3 (PVT)_data.csv")

pvt_df

pvt_df.columns

which = ["(1-1)", "(1-2)", "(1-3)", "(1-4)", "(1-5)", "(2-1)", "(2-2)", "(2-3)", "(2-4)", "(2-5)"]


results_name = []
results_rho = []
results_pvalue = []

for trial in which:
  columns = ['Mean_RT_resp100_1000ms '+trial, "Total_resp_lapses500_1000ms "+trial, "Total_omissions "+trial, "HRV_LFHF "+trial,
           "HRV_SDNN "+trial, "ECG_Rate_Mean "+trial]
  scales = ["KSS_after_trial "+trial, "SAS_alert_after_trial "+trial, "SAS_sleep_after_trial "+trial]
  for scale in scales:
    for col in columns:
      name = "{s} & {c}".format(s=scale, c=col)
      results_name.append(name)
      print(name)
      result = ss.spearmanr(pvt_df[scale], pvt_df[col], nan_policy='omit', alternative='two-sided')
      results_rho.append(result[0])
      results_pvalue.append(result[1])
      print(result[0])
      print(result[1])
    print("-----------------")


df = pd.DataFrame({'Name': results_name, 'rho': results_rho, 'p-value': results_pvalue})

df

pvalues_fdr = ss.false_discovery_control(df['p-value'], method='by')

df['p-values_fdr'] = pvalues_fdr

df

df.to_csv('Correlation_analysis_FDR-corrected_PVT.csv')

"""## MCT

Correlations for the MCT, FDR-corrected"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


mct_df = pd.read_csv("Statystyki - Badanie 2 - cz. 2 (MCT)_data.csv")

mct_df

which = ["(1-1)", "(1-2)", "(1-3)", "(1-4)", "(1-5)", "(2-1)", "(2-2)", "(2-3)", "(2-4)", "(2-5)"]


results_name = []
results_rho = []
results_pvalue = []

for trial in which:
  columns = ['Mean_RT_respTGT_100_1000ms '+trial, "Total_resp_lapsesTGT_500_1000ms "+trial, "Total_omissions "+trial, "HRV_LFHF "+trial,
           "HRV_SDNN "+trial, "ECG_Rate_Mean "+trial]
  scales = ["KSS_after_trial "+trial, "SAS_alert_after_trial "+trial, "SAS_sleep_after_trial "+trial]
  for scale in scales:
    for col in columns:
      name = "{s} & {c}".format(s=scale, c=col)
      results_name.append(name)
      print(name)
      result = ss.spearmanr(mct_df[scale], mct_df[col], nan_policy='omit', alternative='two-sided')
      results_rho.append(result[0])
      results_pvalue.append(result[1])
      print(result[0])
      print(result[1])
    print("-----------------")


df = pd.DataFrame({'Name': results_name, 'rho': results_rho, 'p-value': results_pvalue})

df

pvalues_fdr = ss.false_discovery_control(df['p-value'], method='by')

df['p-values_fdr'] = pvalues_fdr

df

df.to_csv('Correlation_analysis_FDR-corrected_MCT.csv')

