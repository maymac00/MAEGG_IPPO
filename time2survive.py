import numpy as np

eff_rates = [0.2, 0.4, 0.6, 0.8]
policies_t2s = {eff: {} for eff in eff_rates}
policies_t2s_std = {eff: {} for eff in eff_rates}
policies_t2s_meds = {eff: {} for eff in eff_rates}
policies_t2s_iqr = {eff: {} for eff in eff_rates}

# eff 0.2
policies_t2s[0.2][0] = np.array([117.8988,      np.nan,      np.nan,      np.nan,      np.nan])
policies_t2s_std[0.2][0] = np.array([28.69,      np.nan,      np.nan,      np.nan,      np.nan])
policies_t2s_meds[0.2][0] = np.array([ 113.0, np.nan, np.nan, np.nan, np.nan])
policies_t2s_iqr[0.2][0] = np.array([57.0, 0, 0, 0, 0])

policies_t2s[0.2][1] = np.array([ 109.43,350.26,270.79,282.02,306.95 ])
policies_t2s_std[0.2][1] = np.array([ 25.17,38.83,53.31,52.91,47.24 ])
policies_t2s_meds[0.2][1] = np.array([ 107.0,349.5,271.0,280.5,309.0 ])
policies_t2s_iqr[0.2][1] = np.array([ 31.0,50.0,73.5,70.0,55.25 ])

policies_t2s[0.2][10] = np.array([ 118.16,303.92,326.91,308.83,305.86 ])
policies_t2s_std[0.2][10] = np.array([ 21.7,41.92,37.51,41.9,40.64 ])
policies_t2s_meds[0.2][10] = np.array([ 117.0,304.0,327.0,311.0,307.0 ])
policies_t2s_iqr[0.2][10] = np.array([ 29.0,52.0,43.25,51.0,55.25 ])

policies_t2s[0.2][1000] = np.array([ 140.38,334.98,355.73,347.07,351.89 ])
policies_t2s_std[0.2][1000] = np.array([ 39.74,53.64,51.12,52.25,51.59 ])
policies_t2s_meds[0.2][1000] = np.array([ 134.0,334.0,355.0,346.0,351.0 ])
policies_t2s_iqr[0.2][1000] = np.array([ 50.0,72.0,67.0,70.0,68.5 ])

# eff 0.4
policies_t2s[0.4][0] = np.array([ 121.43,110.28, np.nan,np.nan,np.nan ])
policies_t2s_std[0.4][0] = np.array([ 30.55,25.42, np.nan,np.nan,np.nan ])
policies_t2s_meds[0.4][0] = np.array([ 116.0,106.0, np.nan, np.nan, np.nan])
policies_t2s_iqr[0.4][0] = np.array([ 37.25,32.0,np.nan, np.nan, np.nan])


policies_t2s[0.4][1] = np.array([ 152.49,151.58,278.42,237.35,286.68 ])
policies_t2s_std[0.4][1] = np.array([ 42.33,41.56,43.53,44.99,41.45 ])
policies_t2s_meds[0.4][1] = np.array([ 146.0,145.0,274.0,231.0,281.5 ])
policies_t2s_iqr[0.4][1] = np.array([ 51.0,51.0,54.75,56.0,52.0 ])


policies_t2s[0.4][10] = np.array([ 122.17,113.82,213.94,241.83,200.69 ])
policies_t2s_std[0.4][10] =  np.array([ 30.13,27.66,30.7,27.19,31.61 ])
policies_t2s_meds[0.4][10] = np.array([ 118.0,110.0,214.0,241.0,200.0 ])
policies_t2s_iqr[0.4][10] = np.array([ 37.0,35.0,42.25,35.0,41.0 ])

policies_t2s[0.4][1000] = np.array([ 113.98,119.69,224.31,204.54,225.59 ])
policies_t2s_std[0.4][1000] = np.array([ 26.8,28.92,32.36,32.71,29.69 ])
policies_t2s_meds[0.4][1000] = np.array([ 110.0,118.0,226.0,203.0,225.0 ])
policies_t2s_iqr[0.4][1000] = np.array([ 34.25,37.25,41.0,44.25,40.0 ])

# eff 0.6
policies_t2s[0.6][0] = np.array([ 127.95,124.97,133.09,np.nan,np.nan ])
policies_t2s_std[0.6][0] = np.array([ 33.92,34.71,36.79,np.nan,np.nan ])
policies_t2s_meds[0.6][0] = np.array([ 119.0,122.5,132.0,np.nan,np.nan ])
policies_t2s_iqr[0.6][0] = np.array([ 42.0,43.25,49.0,np.nan,np.nan ])

policies_t2s[0.6][1] = np.array([ 127.82,129.2,130.74,196.96,189.77 ])
policies_t2s_std[0.6][1] = np.array([ 30.35,32.04,34.06,23.24,22.9 ])
policies_t2s_meds[0.6][1] = np.array([ 128.0,125.0,127.0,197.0,191.0 ])
policies_t2s_iqr[0.6][1] = np.array([ 37.0,40.5,45.25,31.0,29.0 ])

policies_t2s[0.6][10] = np.array([ 127.4,126.95,124.43,193.81,187.78 ])
policies_t2s_std[0.6][10] = np.array([ 27.89,28.48,28.89,25.46,22.44 ])
policies_t2s_meds[0.6][10] = np.array([ 129.0,123.0,121.0,192.5,187.0 ])
policies_t2s_iqr[0.6][10] = np.array([ 36.25,36.0,39.25,34.0,29.0 ])

policies_t2s[0.6][1000] = np.array([ 137.75,139.23,139.78,195.92,199.44 ])
policies_t2s_std[0.6][1000] = np.array([ 39.61,37.81,38.99,28.66,27.08 ])
policies_t2s_meds[0.6][1000] = np.array([ 129.0,120.0,125.0,192.0,188.0 ])
policies_t2s_iqr[0.6][1000] = np.array([ 45.25,42.0,44.25,29.0,29.0 ])


policies_t2s[0.8][0] = np.array([ 153.08,150.87,146.87,150.75, np.nan ])
policies_t2s_std[0.8][0] = np.array([ 45.35,43.54,43.43,44.11, np.nan ])
policies_t2s_meds[0.8][0] = np.array([ 149.0,144.0,143.0,147.0,np.nan ])
policies_t2s_iqr[0.8][0] = np.array([ 55.0,60.5,55.25,59.0,np.nan ])


policies_t2s[0.8][1] = np.array([ 139.83,147.64,145.55,143.52,183.62 ])
policies_t2s_std[0.8][1] = np.array([ 34.67,35.56,31.9,33.78,21.08 ])
policies_t2s_meds[0.8][1] = np.array([ 138.0,145.0,148.0,143.0,184.0 ])
policies_t2s_iqr[0.8][1] = np.array([ 50.25,53.0,49.0,48.0,30.0 ])

policies_t2s[0.8][10] = np.array([ 141.93,129.88,145.4,144.63,191.13 ])
policies_t2s_std[0.8][10] = np.array([ 31.05,30.02,29.37,29.03,20.37 ])
policies_t2s_meds[0.8][10] = np.array([ 145.0,129.0,143.5,146.0,189.0 ])
policies_t2s_iqr[0.8][10] = np.array([ 38.5,46.0,33.25,38.0,26.0 ])

policies_t2s[0.8][1000] = np.array([ 139.94,137.76,140.01,138.99,176.45 ])
policies_t2s_std[0.8][1000] = np.array([ 34.09,31.42,32.3,30.43,19.31 ])
policies_t2s_meds[0.8][1000] = np.array([ 138.0,136.0,140.0,137.0,173.0 ])
policies_t2s_iqr[0.8][1000] = np.array([ 46.25,42.25,43.0,44.0,24.0 ])

