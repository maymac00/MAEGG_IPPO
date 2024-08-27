import numpy as np

from value_vectors import policies

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

policies_t2s[0.2][1] = np.array([ 130.15,348.68,324.58,347.09,322.42 ])
policies_t2s_std[0.2][1] = np.array([ 32.56,45.5,51.58,45.69,53.93 ])
policies_t2s_meds[0.2][1] = np.array([ 126.0,348.0,325.0,347.0,323.0 ])
policies_t2s_iqr[0.2][1] = np.array([58.0, 89.0, 63.0, 94.0, 83.0])

policies_t2s[0.2][10] = np.array([ 144.86,358.55,373.7,370.08,359.05 ])
policies_t2s_std[0.2][10] = np.array([ 41.0,53.58,49.68,51.74,52.87 ])
policies_t2s_meds[0.2][10] = np.array([ 138.0,356.0,373.0,371.0,358.0 ])
policies_t2s_iqr[0.2][10] = np.array([ 56.0,77.0,76.0,72.0,74.0 ])

policies_t2s[0.2][1000] = np.array([ 140.38,334.98,355.73,347.07,351.89 ])
policies_t2s_std[0.2][1000] = np.array([ 39.74,53.64,51.12,52.25,51.59 ])
policies_t2s_meds[0.2][1000] = np.array([ 134.0,334.0,355.0,346.0,351.0 ])
policies_t2s_iqr[0.2][1000] = np.array([ 50.0,72.0,67.0,70.0,68.5 ])

# eff 0.4
policies_t2s[0.4][0] = np.array([ 143.1,141.66, np.nan,np.nan,np.nan ])
policies_t2s_std[0.4][0] = np.array([45.61,41.08, np.nan,np.nan,np.nan ])
policies_t2s_meds[0.4][0] = np.array([135.0, 136.0, np.nan, np.nan, np.nan])
policies_t2s_iqr[0.4][0] = np.array([56.0,57, np.nan, np.nan, np.nan])


policies_t2s[0.4][1] = np.array([ 152.49,151.58,278.42,237.35,286.68 ])
policies_t2s_std[0.4][1] = np.array([ 42.33,41.56,43.53,44.99,41.45 ])
policies_t2s_meds[0.4][1] = np.array([ 146.0,145.0,274.0,231.0,281.5 ])
policies_t2s_iqr[0.4][1] = np.array([ 51.0,51.0,54.75,56.0,52.0 ])


policies_t2s[0.4][10] = np.array([ 143.35,135.94,260.96,260.88,227.46 ])
policies_t2s_std[0.4][10] = np.array([ 39.49,34.78,36.35,35.96,39.2 ])
policies_t2s_meds[0.4][10] = np.array([ 138.0,131.0,258.0,258.0,224.0 ])
policies_t2s_iqr[0.4][10] = np.array([ 50.0,45.0,47.0,45.0,52.0 ])

policies_t2s[0.4][1000] = np.array([ 129.59,128.44,248.13,233.78,226.75 ])
policies_t2s_std[0.4][1000] = np.array([ 33.87,32.85,28.89,31.51,32.9 ])
policies_t2s_meds[0.4][1000] = np.array([ 125.0,124.0,246.0,232.0,226.0 ])
policies_t2s_iqr[0.4][1000] = np.array([ 44.0,42.0,38.0,40.0,44.0 ])

# eff 0.6
policies_t2s[0.6][0] = np.array([ 127.95,124.97,133.09,np.nan,np.nan ])
policies_t2s_std[0.6][0] = np.array([ 33.92,34.71,36.79,np.nan,np.nan ])
policies_t2s_meds[0.6][0] = np.array([ 119.0,122.5,132.0,np.nan,np.nan ])
policies_t2s_iqr[0.6][0] = np.array([ 119.0,122.5,132.0,np.nan,np.nan ])

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

