import numpy as np

eff_rates = [0.2, 0.4, 0.6, 0.8]
policies_sr_mean = {eff: {} for eff in eff_rates}
policies_sr_std = {eff: {} for eff in eff_rates}
policies_sr_meds = {eff: {} for eff in eff_rates}
policies_sr_iqr = {eff: {} for eff in eff_rates}

# eff 0.2
# amm ecai/db0_effrate0.2_we0_ECAI_new
policies_sr_mean[0.2][0] = np.array([1.0, 0.007, 0.031, 0.013, 0.019])
policies_sr_std[0.2][0] = np.array([0.0, 0.0837, 0.1733, 0.1132, 0.1365])
policies_sr_meds[0.2][0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_iqr[0.2][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.2][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.2][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.2][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.2][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.2][1000] = np.array([ 1.0,0.999,0.993,0.991,0.989 ])
policies_sr_std[0.2][1000] = np.array([ 0.0, 0.03160, 0.08337, 0.094440, 0.1043])
policies_sr_meds[0.2][1000] = np.array([ 1.0,1.0,1.0,1.0,1.0 ])
policies_sr_iqr[0.2][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


# eff 0.4
policies_sr_mean[0.4][0] = np.array([1.0, 1.0, 0.0, 0.007, 0.003])
policies_sr_std[0.4][0] = np.array([0.0, 0.0, 0.0, 0.0833, 0.0546])
policies_sr_meds[0.4][0] = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
policies_sr_iqr[0.4][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.4][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.4][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.4][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.4][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.4][1000] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.4][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.4][1000] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.4][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# eff 0.6
policies_sr_mean[0.6][0] = np.array([1., 1., 1., 0.0, 0.0])
policies_sr_std[0.6][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.6][0] = np.array([1., 1., 1., 0.0, 0.0])
policies_sr_iqr[0.6][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.6][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.6][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.6][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.6][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.6][1000] =  np.array([ 1.0,1.0,1.0,1.0,1.0 ])
policies_sr_std[0.6][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.6][1000] = np.array([ 1.0,1.0,1.0,1.0,1.0 ])
policies_sr_iqr[0.6][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# eff 0.8
policies_sr_mean[0.8][0] = np.array([1., 1., 1., 1., 0.0])
policies_sr_std[0.8][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.8][0] = np.array([1., 1., 1., 1., 0.0])
policies_sr_iqr[0.8][0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.8][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.8][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.8][10] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.8][10] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

policies_sr_mean[0.8][1000] = np.array([1., 1., 1., 1.,1.])
policies_sr_std[0.8][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
policies_sr_meds[0.8][1000] = np.array([1., 1., 1., 1.,1.])
policies_sr_iqr[0.8][1000] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])