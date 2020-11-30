# Python/STAN Implementation of Multiplicative Media Mix Model
The methodology of this project is based on [this paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf) by Google, but is applied to a more complicated, real-world setting, where 1) there are 13 media channels and 46 control variables; 2) models are built in a stacked way.    
     


# 1. Introduction
Marketing Mix Model,  or  Media Mix Model (MMM) is used by advertisers to measure how their media spending contributes to sales, so as to optimize future budget allocation. **ROAS** (return on ad spend) and **mROAS** (marginal ROAS) are the key metrics to look at. High ROAS indicates the channel is efficient, high mROAS means increasing spend in the channel will yield a high return based on current spending level.   
    
**Procedures**        

1. Fit a multivariate regression model, using media channels' impressions (or spending) and control variables to predict sales;

2. Decompose sales to each media channel's contribution. Channel contribution is calculated by comparing original sales and predicted sales upon removal of the channel;

3. Compute ROAS and mROAS using channel contribution and spending. 
  
  ​    

**Intuition of MMM**    
- Offline channel's influence is hard to track. E.g., a customer saw a TV ad, and made a purchase at store.
- Media channels' influences are intertwined.    

**Actual Customer Journey: Multiple Touchpoints**    
A customer saw a product on TV > clicked on a display ad > clicked on a paid seach ad > made a purchase of $30. In this case, 3 touchpoints contributed to the conversion, and they should all get credits for this conversion.    
![actual customer journey - multiple touchpoints](img/mmm_stan_multitouch.png)    

​    

**What's trackable: Last Digital Touchpoint**    
Usually, only the last digital touchpoint can be tracked. In this case, SEM, and it will get all credits for this conversion.    
![what can be tracked - last touchpoint](img/mmm_stan_lasttouch.png)    
So, a good attribution model should take into account all the relevant variables leading to conversion.    

​    

## 1.1 Multiplicative MMM
Since media channels work interactively, a multiplicative model structure is adopted:    
![multiplicative MMM](img/mmm_stan_formular_mmm_multiplicative.png)
Take log of both sides, we get the linear form (log-log model):    
![multiplicative MMM - linear form](img/mmm_stan_formular_mmm_linear.png)    

**Constraints on Coefficients**

1. Media coefficients are positive.

2. Control variables like discount, macro economy, event/retail holiday are expected to have positive impact on sales, their coefficients should also be positive.

   ​    

## 1.2 Adstock
Media effect on sales may lag behind the original exposure and extend several weeks. The carry-over effect is modeled by Adstock:    
![adstock transformation](img/mmm_stan_formular_adstock.png)    
L: length of the media effect    
P: peak/delay of the media effect, how many weeks it's lagging behind first exposure    
D: decay/retention rate of the media channel, concentration of the effect    
The media effect of current weeks is a weighted average of current week and previous (L− 1) weeks.    
    
**Adstock Example**    
![adstock example](img/mmm_stan_adstock_example.png)    

​    

**Adstock with Varying Decay**    
The larger the decay, the more scattered the effect.    
![adstock parameter - decay](img/mmm_stan_adstock_decay.png)    
**Adstock with Varying Length**    
The impact of length is relatively minor. In model training, length could be fixed to 8 weeks or a period long enough for the media effect to finish.    
![adstock parameter - length](img/mmm_stan_adstock_length.png)   
    
**Implementation**


```python
import numpy as np
import pandas as pd

def apply_adstock(x, L, P, D):
    '''
    params:
    x: original media variable, array
    L: length
    P: peak, delay in effect
    D: decay, retain rate
    returns:
    array, adstocked media variable
    '''
    x = np.append(np.zeros(L-1), x)
    
    weights = np.zeros(L)
    for l in range(L):
        weight = D**((l-P)**2)
        weights[L-1-l] = weight
    
    adstocked_x = []
    for i in range(L-1, len(x)):
        x_array = x[i-L+1:i+1]
        xi = sum(x_array * weights)/sum(weights)
        adstocked_x.append(xi)
    adstocked_x = np.array(adstocked_x)
    return adstocked_x

def adstock_transform(df, md_cols, adstock_params):
    md_df = pd.DataFrame()
    for md_col in md_cols:
        md = md_col.split('_')[-1]
        L, P, D = adstock_params[md]['L'], adstock_params[md]['P'], adstock_params[md]['D']
        xa = apply_adstock(df[md_col].values, L, P, D)
        md_df[md_col] = xa
    return md_df
```


## 1.2 Diminishing Return    
After a certain saturation point, increasing spend will yield diminishing marginal return, the channel will be losing efficiency as you keep overspending on it. The diminishing return is modeled by Hill function:    
![Hill function](img/mmm_stan_formular_hill.png)    
K: half saturation point    
S: slope    
    
**Hill function with varying K and S**
![Hill function with varying K and S](img/mmm_stan_hill_function_demo.png)    

​    

**Implementation**

```python
def hill_transform(x, ec, slope):
    return 1 / (1 + (x / ec)**(-slope))
```



# 2. Model Specification & Implementation

## Data    
Four years' (209 weeks) records of sales, media impression and media spending at weekly level.   
    
**1. Media Variables**
- Media Impression (prefix='mdip_'): impressions of 13 media channels: direct mail, insert, newspaper, digital audio, radio, TV, digital video, social media, online display, email, SMS, affiliates, SEM.
- Media Spending (prefix='mdsp_'): spending of media channels.
  

**2. Control Variables**    
- Macro Economy (prefix='me_'): CPI, gas price.
- Markdown (prefix='mrkdn_'): markdown/discount.
- Store Count ('st_ct')
- Retail Holidays (prefix='hldy_'): one-hot encoded.
- Seasonality (prefix='seas_'): month, with Nov and Dec further broken into to weeks. One-hot encoded.
  

**3. Sales Variable** ('sales')


```python
df = pd.read_csv('data.csv')

# 1. media variables
# media impression
mdip_cols=[col for col in df.columns if 'mdip_' in col]
# media spending
mdsp_cols=[col for col in df.columns if 'mdsp_' in col]

# 2. control variables
# macro economics variables
me_cols = [col for col in df.columns if 'me_' in col]
# store count variables
st_cols = ['st_ct']
# markdown/discount variables
mrkdn_cols = [col for col in df.columns if 'mrkdn_' in col]
# holiday variables
hldy_cols = [col for col in df.columns if 'hldy_' in col]
# seasonality variables
seas_cols = [col for col in df.columns if 'seas_' in col]
base_vars = me_cols+st_cols+mrkdn_cols+va_cols+hldy_cols+seas_cols

# 3. sales variables
sales_cols =['sales']
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wk_strt_dt</th>
      <th>mdip_dm</th>
      <th>mdip_inst</th>
      <th>mdip_nsp</th>
      <th>mdip_auddig</th>
      <th>mdip_audtr</th>
      <th>mdip_vidtr</th>
      <th>mdip_viddig</th>
      <th>mdip_so</th>
      <th>mdip_on</th>
      <th>mdip_em</th>
      <th>mdip_sms</th>
      <th>mdip_aff</th>
      <th>mdip_sem</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-03</td>
      <td>4863885</td>
      <td>29087520</td>
      <td>2421933</td>
      <td>692315</td>
      <td>37778097</td>
      <td>10038746</td>
      <td>2111112</td>
      <td>0</td>
      <td>3271007</td>
      <td>1514755</td>
      <td>27281</td>
      <td>197828</td>
      <td>83054</td>
      <td>72051457.64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-10</td>
      <td>20887502</td>
      <td>8345120</td>
      <td>3984494</td>
      <td>475810</td>
      <td>12063657</td>
      <td>9847977</td>
      <td>587184</td>
      <td>0</td>
      <td>4260715</td>
      <td>2234569</td>
      <td>27531</td>
      <td>123688</td>
      <td>83124</td>
      <td>78794770.54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-17</td>
      <td>11097724</td>
      <td>17276800</td>
      <td>1846832</td>
      <td>784732</td>
      <td>5770115</td>
      <td>7235336</td>
      <td>1015658</td>
      <td>0</td>
      <td>4405992</td>
      <td>1616990</td>
      <td>55267</td>
      <td>186781</td>
      <td>79768</td>
      <td>70071185.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-24</td>
      <td>1023446</td>
      <td>18468480</td>
      <td>2394834</td>
      <td>1032301</td>
      <td>12174000</td>
      <td>8625122</td>
      <td>2149160</td>
      <td>0</td>
      <td>6638320</td>
      <td>1897998</td>
      <td>32470</td>
      <td>122389</td>
      <td>138936</td>
      <td>68642464.59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-31</td>
      <td>21109811</td>
      <td>26659920</td>
      <td>3312008</td>
      <td>400456</td>
      <td>31656134</td>
      <td>19785657</td>
      <td>2408661</td>
      <td>0</td>
      <td>4347752</td>
      <td>2569158</td>
      <td>55878</td>
      <td>209969</td>
      <td>87531</td>
      <td>86190784.65</td>
    </tr>
  </tbody>
</table>
</div>



## Model Architecture
The model is built in a stacked way. Three models are trained:   
- Control Model
- Marketing Mix Model
- Diminishing Return Model    
![mmm_stan_model_architecture.png](img/mmm_stan_model_architecture.png)

​    

## 2.1 Control Model / Base Sales Model    

**Goal**: predict base sales (X_ctrl) as an input variable to MMM, this represents the baseline sales trend without any marketing activities.    
![control model formular.png](img/mmm_stan_model_control.png)    
X1: control variables positively related with sales, including macro economy, store count, markdown, holiday.    
X2: control variables that may have either positive or negtive impact on sales: seasonality.    
Target variable: ln(sales).    
The variables are centralized by mean.
    
**Priors**    
![control model priors](img/mmm_stan_model_control_prior.png)    

​    

**Implementation**    


```python
import pystan
import os
os.environ['CC'] = 'gcc-10'
os.environ['CXX'] = 'g++-10'

def apply_mean_center(x):
    mu = np.mean(x)
    xm = x/mu
    return xm, mu

def mean_center_trandform(df, cols):
    df_new = pd.DataFrame()
    sc = {}
    for col in cols:
        x = df[col].values
        df_new[col], mu = apply_mean_center(x)
        sc[col] = mu
    return df_new, sc

def mean_log1p_trandform(df, cols):
    df_new = pd.DataFrame()
    sc = {}
    for col in cols:
        x = df[col].values
        xm, mu = apply_mean_center(x)
        sc[col] = mu
        df_new[col] = np.log1p(xm)
    return df_new, sc

# mean-centralize: sales, numeric base_vars
df_ctrl, sc_ctrl = mean_center_trandform(df, ['sales']+me_cols+st_cols+mrkdn_cols)
df_ctrl = pd.concat([df_ctrl, df[hldy_cols+seas_cols]], axis=1)

# variables positively related to sales: macro economy, store count, markdown, holiday
pos_vars = [col for col in base_vars if col not in seas_cols]
X1 = df_ctrl[pos_vars].values

# variables may have either positive or negtive impact on sales: seasonality
pn_vars = seas_cols
X2 = df_ctrl[pn_vars].values

ctrl_data = {
    'N': len(df_ctrl),
    'K1': len(pos_vars), 
    'K2': len(pn_vars), 
    'X1': X1,
    'X2': X2, 
    'y': df_ctrl['sales'].values,
    'max_intercept': min(df_ctrl['sales'])
}

ctrl_code1 = '''
data {
  int N; // number of observations
  int K1; // number of positive predictors
  int K2; // number of positive/negative predictors
  real max_intercept; // restrict the intercept to be less than the minimum y
  matrix[N, K1] X1;
  matrix[N, K2] X2;
  vector[N] y; 
}

parameters {
  vector<lower=0>[K1] beta1; // regression coefficients for X1 (positive)
  vector[K2] beta2; // regression coefficients for X2
  real<lower=0, upper=max_intercept> alpha; // intercept
  real<lower=0> noise_var; // residual variance
}

model {
  // Define the priors
  beta1 ~ normal(0, 1); 
  beta2 ~ normal(0, 1); 
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  // The likelihood
  y ~ normal(X1*beta1 + X2*beta2 + alpha, sqrt(noise_var));
}
'''

sm1 = pystan.StanModel(model_code=ctrl_code1, verbose=True)
fit1 = sm1.sampling(data=ctrl_data, iter=2000, chains=4)
fit1_result = fit1.extract()
```


```python
# extract control model parameters and predict base sales -> df['base_sales']
def extract_ctrl_model(fit_result, pos_vars=pos_vars, pn_vars=pn_vars, 
                       extract_param_list=False):
    ctrl_model = {}
    ctrl_model['pos_vars'] = pos_vars
    ctrl_model['pn_vars'] = pn_vars
    ctrl_model['beta1'] = fit_result['beta1'].mean(axis=0).tolist()
    ctrl_model['beta2'] = fit_result['beta2'].mean(axis=0).tolist()
    ctrl_model['alpha'] = fit_result['alpha'].mean()
    if extract_param_list:
        ctrl_model['beta1_list'] = fit_result['beta1'].tolist()
        ctrl_model['beta2_list'] = fit_result['beta2'].tolist()
        ctrl_model['alpha_list'] = fit_result['alpha'].tolist()
    return ctrl_model

def ctrl_model_predict(ctrl_model, df):
    pos_vars, pn_vars = ctrl_model['pos_vars'], ctrl_model['pn_vars'] 
    X1, X2 = df[pos_vars], df[pn_vars]
    beta1, beta2 = np.array(ctrl_model['beta1']), np.array(ctrl_model['beta2'])
    alpha = ctrl_model['alpha']
    y_pred = np.dot(X1, beta1) + np.dot(X2, beta2) + alpha
    return y_pred

base_sales_model = extract_ctrl_model(fit1_result, pos_vars=pos_vars, pn_vars=pn_vars)
base_sales = ctrl_model_predict(base_sales_model, df_ctrl)
df['base_sales'] = base_sales*sc_ctrl['sales']
```

MAPE of control model: 8.63%



## 2.2 Marketing Mix Model

**Goal**:
- Find appropriate adstock parameters for media channels;
- Decompose sales to media channels' contribution (and non-marketing contribution).

![marketing mix model formular](img/mmm_stan_model_mmm.png)    
L: length of media impact    
P: peak of media impact    
D: decay of media impact    
X: adstocked media impression variables and base sales    
Target variable: ln(sales)    
Variables are centralized by mean.
    
**Priors**    
![marketing mix model priors](img/mmm_stan_model_mmm_prior.png) 
    
**Implementation**    


```python
df_mmm, sc_mmm = mean_log1p_trandform(df, ['sales', 'base_sales'])
mu_mdip = df[mdip_cols].apply(np.mean, axis=0).values
max_lag = 8
num_media = len(mdip_cols)
# padding zero * (max_lag-1) rows
X_media = np.concatenate((np.zeros((max_lag-1, num_media)), df[mdip_cols].values), axis=0)
X_ctrl = df_mmm['base_sales'].values.reshape(len(df),1)
model_data2 = {
    'N': len(df),
    'max_lag': max_lag, 
    'num_media': num_media,
    'X_media': X_media, 
    'mu_mdip': mu_mdip,
    'num_ctrl': X_ctrl.shape[1],
    'X_ctrl': X_ctrl, 
    'y': df_mmm['sales'].values
}

model_code2 = '''
functions {
  // the adstock transformation with a vector of weights
  real Adstock(vector t, row_vector weights) {
    return dot_product(t, weights) / sum(weights);
  }
}
data {
  // the total number of observations
  int<lower=1> N;
  // the vector of sales
  real y[N];
  // the maximum duration of lag effect, in weeks
  int<lower=1> max_lag;
  // the number of media channels
  int<lower=1> num_media;
  // matrix of media variables
  matrix[N+max_lag-1, num_media] X_media;
  // vector of media variables' mean
  real mu_mdip[num_media];
  // the number of other control variables
  int<lower=1> num_ctrl;
  // a matrix of control variables
  matrix[N, num_ctrl] X_ctrl;
}
parameters {
  // residual variance
  real<lower=0> noise_var;
  // the intercept
  real tau;
  // the coefficients for media variables and base sales
  vector<lower=0>[num_media+num_ctrl] beta;
  // the decay and peak parameter for the adstock transformation of
  // each media
  vector<lower=0,upper=1>[num_media] decay;
  vector<lower=0,upper=ceil(max_lag/2)>[num_media] peak;
}
transformed parameters {
  // the cumulative media effect after adstock
  real cum_effect;
  // matrix of media variables after adstock
  matrix[N, num_media] X_media_adstocked;
  // matrix of all predictors
  matrix[N, num_media+num_ctrl] X;
  
  // adstock, mean-center, log1p transformation
  row_vector[max_lag] lag_weights;
  for (nn in 1:N) {
    for (media in 1 : num_media) {
      for (lag in 1 : max_lag) {
        lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1 - peak[media]) ^ 2);
      }
     cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
     X_media_adstocked[nn, media] <- log1p(cum_effect/mu_mdip[media]);
    }
  X <- append_col(X_media_adstocked, X_ctrl);
  } 
}
model {
  decay ~ beta(3,3);
  peak ~ uniform(0, ceil(max_lag/2));
  tau ~ normal(0, 5);
  for (i in 1 : num_media+num_ctrl) {
    beta[i] ~ normal(0, 1);
  }
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  y ~ normal(tau + X * beta, sqrt(noise_var));
}
'''

sm2 = pystan.StanModel(model_code=model_code2, verbose=True)
fit2 = sm2.sampling(data=model_data2, iter=1000, chains=3)
fit2_result = fit2.extract()
```


```python
# extract mmm parameters
def extract_mmm(fit_result, max_lag=max_lag, 
                media_vars=mdip_cols, ctrl_vars=['base_sales'], 
                extract_param_list=True):
    mmm = {}
    
    mmm['max_lag'] = max_lag
    mmm['media_vars'], mmm['ctrl_vars'] = media_vars, ctrl_vars
    mmm['decay'] = decay = fit_result['decay'].mean(axis=0).tolist()
    mmm['peak'] = peak = fit_result['peak'].mean(axis=0).tolist()
    mmm['beta'] = fit_result['beta'].mean(axis=0).tolist()
    mmm['tau'] = fit_result['tau'].mean()
    if extract_param_list:
        mmm['decay_list'] = fit_result['decay'].tolist()
        mmm['peak_list'] = fit_result['peak'].tolist()
        mmm['beta_list'] = fit_result['beta'].tolist()
        mmm['tau_list'] = fit_result['tau'].tolist()
    
    adstock_params = {}
    media_names = [col.replace('mdip_', '') for col in media_vars]
    for i in range(len(media_names)):
        adstock_params[media_names[i]] = {
            'L': max_lag,
            'P': peak[i],
            'D': decay[i]
        }
    mmm['adstock_params'] = adstock_params
    return mmm

mmm = extract_mmm(fit2, max_lag=max_lag, media_vars=mdip_cols, ctrl_vars=['base_sales'])
```

**Distribution of Media Coefficients**    
red line: mean, green line: median    
![media coefficients distribution](img/mmm_stan_media_coefficient.png)



### Decompose sales to media channels' contribution

Each media channel's contribution = total sales - sales upon removal of the channel    
In the previous model fitting step, parameters of the log-log model have been found:    
![mmm_stan_decompose_contrib1](img/mmm_stan_decompose_contrib1.png)    
Plug them into the multiplicative model:    
![mmm_stan_decompose_contrib2](img/mmm_stan_decompose_contrib2.png)    
![mmm_stan_decompose_contrib3](img/mmm_stan_decompose_contrib3.png)    


```python
# decompose sales to media contribution
def mmm_decompose_contrib(mmm, df, original_sales=df['sales']):
    adstock_params = mmm['adstock_params']
    beta, tau = mmm['beta'], mmm['tau']
    media_vars, ctrl_vars = mmm['media_vars'], mmm['ctrl_vars']
    num_media, num_ctrl = len(media_vars), len(ctrl_vars)
    
    # X_media2: adstocked, mean-centered media variables + 1
    X_media2 = adstock_transform(df, media_vars, adstock_params)
    X_media2, sc_mmm2 = mean_center_trandform(X_media2, media_vars)
    X_media2 = X_media2 + 1
    # X_ctrl2, mean-centered control variables + 1
    X_ctrl2, sc_mmm2_1 = mean_center_trandform(df[ctrl_vars], ctrl_vars)
    X_ctrl2 = X_ctrl2 + 1
    # y_true2, mean-centered sales variable + 1
    y_true2, sc_mmm2_2 = mean_center_trandform(df, ['sales'])
    y_true2 = y_true2 + 1
    sc_mmm2.update(sc_mmm2_1)
    sc_mmm2.update(sc_mmm2_2)
    X2 = pd.concat([X_media2, X_ctrl2], axis=1)

    # 1. compute each media/control factor: 
    # log-log model: log(sales) = log(X[0])*beta[0] + ... + log(X[13])*beta[13] + tau
    # multiplicative model: sales = X[0]^beta[0] * ... * X[13]^beta[13] * e^tau
    # each factor = X[i]^beta[i]
    # intercept = e^tau
    factor_df = pd.DataFrame(columns=media_vars+ctrl_vars+['intercept'])
    for i in range(num_media):
        colname = media_vars[i]
        factor_df[colname] = X[colname] ** beta[i]
    for i in range(num_ctrl):
        colname = ctrl_vars[i]
        factor_df[colname] = X[colname] ** beta[num_media+i]
    factor_df['intercept'] = np.exp(tau)

    # 2. calculate the product of all factors -> y_pred
    y_pred = factor_df.apply(np.prod, axis=1)
    factor_df['y_pred'], factor_df['y_true2'] = y_pred, y_true2
    # baseline = intercept * control factor
    factor_df['baseline'] = factor_df[['intercept']+ctrl_vars].apply(np.prod, axis=1)

    # 3. calculate each media factor's contribution
    # media contribution = total sales – sales upon removal of the media factor
    mc_df = pd.DataFrame(columns=media_vars+['baseline'])
    for col in media_vars:
        mc_df[col] = factor_df['y_true2'] - factor_df['y_true2']/factor_df[col]
    mc_df['baseline'] = factor_df['baseline']
    mc_df['y_true2'] = factor_df['y_true2']

    # 4. scale contribution
    # predicted total media contribution: product of all media factors
    mc_df['mc_pred'] = mc_df[media_vars].apply(np.sum, axis=1)
    # true total media contribution: total volume - baseline
    mc_df['mc_true'] = mc_df['y_true2'] - mc_df['baseline']
    # predicted total media contribution is slightly different from true total media contribution
    # scale each media factor’s contribution by removing the delta volume proportionally
    mc_df['mc_delta'] = mc_df['mc_true'] - mc_df['mc_pred']
    for col in media_vars:
        mc_df[col] = mc_df[col] - mc_df['mc_delta']*mc_df[col]/mc_df['mc_pred']

    # 5. scale mc_df based on original sales
    mc_df['sales'] = original_sales
    for col in media_vars+['baseline']:
        mc_df[col] = mc_df[col]*mc_df['sales']/mc_df['y_true2']
    return mc_df

def calc_media_contrib_pct(mc_df, media_vars=mdip_cols, sales_col='sales', period=52):
    '''
    returns:
    mc_pct: percentage over total sales
    mc_pct2: percentage over incremental sales (sales contributed by media channels)
    '''
    mc_pct = {}
    mc_pct2 = {}
    s = 0
    if period is None:
        for col in (media_vars+['baseline']):
            mc_pct[col] = (mc_df[col]/mc_df[sales_col]).mean()
    else:
        for col in (media_vars+['baseline']):
            mc_pct[col] = (mc_df[col]/mc_df[sales_col])[-period:].mean()
    for m in media_vars:
        s += mc_pct[m]
    for m in media_vars:
        mc_pct2[m] = mc_pct[m]/s
    return mc_pct, mc_pct2

mc_df = mmm_decompose_media_contrib(mmm, df, y_true=df['sales_ln'])
adstock_params = mmm['adstock_params']
mc_pct, mc_pct2 = calc_media_contrib_pct(mc_df, period=52)
```

RMSE (log-log model):  0.04977    
MAPE (multiplicative model):  15.71%    
    
**Adstock Parameters**    
```python
{'dm': {'L': 8, 'P': 0.8147057071636012, 'D': 0.5048365638721349},
 'inst': {'L': 8, 'P': 0.6339321363933637, 'D': 0.40532404247040194},
 'nsp': {'L': 8, 'P': 1.1076944292039324, 'D': 0.4612905130128658},
 'auddig': {'L': 8, 'P': 1.8834110997525702, 'D': 0.5117823761413419},
 'audtr': {'L': 8, 'P': 1.9892680621155827, 'D': 0.5046141055524362},
 'vidtr': {'L': 8, 'P': 0.05520253973872224, 'D': 0.0846136627657064},
 'viddig': {'L': 8, 'P': 1.862571613911107, 'D': 0.5074553132446618},
 'so': {'L': 8, 'P': 1.7027472358912694, 'D': 0.5046386226501091},
 'on': {'L': 8, 'P': 1.4169662215350334, 'D': 0.4907407637366824},
 'em': {'L': 8, 'P': 1.0590065753144235, 'D': 0.44420264450045377},
 'sms': {'L': 8, 'P': 1.8487648735160152, 'D': 0.5090970201714644},
 'aff': {'L': 8, 'P': 0.6018657109295106, 'D': 0.39889023002777724},
 'sem': {'L': 8, 'P': 1.34945185610011, 'D': 0.47875793676213835}}
```
**Notes**:
- For SEM, P=1.3, D=0.48 does not make a lot of sense to me, because SEM is expected to have immediate and concentrated impact (P=0, low decay). Same with online display.    
- Try more specific priors in future model.

​    

## 2.3 Diminishing Return Model    

**Goal**: for each channel, find the relationship (fit a Hill function) between spending and contribution, so that ROAS and marginal ROAS can be calculated.    
![diminishing return model formular](img/mmm_stan_model_return.png)    
x: adstocked media channel spending   
K: half saturation    
S: shape    
Target variable: the media channel's contribution    
Variables are centralized by mean.
    
**Priors**    
![diminishing return model priors](img/mmm_stan_model_return_prior.png)          
**Implementation**    


```python
def create_hill_model_data(df, mc_df, adstock_params, media):
    y = mc_df['mdip_'+media].values
    L, P, D = adstock_params[media]['L'], adstock_params[media]['P'], adstock_params[media]['D']
    x = df['mdsp_'+media].values
    x_adstocked = apply_adstock(x, L, P, D)
    # centralize
    mu_x, mu_y = x_adstocked.mean(), y.mean()
    sc = {'x': mu_x, 'y': mu_y}
    x = x_adstocked/mu_x
    y = y/mu_y
        
    model_data = {
        'N': len(y),
        'y': y,
        'X': x
    }
    return model_data, sc

model_code3 = '''
functions {
  // the Hill function
  real Hill(real t, real ec, real slope) {
  return 1 / (1 + (t / ec)^(-slope));
  }
}

data {
  // the total number of observations
  int<lower=1> N;
  // y: vector of media contribution
  vector[N] y;
  // X: vector of adstocked media spending
  vector[N] X;
}

parameters {
  // residual variance
  real<lower=0> noise_var;
  // regression coefficient
  real<lower=0> beta_hill;
  // ec50 and slope for Hill function of the media
  real<lower=0,upper=1> ec;
  real<lower=0> slope;
}

transformed parameters {
  // a vector of the mean response
  vector[N] mu;
  for (i in 1:N) {
    mu[i] <- beta_hill * Hill(X[i], ec, slope);
  }
}

model {
  slope ~ gamma(3, 1);
  ec ~ beta(2, 2);
  beta_hill ~ normal(0, 1);
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01); 
  y ~ normal(mu, sqrt(noise_var));
}
'''

# pipeline for training one hill model for a media channel
def train_hill_model(df, mc_df, adstock_params, media, sm):
    data, sc = create_hill_model_data(df, mc_df, adstock_params, media)
    fit = sm.sampling(data=data, iter=2000, chains=4)
    fit_result = fit.extract()
    hill_model = {
        'beta_hill_list': fit_result['beta_hill'].tolist(),
        'ec_list': fit_result['ec'].tolist(),
        'slope_list': fit_result['slope'].tolist(),
        'sc': sc,
        'data': {
            'X': data['X'].tolist(),
            'y': data['y'].tolist(),
        }
    }
    return hill_model
```


```python
# extract params
def extract_hill_model_params(hill_model, method='mean'):
    if method=='mean':
        hill_model_params = {
            'beta_hill': np.mean(hill_model['beta_hill_list']), 
            'ec': np.mean(hill_model['ec_list']), 
            'slope': np.mean(hill_model['slope_list'])
        }
    elif method=='median':
        hill_model_params = {
            'beta_hill': np.median(hill_model['beta_hill_list']), 
            'ec': np.median(hill_model['ec_list']), 
            'slope': np.median(hill_model['slope_list'])
        }
    return hill_model_params

def hill_model_predict(hill_model_params, x):
    beta_hill, ec, slope = hill_model_params['beta_hill'], hill_model_params['ec'], hill_model_params['slope']
    y_pred = beta_hill * hill_transform(x, ec, slope)
    return y_pred

# train hill models for all media channels
sm3 = pystan.StanModel(model_code=model_code3, verbose=True)
hill_models = {}
to_train = ['dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem']
for media in to_train:
    print('training for media: ', media)
    hill_model = train_hill_model(df, mc_df, adstock_params, media, sm3)
    hill_models[media] = hill_model

# extract params by mean
hill_model_params_mean, hill_model_params_med = {}, {}
for md in list(hill_models.keys()):
    hill_model = hill_models[md]
    params1 = extract_hill_model_params(hill_model, method='mean')
    params1['sc'] = hill_model['sc']
    hill_model_params_mean[md] = params1
```

**Distribution of K (Half Saturation Point)**    
![half saturation distribution](img/mmm_stan_ec.png)    
**Distribution of S (Slope)**    
![slope distribution](img/mmm_stan_slope.png)    
**Diminishing Return Model (Fitted Hill Curve)**    
![fitted hill](img/mmm_stan_hill.png)    



### Calculate overall ROAS and weekly ROAS
- Overall ROAS = total media contribution / total media spending
- Weekly ROAS = weekly media contribution / weekly media spending


```python
# adstocked media spending
ms_df = pd.DataFrame()
for md in list(hill_models.keys()):
    hill_model = hill_models[md]
    x = np.array(hill_model['data']['X']) * hill_model['sc']['x']
    ms_df['mdsp_'+md] = x

# calc overall ROAS of a given period
def calc_roas(mc_df, ms_df, period=None):
    roas = {}
    md_names = [col.split('_')[-1] for col in ms_df.columns]
    for i in range(len(md_names)):
        md = md_names[i]
        sp, mc = ms_df['mdsp_'+md], mc_df['mdip_'+md]
        if period is None:
            md_roas = mc.sum()/sp.sum()
        else:
            md_roas = mc[-period:].sum()/sp[-period:].sum()
        roas[md] = md_roas
    return roas

# calc weekly ROAS
def calc_weekly_roas(mc_df, ms_df):
    weekly_roas = pd.DataFrame()
    md_names = [col.split('_')[-1] for col in ms_df.columns]
    for md in md_names:
        weekly_roas[md] = mc_df['mdip_'+md]/ms_df['mdsp_'+md]
    weekly_roas.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return weekly_roas

roas_1y = calc_roas(mc_df, ms_df, period=52)
weekly_roas = calc_weekly_roas(mc_df, ms_df)
roas1y_df = pd.DataFrame(index=weekly_roas.columns.tolist())
roas1y_df['roas_mean'] = weekly_roas[-52:].apply(np.mean, axis=0)
roas1y_df['roas_median'] = weekly_roas[-52:].apply(np.median, axis=0)
```

**Distribution of Weekly ROAS** (Recent 1 Year)    
red line: mean, green line: median
![weekly roas](img/mmm_stan_dist_roas_1y.png)



### Calculate mROAS
1. Current spending level ```cur_sp``` is represented by mean or median of weekly spending.    
Next spending level ```next_sp``` is increasing ```cur_sp``` by 1%.
2. Plug ```cur_sp``` and ```next_sp``` into the Hill function:    
Current media contribution ```cur_mc``` = Hill(```cur_sp```)    
Next-level media contribution ```next_mc``` = Hill(```next_sp```)    
3. **mROAS** = (```next_mc``` - ```cur_mc```) / (0.01 * ```cur_sp```)


```python
def calc_mroas(hill_model, hill_model_params, method='median', period=52):
    '''
    calculate mROAS for a media channel in a given period
    params:
    hill_model: a dict containing model data and scaling factor
    hill_model_params: a dict containing beta_hill, ec, slope
    method: the way to represent current weekly spending level: 'mean', 'median'. 
        default median, since spending tends to be right-skewed.
    period: in weeks, the period used to calculate ROAS and mROAS. 52 is last one year.
    return:
    mROAS value
    '''
    mu_x, mu_y = hill_model['sc']['x'], hill_model['sc']['y']
    # get current media spending level over the period specified
    if period is None:
        if method=='median':
            cur_sp = np.median(hill_model['data']['X'])
        elif method=='mean':
            cur_sp = np.mean(hill_model['data']['X'])
    else:
        if method=='median':
            cur_sp = np.median(hill_model['data']['X'][-period:])
        elif method=='mean':
            cur_sp = np.mean(hill_model['data']['X'][-period:])
    # media contribution under current spending level
    cur_mc = hill_model_predict(hill_model_params, cur_sp) * mu_y
    # next spending level: increase by 1%
    next_sp = cur_sp * 1.01
    # media contribution under next spending level
    next_mc = hill_model_predict(hill_model_params, next_sp) * mu_y
    
    # mROAS
    delta_mc = next_mc - cur_mc
    delta_sp = cur_sp * 0.01 * mu_x
    mroas = delta_mc/delta_sp
    return mroas

# calc mROAS based on mean and median of weekly spending
mroas_mean, mroas_med = {}, {}
for md in list(hill_models.keys()):
    hill_model = hill_models[md]
    hill_model_params = hill_model_params_mean[md]
    mroas_mean[md] = calc_mroas(hill_model, hill_model_params, method='mean', period=52)
    mroas_med[md] = calc_mroas(hill_model, hill_model_params, method='median', period=52)
```

**ROAS & mROAS**    
'roas_avg': overall ROAS = total contribution / total spending    
'roas_mean': mean of weekly ROAS    
'roas_median': median of weekly ROAS    
'mroas_mean': mROAS calculated based on mean of weekly spending as current spending level    
'mroas_median': mROAS calculated based on median of weekly spending as current spending level    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roas_mean</th>
      <th>roas_median</th>
      <th>mroas_mean</th>
      <th>mroas_median</th>
      <th>roas_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dm</th>
      <td>2.551370</td>
      <td>2.412951</td>
      <td>4.740435</td>
      <td>4.740438</td>
      <td>2.619002</td>
    </tr>
    <tr>
      <th>inst</th>
      <td>5.378604</td>
      <td>5.060652</td>
      <td>10.096348</td>
      <td>9.348433</td>
      <td>5.852283</td>
    </tr>
    <tr>
      <th>nsp</th>
      <td>6.157474</td>
      <td>4.911293</td>
      <td>7.138000</td>
      <td>6.888607</td>
      <td>8.177945</td>
    </tr>
    <tr>
      <th>auddig</th>
      <td>20.562877</td>
      <td>18.291145</td>
      <td>14.474924</td>
      <td>16.671421</td>
      <td>20.621256</td>
    </tr>
    <tr>
      <th>audtr</th>
      <td>4.547045</td>
      <td>3.725285</td>
      <td>6.489088</td>
      <td>7.003847</td>
      <td>4.480175</td>
    </tr>
    <tr>
      <th>vidtr</th>
      <td>14.669730</td>
      <td>12.596672</td>
      <td>15.470877</td>
      <td>16.400834</td>
      <td>11.044632</td>
    </tr>
    <tr>
      <th>viddig</th>
      <td>3.354704</td>
      <td>3.027100</td>
      <td>4.460041</td>
      <td>5.457326</td>
      <td>3.665650</td>
    </tr>
    <tr>
      <th>so</th>
      <td>2.553423</td>
      <td>2.480701</td>
      <td>1.488556</td>
      <td>1.792750</td>
      <td>2.540194</td>
    </tr>
    <tr>
      <th>on</th>
      <td>4.660522</td>
      <td>4.254862</td>
      <td>5.927870</td>
      <td>6.575460</td>
      <td>4.831279</td>
    </tr>
    <tr>
      <th>sem</th>
      <td>2.102519</td>
      <td>2.131076</td>
      <td>3.114688</td>
      <td>4.646537</td>
      <td>2.062126</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Results & Marketing Budget Optimization    
**Media Channel Contribution**    
80% sales are contributed by non-marketing factors, marketing channels contributed 20% sales.    
![marketing contribution plot](img/mmm_stan_media_contribution.png)    
Top contributors: TV, affiliates, SEM    
![media contribution percentage plot](img/mmm_stan_media_contribution_pct.png)    
**ROAS**    
High ROAS: TV, insert, online display    
![media channels contribution roas plot](img/mmm_stan_media_contrib_roas.png)    
**mROAS**    
High mROAS: TV, insert, radio, online display    
![media channels roas mroas plot](img/mmm_stan_media_roas_mroas.png)    
Note: trivial channels: newspaper, digital audio, digital video, social (spending/impression too small to be qualified, so that their results are not trustworthy).



## References

[1] Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf    
[2] STAN tutorials:    
Prior Choice Recommendations. https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations    
https://www.cnpython.com/pypi/pystan    
https://mc-stan.org/users/documentation/case-studies/pystan_workflow.html    
https://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb    
HMC sampling: https://education.illinois.edu/docs/default-source/carolyn-anderson/edpsy590ca/lectures/9-hmc-and-stan/hmc_n_stan_post.pdf       



**Thanks for reading! If you like this project, please leave a :star2: :)**
