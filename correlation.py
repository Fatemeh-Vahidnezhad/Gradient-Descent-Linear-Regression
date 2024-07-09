import numpy as np 
import pandas as pd
from scipy.stats import t


def mean(data):
    # return np.nanmean(data)
    return np.mean(data)


def corr(data, target):
    # mask = ~np.isnan(data) & ~np.isnan(target)
    # data = data[mask]
    # target = target[mask]
    n = len(data)
    if len(data) == 0 or len(target) == 0:
        return np.nan, np.nan  # Return NaN if no valid data pairs
    
    mean_data = mean(data)
    mean_targ = mean(target)
    numerator = np.sum((data - mean_data) * (target - mean_targ))
    std_data = np.sqrt(np.sum((data - mean_data)**2))
    std_target = np.sqrt(np.sum((target - mean_targ)**2))
    
    if std_data == 0 or std_target == 0:
        return np.nan, np.nan  # Return NaN if the standard deviation is zero (division by zero)
    
    denominator = std_data * std_target
    Correlation = numerator/denominator

    if Correlation == 1 or Correlation == -1:
        return Correlation, 0.0 
    
    # Compute the t-statistic
    t_stat_denominator = (1 - Correlation**2)
    if t_stat_denominator <= 0:
        return Correlation, np.nan  # Return NaN if invalid t-statistic calculation
    
    t_stat = Correlation * np.sqrt((n - 2)/t_stat_denominator)
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))
    if p_value <=0.05:
        return Correlation, p_value
    else:
        return Correlation, 1


def corr_matrix(df):
    data = np.array(df)
    correlation_matrix = np.zeros((data.shape[1], data.shape[1]))
    pvalue_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            data_col = data[:, i]
            target = data[:, j]
            correlation_matrix[i, j], pvalue_matrix[i, j] = corr(data_col, target)
    return (
        pd.DataFrame(correlation_matrix, columns=df.columns, index=df.columns),
        pd.DataFrame(pvalue_matrix, columns=df.columns, index=df.columns)
    )
            


path = 'C:/Users/Fatemeh/Documents/Online_books_stanford/datasets/keloid.csv'
df = pd.read_csv(path)
df = df.fillna(0)
df = df[['KELOID', 'FEVER'
         , 'DIABET', 'ADHD', 'WART', 'HEADACHE',
       'HYPERTENSION', 'GERD', 'FAILURE TO THRIVE', 'Elevated blood pressure',
       'otitis', 'SLEEP', 'Anxiety', 'Hypertrophic', 'stroke', 'ARTHR',
       'fibrosis', 'Osteopenia', 'Atopic dermatitis', 'Acne vulgaris',
       'ACUTE PHARYNGITIS', 'OTHER CONVULSIONS', 'Feeding difficulties',
       'FAMILY HISTORY OF ASTHMA', 'Allergic rhinitis', 'Abnormality of gait',
       'ACUTE URI NOS', 'hearing loss', 'Sickle cell disease', 'ASTHMA',
       'Eczema', 'CONGENITAL DIPLEGIA', 'obesity_bmi', 'Scoliosis',
       'genetic disease', 'Vitamin D', 'COUGH', 'Pharyngitis', 'Vitamin', 'AGE'
    ]]
# print(df.head())
(correlation_matrix, pvalue_matrix) = corr_matrix(df)
# print(result)
correlation_matrix.to_csv('correlation_matrix_0.csv')
pvalue_matrix.to_csv('pvalue_matrix_0.csv')

# mask = ~np.isnan(df['KELOID']) & ~np.isnan(df['FEVER'])
# data = df[mask]
# print(data)
