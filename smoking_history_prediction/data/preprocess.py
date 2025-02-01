import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Load raw data from CSV."""
    return pd.read_csv(filepath)

def clean_and_process_data(df):
    df = df.drop_duplicates()

    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])
    df["DRK_YN"] = le.fit_transform(df["DRK_YN"])

    for i in df.columns:
        df[i] = df[i].astype('float64')
    
    df.loc[df["SMK_stat_type_cd"] == 3.0, "SMK_stat_type_cd"] = 0.0
    df.loc[df["SMK_stat_type_cd"] == 2.0, "SMK_stat_type_cd"] = 0.0
    
    scaler = StandardScaler()
    columns_to_be_scaled = ['age','height','weight','waistline','sight_left','sight_right','hear_left','hear_right',
                            'SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole','triglyceride','hemoglobin','urine_protein',
                            'serum_creatinine','SGOT_AST','SGOT_ALT','gamma_GTP']
    df[columns_to_be_scaled] = scaler.fit_transform(df[columns_to_be_scaled])

    X = df.drop("SMK_stat_type_cd", axis=1)
    y = df["SMK_stat_type_cd"]
    undersampler = RandomUnderSampler(sampling_strategy=1)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["SMK_stat_type_cd"] = y_resampled

    return df_resampled

def save_data(df, filepath):
    df.to_csv(filepath, index=False)
