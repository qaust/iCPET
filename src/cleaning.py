import numpy as np
import pandas as pd
from pathlib import Path
from config import model_config


SEED = 123
np.random.seed(seed=SEED)
rand_day_offset = np.random.randint(1, 10_000)
DATE_OFFSET = pd.Timedelta(rand_day_offset, 'days')
print(f"Date offset: {DATE_OFFSET}")


def get_n_smallest_date(data, n):
	"""
	:param data: Date column.
	:param n: nth smallest date to return.
	:return: nth smallest date.
	"""
	if data.shape[-1] < n:
		return pd.NaT
	else:
		return data.nsmallest(n, keep='all')[-1]

def calculate_percent_change(data, initial_index, final_index):
	"""
	:param data: Data for group.
	:param initial_index: First patient observation for percent change calculation.
	:param final_index: Last patient observation for percent change calculation.
	:return: Percent change, if two conditions are satisfied:
		- Data exists (first logical condition).
		- Initial observation is not zero (second logical condition) to avoid div by zero.
	"""
	if len(data) > final_index and data.iloc[initial_index] != 0:
		return (data.iloc[final_index] / data.iloc[initial_index]) - 1
	else:
		return np.nan
	
def count_positive_values(row):
    """
    :param row: Row of values to count over.
    :return: Count of positive values. 
    """
    return sum(row > 0)

def process_clinical_data():
    """Process iCPET data"""
    clinical = pd.read_csv(
        Path('../data/iCPET_data_6-28-2023.csv'),
        parse_dates=True
    )

    vq_dates = pd.read_csv(
        Path('../data/vq_dates.csv'),
        parse_dates=True
    )
    vq_dates['vq_date'] = pd.to_datetime(vq_dates['vq_date'])

    # Clean column names
    clinical.columns = (
        clinical.columns
        .str.lower()
        .str.replace(r' +|/', r'_', regex=True)
        .str.replace(r'\(|\)|-', r'', regex=True)
        .str.replace('%_', 'percent_', regex=True)
        .str.replace('%', 'percent_', regex=True)
        .str.replace('1st', 'first')
    )

    # Make what should be nan values nan values
    clinical = clinical.replace('na', np.nan)
    clinical = clinical.replace(' ', np.nan)
    clinical['hyperventilation'].fillna(0, inplace=True)

    # Join VQ dates to clinical data
    clinical = clinical.join(
        other=vq_dates.set_index('pe_number_clean'),
        on='pe_number_clean',
        validate='1:1'
    )

    # Set correct dtypes
    clinical.gender = (
        clinical.gender
        .replace({0: 'Female', 1: 'Male'})
        .astype('category')
    )
    clinical.race = (
        clinical.race
        .replace({0: 'White', 1: 'Non-White', 2: 'Non-White'})
        .astype('category')
    )
    clinical.resolved_pe = (
        clinical.resolved_pe
        .replace({0: 'Unresolved', 1: 'Resolved'})
        .astype('category')
    )

    clinical.enrollment_id =                    pd.to_numeric(clinical.enrollment_id)
    clinical.ph_id =                            pd.to_numeric(clinical.ph_id)
    clinical.first_icpet =                      pd.to_datetime(clinical.first_icpet)
    clinical.pe_study_number =                  clinical.pe_study_number.str.replace('PE ', 'PE')
    clinical.dob =                              pd.to_datetime(clinical.dob)

    clinical.normal_study =                     clinical.normal_study.astype('category')
    clinical.resting_pah =                      clinical.resting_pah.astype('category')
    clinical.borderline_ph =                    clinical.borderline_ph.astype('category')
    clinical.resting_hfpef =                    clinical.resting_hfpef.astype('category')
    clinical.eph =                              clinical.eph.astype('category')
    clinical.exercise_hfpef =                   clinical.exercise_hfpef.astype('category')
    clinical.resting_ph_exercise_hfpef =        clinical.resting_ph_exercise_hfpef.astype('category')
    clinical.exercise_ph_resting_hfpef =        clinical.exercise_ph_resting_hfpef.astype('category')
    clinical.deconditioning =                   clinical.deconditioning.astype('category')
    clinical.preload_insufficiency =            clinical.preload_insufficiency.astype('category')
    clinical.inappropriate_o2_extraction =      clinical.inappropriate_o2_extraction.astype('category')
    clinical.systemic_htn_response =            clinical.systemic_htn_response.astype('category')
    clinical.approached_ventilatory_ceiling =   clinical.approached_ventilatory_ceiling.astype('category')
    clinical.surpassed_ventilatory_ceiling =    clinical.surpassed_ventilatory_ceiling.astype('category')
    clinical.hyperventilation =                 clinical.hyperventilation.astype('category')

    clinical['hyperventilation_num'] =          pd.to_numeric(clinical.hyperventilation)
    clinical.peak_pvr_wu =                      pd.to_numeric(clinical.peak_pvr_wu)
    clinical.peak_vo2_ml_min =                  clinical.peak_vo2_ml_min.astype(np.float64)
    clinical.estimated_peak_vo2_ml_min =        clinical.estimated_peak_vo2_ml_min.astype(np.float64)
    clinical.vo2_hr_peak_percent_  =            clinical.vo2_hr_peak_percent_.astype(np.float64)

    clinical['mpap_co_ratio'] =                 clinical['peak_measured_mpap_mmhg'] / clinical['peak_fick_co']

    # Create categorical column 'normal' where
    clinical['normal'] = -1
    clinical.loc[clinical.percent_peak_vo2 > 80, 'normal'] = 1
    clinical.loc[clinical.percent_peak_vo2 <= 80, 'normal'] = 0
    clinical.normal = clinical.normal.astype('category')

    # Mask/offset dates
    clinical['first_icpet_mask'] = clinical.first_icpet + DATE_OFFSET
    clinical['vq_date'] = clinical.vq_date + DATE_OFFSET
    clinical['dob_mask'] = clinical.dob + DATE_OFFSET
    clinical['study_age'] = (clinical.first_icpet_mask - clinical.dob_mask) / pd.Timedelta(365.25, 'days')
    clinical['study_date_mask'] = clinical.first_icpet_mask

    # Drop irrelevant columns
    drop_cols = [
        'icpet_study_results',
        'enrollment_id',
        'ph_id',
        'icpet_id',
        'first_icpet',
        'dob',
        'first_icpet_mask',
    ]
    clinical = clinical.drop(columns=drop_cols).copy()

    # Sort values
    clinical = clinical.sort_values(by=['pe_number_clean', 'study_date_mask'])

    # Reset index
    clinical = clinical.reset_index(drop=True)

    return clinical


def process_ct_data():
    """Process radiographic data."""

    pe = pd.read_csv(
        Path('../data/PE-corrected.csv'),
        parse_dates=True
    )

    # Clean column names
    pe.columns = (
        pe.columns
        .str.lower()
        .str.replace(r' +|/|\+', r'_', regex=True)
        .str.replace(r'\(|\)|-', r'', regex=True)
    )

    # Rename columns
    pe = pe.rename({'patient_name': 'pe_number_clean'}, axis=1)

    print(f"Original shape: {pe.shape}")

    # Keep only relevant series
    series_2 = (pe['series_id'] == 2)
    series_3 = (pe['series_id'] == 3)
    series_4 = (pe['series_id'] == 4)
    series_5 = (pe['series_id'] == 5)
    series_6 = (pe['series_id'] == 6)
    series_to_keep = series_2 | series_3 | series_4 | series_5 | series_6
    print(f"# to keep (correct series): {sum(series_to_keep)}")
    pe = pe.loc[series_to_keep, :]

    print(f"Shape after removing series: {pe.shape}")

    # De-duplicate on patient id and study date taking the scan with the higher number of slices
    pe = pe.sort_values(by='series_id', ascending=False)
    pe = pe.drop_duplicates(subset=['pe_number_clean', 'study_date'], keep='first')

    print(f"Shape after removing depulicating (pe number and study date): {pe.shape}")

    print(sorted(pe.pe_number_clean.unique()))

    # Convert to datetime
    pe.study_date = pd.to_datetime(pe.study_date, format='%Y%m%d')
    pe.dob = pd.to_datetime(pe.dob, format='%Y%m%d')

    # Offset dates
    pe['study_date_mask'] = pe.study_date + DATE_OFFSET
    pe['dob_mask'] = pe.dob + DATE_OFFSET

    # Calculate age as of study date
    pe['study_age'] = (pe.study_date_mask - pe.dob_mask) / pd.Timedelta(365.25, 'days')

    # Calculate total and lobe-level clot burden as sum
    pe['total_clot_burden'] = pe[model_config.all_segments].sum(axis=1)
    pe['superior_right']    = pe[model_config.superior_right_segments].sum(axis=1)
    pe['superior_left']     = pe[model_config.superior_left_segments].sum(axis=1)
    pe['middle_right']      = pe[model_config.middle_right_segments].sum(axis=1)
    pe['inferior_right']    = pe[model_config.inferior_right_segments].sum(axis=1)
    pe['inferior_left']     = pe[model_config.inferior_left_segments].sum(axis=1)

    # Drop original dates and age columns
    pe = pe.drop(columns=['study_date', 'dob', 'age'])

    # Get cumulative counts for each patient
    pe = pe.sort_values(by=['pe_number_clean', 'study_date_mask'])
    pe['pe_obs'] = pe.groupby('pe_number_clean', group_keys=False).cumcount()

    # Reset index
    pe = pe.reset_index(drop=True)

    return pe
    


def combine_data():

    clinical = process_clinical_data()
    clinical.to_pickle(Path('../data/clinical.pkl'))
    pe = process_ct_data()
    pe.to_pickle(Path('../data/pe.pkl'))

    df = pe.join(
        other=clinical.set_index('pe_number_clean'),
        on='pe_number_clean',
        how='outer',
        lsuffix='_pe',
        rsuffix='_cl',
        validate='m:1'
    )

    print(f"Unique PE numbers:\n{df.pe_number_clean.unique()}\n")
    print(df.shape)
    # An NaN in pe_obs indicates that the observation was not in the CT data
    # Fill with a zero so it will still get included in summary statistics
    df.pe_obs.fillna(0, inplace=True)
    print(df.shape)

    pe_all = set(pe.pe_number_clean.unique())
    icpet_all = set(clinical.pe_number_clean.unique())
    assert pe_all.intersection(icpet_all) == icpet_all.intersection(pe_all)
    both = pe_all.intersection(icpet_all)
    ct_only = pe_all.difference(icpet_all)
    icpet_only = icpet_all.difference(pe_all)

    print(f"# Unique PE numbers (CT data): {len(pe.pe_number_clean.unique())}")
    print(f"# Unique PE numbers (iCPET data): {len(clinical.pe_number_clean.unique())}")
    print(f"# Unique PE numbers (overall): {len(df.pe_number_clean.unique())}")
    print(f"\nPE numbers in BOTH ({len(both)}):\n{list(sorted(both))}")
    print(f"\nPE numbers in CT DATA ONLY ({len(ct_only)}):\n{sorted(ct_only)}")
    print(f"\nPE numbers in iCPET DATA ONLY ({len(icpet_only)}):\n{sorted(icpet_only)}")

    # Set index to PE number
    df.index = df.pe_number_clean + '_' + df.pe_obs.apply(int).apply(str)

    # Create columns for the nth smallest date
    for obs_number in range(1, int(df.pe_obs.max())+1):
        df[f"date_of_obs_{int(obs_number-1)}"] = (
            df
            .groupby('pe_number_clean')['study_date_mask_pe']
            .transform(lambda x: get_n_smallest_date(x, obs_number))
        )
        
    # Calculate durations from date to date and percent changes
    # Store results in a dictionary to then concat with the original df
    # (see https://github.com/pandas-dev/pandas/issues/42477)
    dict_of_columns = dict()
    for start_obs in range(0, int(df.pe_obs.max())-1):
        for end_obs in range(1, int(df.pe_obs.max())):
            if start_obs >= end_obs:
                continue
            # Calculate durations
            dict_of_columns[f'duration_{start_obs}_to_{end_obs}_days'] = (
                (df[f'date_of_obs_{end_obs}'] - df[f'date_of_obs_{start_obs}']).dt.days
            )
            # Calculate percent changes
            for feature in model_config.num_targets:
                dict_of_columns[f'pct_change_{feature}_{start_obs}_to_{end_obs}'] = (
                    df
                    .groupby('pe_number_clean')[feature]
                    .transform(calculate_percent_change, start_obs, end_obs)
                )
    df = pd.concat([df, pd.DataFrame(dict_of_columns)], axis=1)

    # Calculate percent change per day
    df['pct_change_total_clot_burden_per_day_0_to_1'] = df['pct_change_total_clot_burden_0_to_1'] / df['duration_0_to_1_days']

    # Get count of positive clot burden values across all segments 
    # Meant to give a sense of the distribution of clot burden
    segment_clot_burden_columns = list(set(model_config.num_targets).difference(set(['total_clot_burden'])))
    df['num_positive_clot_burden_segments'] = df[segment_clot_burden_columns].apply(count_positive_values, axis=1)

    # Check shape
    print(f"New shape: {df.shape}")

    # Number of scans
    df['num_scans'] = df.groupby('pe_number_clean')['total_clot_burden'].transform('count')

    # Get total number of visits by PE_number
    # Can use later to subset into patients with multiple observations.
    df['total_visits'] = df.groupby('pe_number_clean')['pe_number_clean'].transform('count')

    # Compute number of days since first observation
    # Can use later to line up timelines rather than use masked (meaningless) dates.
    df = df.sort_values(by='study_date_mask_pe')
    df['date_first_visit'] = df.groupby('pe_number_clean')['study_date_mask_pe'].transform(min)
    df['date_last_visit'] = df.groupby('pe_number_clean')['study_date_mask_pe'].transform(max)
    df['date_range'] = (df['date_last_visit'] - df['date_first_visit']).dt.days
    df['duration_0_to_t_days'] = (df['study_date_mask_pe'] - df['date_first_visit']).dt.days

    # Compute number of days since iCPET study
    df['duration_t_to_iCPET_days'] = (df['study_date_mask_pe'] - df['study_date_mask_cl']).dt.days

    # Compute number of days since VQ study
    df['duration_t_to_VQ_days'] = (df['study_date_mask_pe'] - df['vq_date']).dt.days

    # Compute previous clot burden and pct_change in clot burden
    # To be used for graphing/segmenting/comparisons
    df = df.sort_values(by=['pe_number_clean', 'study_date_mask_pe'])
    df['previous_clot_burden'] = df.groupby('pe_number_clean')['total_clot_burden'].shift()
    df['previous_study_date'] = df.groupby('pe_number_clean')['study_date_mask_pe'].shift()
    df['pct_change_clot_burden'] = (df['total_clot_burden'] / df['previous_clot_burden']) - 1

    # Compute max number of days from first visit
    # Can determine if patient has 'long' or 'short' term observations
    df['duration_prev_to_t_days'] = (df['study_date_mask_pe'] - df['previous_study_date']).dt.days
    df['change_clot_burden_per_day'] = (df['total_clot_burden'] - df['previous_clot_burden']) / df['duration_prev_to_t_days']
    df['max_days_from_first_visit'] = df.groupby('pe_number_clean')['duration_0_to_t_days'].transform(max)
    df['long_term_obs'] = df['max_days_from_first_visit'] > 365

    # Calculate maximum clot burden and categorize into groups
    # May make visualization of progression somewhat easier
    df['max_clot_burden'] = df.groupby('pe_number_clean')['total_clot_burden'].transform(max)
    qcut_labels= ['0-20 percentile', '20-40 percentile', '40-60 percentile', '60-80 percentile', '80-100 percentile']
    df['max_clot_burden_category'] = pd.qcut(df['max_clot_burden'], q=len(qcut_labels), labels=qcut_labels)

    # Create numeric PE number for graphing and sorting
    df['pe_number_clean'] = pd.to_numeric(df['pe_number_clean'].str.replace('PE', '')).astype(np.float64)

    return df

def main():
    df = combine_data()
    df.to_pickle(Path('../data/df_clean.pkl'))
    df.to_csv(Path('../data/df_clean.csv'))

if __name__=='__main__':
    main()