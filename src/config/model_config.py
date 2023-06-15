cat_targets = [
	'resolved_pe'
]

num_targets = [
	'total_clot_burden',
	'centralartery',
	'apical_rs1',
	'posterior_rs2',
	'anterior_rs3',
	'lateral_rs4',
	'medial_rs5',
	'superior_rs6',
	'medial_basal_rs7',
	'anterior_basal_rs8',
	'lateral_basal_rs9',
	'posterior_basal_rs10',
	'apical_ls1',
	'posterior_ls2',
	'anterior_ls3',
	'superior_ls4',
	'inferior_ls5',
	'superior_ls6',
	'anteromedial_basal_ls7_8',
	'lateral_basal_ls9',
	'posterior_basal_ls10',
]

body_feat = [
    'volume_visceral_fat', 
    'density_visceral_fat', 
    'mass_visceral_fat',
    'volume_subcutaneous_fat', 
    'density_subcutaneous_fat', 
    'mass_subcutaneous_fat',
    'volume_intermuscular_fat', 
    'density_intermuscular_fat', 
    'mass_intermuscular_fat',
    'volume_muscle', 
    'density_muscle', 
    'mass_muscle',
    'volume_bone', 
    'density_bone', 
    'mass_bone',
    'bmi',
    'bsa',
]

cardiopulmonary_feat = [
    'emphysema_volume_950hu',
    'lung_volume',
    'extrapulmonary_artery_volume',
    'extrapulmonary_vein_volume',
    'intrapulmonary_artery_volume',
    'intrapulmonary_vein_volume',
    'artery_vein_ratio',
    'bv5',
    'bv10',
    'pb_larger_10',
    'pv_diameter',
    'a_diameter',
    'pv_a',
    'heart_volume',
    'airway_volume',
    'airway_ratio',
    'ild_volume',
    'ild_ratio',
]

controls = [
    'age',
    'gender_cl',
    'race',
]

controls_encoded = [
    'age',
    'gender_cl_Male',
    'race_White',
]

control_options = [
    None,
    ['age'],
    ['gender_cl_Male'],
    ['age', 'gender_cl_Male']
]

icpet_num_feat = [
    'rer',
    'vo2_ml_kg_min_at_at',
    'peak_vo2_ml_kg_min',
    'estimated_peak_vo2_ml_kg_min',
    'percent_vo2_at_at',
    'peak_vo2_ml_min',
    'estimated_peak_vo2_ml_min',
    'percent_peak_vo2',
    'percent_co_achieved',
    've_vco2_slope',
    've_vco2_at_at',
    'vo2_hr_peak_percent_',
    'vo2_work_slope_output',
    'mets',
    'peak_measured_mpap_mmhg',
    'peak_calculated_mpap_mmhg',
    'peak_pvr_wu',
    'peak_arterial_hb',
    'peak_cao2',
    'peak_cvo2',
    'peak_cavo2',
    'peak_paao2',
    'peak_fick_co',
    'peak_vd_vt',
    'peak_cavo2_a_art_hb',
    'peak_pa_elastance_ea_mmhg_ml_m2',
    'hyperventilation_num',
]

icpet_cat_feat = [
    'normal_study',
    'borderline_ph',
    'resting_pah',
    'eph',
    'resting_hfpef',
    'exercise_hfpef',
    'resting_ph_exercise_hfpef',
    'exercise_ph_resting_hfpef',
    'deconditioning',
    'preload_insufficiency',
    'inappropriate_o2_extraction',
    'systemic_htn_response',
    'approached_ventilatory_ceiling',
    'surpassed_ventilatory_ceiling',
    'hyperventilation'
]