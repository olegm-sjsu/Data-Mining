/* Lending Club scorecard, generated from the Python model. Not executed in SAS. */
/* The input table needs the engineered variables built as in lc_features.add_features (for example fico, loan_to_income): */
/* fico, annual_inc, loan_to_income, dti, revol_util, inq_last_6mths, bc_open_to_buy, total_bc_limit, tot_hi_cred_lim, acc_open_past_24mths, mort_acc, percent_bc_gt_75, num_tl_op_past_12m, mths_since_recent_bc, mo_sin_rcnt_tl, mo_sin_rcnt_rev_tl_op, mo_sin_old_rev_tl_op, home_ownership, purpose */
data work.scored;
  set work.loans;
  /* fico */
  if missing(fico) then fico_woe = 0.000000;
  else if fico < 667.000000 then fico_woe = -0.470925;
  else if fico < 672.000000 then fico_woe = -0.391508;
  else if fico < 682.000000 then fico_woe = -0.294020;
  else if fico < 687.000000 then fico_woe = -0.219550;
  else if fico < 692.000000 then fico_woe = -0.136376;
  else if fico < 702.000000 then fico_woe = -0.056730;
  else if fico < 712.000000 then fico_woe = 0.118930;
  else if fico < 722.000000 then fico_woe = 0.281362;
  else if fico < 747.000000 then fico_woe = 0.461619;
  else fico_woe = 0.925257;
  /* annual_inc */
  if missing(annual_inc) then annual_inc_woe = 0.000000;
  else if annual_inc < 31000.000000 then annual_inc_woe = -0.457759;
  else if annual_inc < 39600.000000 then annual_inc_woe = -0.340463;
  else if annual_inc < 45000.000000 then annual_inc_woe = -0.232118;
  else if annual_inc < 52000.000000 then annual_inc_woe = -0.117446;
  else if annual_inc < 60000.000000 then annual_inc_woe = -0.016568;
  else if annual_inc < 68000.000000 then annual_inc_woe = 0.041072;
  else if annual_inc < 78000.000000 then annual_inc_woe = 0.157436;
  else if annual_inc < 90600.000000 then annual_inc_woe = 0.300738;
  else if annual_inc < 115306.400000 then annual_inc_woe = 0.480944;
  else annual_inc_woe = 0.501234;
  /* loan_to_income */
  if missing(loan_to_income) then loan_to_income_woe = 0.000000;
  else if loan_to_income < 0.069868 then loan_to_income_woe = 0.240580;
  else if loan_to_income < 0.100000 then loan_to_income_woe = 0.220982;
  else if loan_to_income < 0.128205 then loan_to_income_woe = 0.209334;
  else if loan_to_income < 0.154138 then loan_to_income_woe = 0.155538;
  else if loan_to_income < 0.181818 then loan_to_income_woe = 0.096643;
  else if loan_to_income < 0.210000 then loan_to_income_woe = -0.000358;
  else if loan_to_income < 0.244154 then loan_to_income_woe = 0.037037;
  else if loan_to_income < 0.285714 then loan_to_income_woe = -0.151130;
  else if loan_to_income < 0.340004 then loan_to_income_woe = -0.281421;
  else loan_to_income_woe = -0.365156;
  /* dti */
  if missing(dti) then dti_woe = 0.000000;
  else if dti < 6.340000 then dti_woe = 0.302340;
  else if dti < 9.250000 then dti_woe = 0.253655;
  else if dti < 11.640000 then dti_woe = 0.194941;
  else if dti < 13.780000 then dti_woe = 0.128305;
  else if dti < 15.890000 then dti_woe = 0.057859;
  else if dti < 18.070000 then dti_woe = -0.015891;
  else if dti < 20.420000 then dti_woe = -0.067795;
  else if dti < 23.070000 then dti_woe = -0.139562;
  else if dti < 26.692000 then dti_woe = -0.213730;
  else dti_woe = -0.348833;
  /* revol_util */
  if missing(revol_util) then revol_util_woe = -0.224694;
  else if revol_util < 21.900000 then revol_util_woe = 0.462180;
  else if revol_util < 34.500000 then revol_util_woe = 0.216068;
  else if revol_util < 43.600000 then revol_util_woe = 0.107494;
  else if revol_util < 51.300000 then revol_util_woe = 0.030976;
  else if revol_util < 58.300000 then revol_util_woe = 0.021319;
  else if revol_util < 65.000000 then revol_util_woe = -0.025211;
  else if revol_util < 71.700000 then revol_util_woe = -0.084741;
  else if revol_util < 79.000000 then revol_util_woe = -0.140278;
  else if revol_util < 87.300000 then revol_util_woe = -0.136784;
  else revol_util_woe = -0.298247;
  /* inq_last_6mths */
  if missing(inq_last_6mths) then inq_last_6mths_woe = 0.000000;
  else if inq_last_6mths < 0.000000 then inq_last_6mths_woe = 0.000000;
  else if inq_last_6mths < 1.000000 then inq_last_6mths_woe = 0.158132;
  else if inq_last_6mths < 2.000000 then inq_last_6mths_woe = -0.057914;
  else inq_last_6mths_woe = -0.270104;
  /* bc_open_to_buy */
  if missing(bc_open_to_buy) then bc_open_to_buy_woe = 0.116727;
  else if bc_open_to_buy < 269.000000 then bc_open_to_buy_woe = -0.420268;
  else if bc_open_to_buy < 739.000000 then bc_open_to_buy_woe = -0.243626;
  else if bc_open_to_buy < 1372.000000 then bc_open_to_buy_woe = -0.185827;
  else if bc_open_to_buy < 2233.000000 then bc_open_to_buy_woe = -0.190956;
  else if bc_open_to_buy < 3419.000000 then bc_open_to_buy_woe = -0.081752;
  else if bc_open_to_buy < 5073.000000 then bc_open_to_buy_woe = -0.009949;
  else if bc_open_to_buy < 7594.400000 then bc_open_to_buy_woe = 0.041472;
  else if bc_open_to_buy < 11997.000000 then bc_open_to_buy_woe = 0.183813;
  else if bc_open_to_buy < 21447.800000 then bc_open_to_buy_woe = 0.317482;
  else bc_open_to_buy_woe = 0.662625;
  /* total_bc_limit */
  if missing(total_bc_limit) then total_bc_limit_woe = 0.137753;
  else if total_bc_limit < 4000.000000 then total_bc_limit_woe = -0.372240;
  else if total_bc_limit < 6200.000000 then total_bc_limit_woe = -0.281270;
  else if total_bc_limit < 8500.000000 then total_bc_limit_woe = -0.194811;
  else if total_bc_limit < 10900.000000 then total_bc_limit_woe = -0.136143;
  else if total_bc_limit < 13700.000000 then total_bc_limit_woe = -0.090428;
  else if total_bc_limit < 17300.000000 then total_bc_limit_woe = -0.014857;
  else if total_bc_limit < 22000.000000 then total_bc_limit_woe = 0.074961;
  else if total_bc_limit < 29000.000000 then total_bc_limit_woe = 0.116718;
  else if total_bc_limit < 41500.000000 then total_bc_limit_woe = 0.284728;
  else total_bc_limit_woe = 0.599913;
  /* tot_hi_cred_lim */
  if missing(tot_hi_cred_lim) then tot_hi_cred_lim_woe = 0.013156;
  else if tot_hi_cred_lim < 22100.000000 then tot_hi_cred_lim_woe = -0.339019;
  else if tot_hi_cred_lim < 34013.000000 then tot_hi_cred_lim_woe = -0.310321;
  else if tot_hi_cred_lim < 46380.000000 then tot_hi_cred_lim_woe = -0.190679;
  else if tot_hi_cred_lim < 62530.400000 then tot_hi_cred_lim_woe = -0.113492;
  else if tot_hi_cred_lim < 90233.000000 then tot_hi_cred_lim_woe = -0.090356;
  else if tot_hi_cred_lim < 140729.600000 then tot_hi_cred_lim_woe = 0.015186;
  else if tot_hi_cred_lim < 196802.700000 then tot_hi_cred_lim_woe = 0.080523;
  else if tot_hi_cred_lim < 260156.600000 then tot_hi_cred_lim_woe = 0.202348;
  else if tot_hi_cred_lim < 360268.700000 then tot_hi_cred_lim_woe = 0.383611;
  else tot_hi_cred_lim_woe = 0.611771;
  /* acc_open_past_24mths */
  if missing(acc_open_past_24mths) then acc_open_past_24mths_woe = 0.137753;
  else if acc_open_past_24mths < 1.000000 then acc_open_past_24mths_woe = 0.374855;
  else if acc_open_past_24mths < 2.000000 then acc_open_past_24mths_woe = 0.258273;
  else if acc_open_past_24mths < 3.000000 then acc_open_past_24mths_woe = 0.160457;
  else if acc_open_past_24mths < 4.000000 then acc_open_past_24mths_woe = 0.048988;
  else if acc_open_past_24mths < 5.000000 then acc_open_past_24mths_woe = -0.052578;
  else if acc_open_past_24mths < 6.000000 then acc_open_past_24mths_woe = -0.124322;
  else if acc_open_past_24mths < 7.000000 then acc_open_past_24mths_woe = -0.245156;
  else acc_open_past_24mths_woe = -0.363576;
  /* mort_acc */
  if missing(mort_acc) then mort_acc_woe = 0.137753;
  else if mort_acc < 0.000000 then mort_acc_woe = 0.000000;
  else if mort_acc < 1.000000 then mort_acc_woe = -0.224713;
  else if mort_acc < 2.000000 then mort_acc_woe = -0.007633;
  else if mort_acc < 3.000000 then mort_acc_woe = 0.146269;
  else if mort_acc < 5.000000 then mort_acc_woe = 0.220122;
  else mort_acc_woe = 0.334749;
  /* percent_bc_gt_75 */
  if missing(percent_bc_gt_75) then percent_bc_gt_75_woe = 0.116802;
  else if percent_bc_gt_75 < 0.000000 then percent_bc_gt_75_woe = 0.000000;
  else if percent_bc_gt_75 < 16.700000 then percent_bc_gt_75_woe = 0.315611;
  else if percent_bc_gt_75 < 33.300000 then percent_bc_gt_75_woe = 0.159256;
  else if percent_bc_gt_75 < 42.900000 then percent_bc_gt_75_woe = 0.075690;
  else if percent_bc_gt_75 < 50.000000 then percent_bc_gt_75_woe = 0.030234;
  else if percent_bc_gt_75 < 66.700000 then percent_bc_gt_75_woe = -0.047059;
  else if percent_bc_gt_75 < 75.000000 then percent_bc_gt_75_woe = -0.156867;
  else if percent_bc_gt_75 < 100.000000 then percent_bc_gt_75_woe = -0.208128;
  else percent_bc_gt_75_woe = -0.217353;
  /* num_tl_op_past_12m */
  if missing(num_tl_op_past_12m) then num_tl_op_past_12m_woe = 0.013156;
  else if num_tl_op_past_12m < 0.000000 then num_tl_op_past_12m_woe = 0.000000;
  else if num_tl_op_past_12m < 1.000000 then num_tl_op_past_12m_woe = 0.290293;
  else if num_tl_op_past_12m < 2.000000 then num_tl_op_past_12m_woe = 0.108225;
  else if num_tl_op_past_12m < 3.000000 then num_tl_op_past_12m_woe = -0.036994;
  else if num_tl_op_past_12m < 4.000000 then num_tl_op_past_12m_woe = -0.168089;
  else num_tl_op_past_12m_woe = -0.386861;
  /* mths_since_recent_bc */
  if missing(mths_since_recent_bc) then mths_since_recent_bc_woe = 0.119034;
  else if mths_since_recent_bc < 3.000000 then mths_since_recent_bc_woe = -0.385785;
  else if mths_since_recent_bc < 5.000000 then mths_since_recent_bc_woe = -0.235149;
  else if mths_since_recent_bc < 8.000000 then mths_since_recent_bc_woe = -0.118171;
  else if mths_since_recent_bc < 11.000000 then mths_since_recent_bc_woe = -0.055047;
  else if mths_since_recent_bc < 15.000000 then mths_since_recent_bc_woe = -0.094923;
  else if mths_since_recent_bc < 20.000000 then mths_since_recent_bc_woe = 0.012167;
  else if mths_since_recent_bc < 27.000000 then mths_since_recent_bc_woe = 0.051868;
  else if mths_since_recent_bc < 41.000000 then mths_since_recent_bc_woe = 0.102426;
  else if mths_since_recent_bc < 64.000000 then mths_since_recent_bc_woe = 0.205301;
  else mths_since_recent_bc_woe = 0.308064;
  /* mo_sin_rcnt_tl */
  if missing(mo_sin_rcnt_tl) then mo_sin_rcnt_tl_woe = 0.013156;
  else if mo_sin_rcnt_tl < 2.000000 then mo_sin_rcnt_tl_woe = -0.330649;
  else if mo_sin_rcnt_tl < 3.000000 then mo_sin_rcnt_tl_woe = -0.235873;
  else if mo_sin_rcnt_tl < 4.000000 then mo_sin_rcnt_tl_woe = -0.163887;
  else if mo_sin_rcnt_tl < 5.000000 then mo_sin_rcnt_tl_woe = -0.076705;
  else if mo_sin_rcnt_tl < 6.000000 then mo_sin_rcnt_tl_woe = -0.023917;
  else if mo_sin_rcnt_tl < 8.000000 then mo_sin_rcnt_tl_woe = 0.072002;
  else if mo_sin_rcnt_tl < 10.000000 then mo_sin_rcnt_tl_woe = 0.071478;
  else if mo_sin_rcnt_tl < 13.000000 then mo_sin_rcnt_tl_woe = 0.073455;
  else if mo_sin_rcnt_tl < 19.000000 then mo_sin_rcnt_tl_woe = 0.232054;
  else mo_sin_rcnt_tl_woe = 0.355007;
  /* mo_sin_rcnt_rev_tl_op */
  if missing(mo_sin_rcnt_rev_tl_op) then mo_sin_rcnt_rev_tl_op_woe = 0.012975;
  else if mo_sin_rcnt_rev_tl_op < 2.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.329575;
  else if mo_sin_rcnt_rev_tl_op < 3.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.248837;
  else if mo_sin_rcnt_rev_tl_op < 5.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.144595;
  else if mo_sin_rcnt_rev_tl_op < 7.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.024106;
  else if mo_sin_rcnt_rev_tl_op < 9.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.031472;
  else if mo_sin_rcnt_rev_tl_op < 11.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.005518;
  else if mo_sin_rcnt_rev_tl_op < 15.000000 then mo_sin_rcnt_rev_tl_op_woe = -0.011915;
  else if mo_sin_rcnt_rev_tl_op < 20.000000 then mo_sin_rcnt_rev_tl_op_woe = 0.110825;
  else if mo_sin_rcnt_rev_tl_op < 32.000000 then mo_sin_rcnt_rev_tl_op_woe = 0.252177;
  else mo_sin_rcnt_rev_tl_op_woe = 0.403396;
  /* mo_sin_old_rev_tl_op */
  if missing(mo_sin_old_rev_tl_op) then mo_sin_old_rev_tl_op_woe = 0.012975;
  else if mo_sin_old_rev_tl_op < 78.000000 then mo_sin_old_rev_tl_op_woe = -0.342569;
  else if mo_sin_old_rev_tl_op < 100.000000 then mo_sin_old_rev_tl_op_woe = -0.143907;
  else if mo_sin_old_rev_tl_op < 122.000000 then mo_sin_old_rev_tl_op_woe = -0.122956;
  else if mo_sin_old_rev_tl_op < 141.000000 then mo_sin_old_rev_tl_op_woe = -0.075191;
  else if mo_sin_old_rev_tl_op < 157.000000 then mo_sin_old_rev_tl_op_woe = -0.041683;
  else if mo_sin_old_rev_tl_op < 176.000000 then mo_sin_old_rev_tl_op_woe = 0.095857;
  else if mo_sin_old_rev_tl_op < 203.000000 then mo_sin_old_rev_tl_op_woe = 0.061595;
  else if mo_sin_old_rev_tl_op < 237.000000 then mo_sin_old_rev_tl_op_woe = 0.131708;
  else if mo_sin_old_rev_tl_op < 298.000000 then mo_sin_old_rev_tl_op_woe = 0.228849;
  else mo_sin_old_rev_tl_op_woe = 0.253806;
  /* home_ownership */
  select (strip(put(home_ownership, $32.)));
    when ('MORTGAGE') home_ownership_woe = 0.198497;
    when ('RENT') home_ownership_woe = -0.171707;
    when ('OWN') home_ownership_woe = -0.067415;
    otherwise home_ownership_woe = -0.561784;
  end;
  /* purpose */
  select (strip(put(purpose, $32.)));
    when ('credit_card') purpose_woe = 0.181553;
    when ('other') purpose_woe = -0.280173;
    when ('major_purchase') purpose_woe = 0.317183;
    when ('debt_consolidation') purpose_woe = -0.037209;
    when ('moving') purpose_woe = -0.281987;
    when ('small_business') purpose_woe = -0.668448;
    when ('home_improvement') purpose_woe = 0.161251;
    when ('wedding') purpose_woe = 0.126953;
    when ('car') purpose_woe = 0.343509;
    when ('house') purpose_woe = 0.006127;
    when ('medical') purpose_woe = -0.257865;
    when ('vacation') purpose_woe = -0.198252;
    otherwise purpose_woe = -0.374009;
  end;
  logit = -1.951371 + (-0.592572 * fico_woe) + (-0.572418 * annual_inc_woe) + (-0.646037 * loan_to_income_woe) + (-0.323807 * dti_woe) + (-0.343030 * revol_util_woe) + (-0.739059 * inq_last_6mths_woe) + (-0.263808 * bc_open_to_buy_woe) + (-0.051219 * total_bc_limit_woe) + (-0.250771 * tot_hi_cred_lim_woe) + (-0.450989 * acc_open_past_24mths_woe) + (-0.015057 * mort_acc_woe) + (-0.036204 * percent_bc_gt_75_woe) + (-0.350180 * num_tl_op_past_12m_woe) + (-0.287862 * mths_since_recent_bc_woe) + (-0.182344 * mo_sin_rcnt_tl_woe) + (-0.035105 * mo_sin_rcnt_rev_tl_op_woe) + (-0.153162 * mo_sin_old_rev_tl_op_woe) + (-0.300767 * home_ownership_woe) + (-0.884355 * purpose_woe);
  pd = 1 / (1 + exp(-logit));
run;