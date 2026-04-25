# Lipschitz_Guided_Training
Lipschitz Guided Training code


Dependencies:

* GPU access
* git clone and build Marabou as described in https://github.com/NeuralNetworkVerification/Marabou
* Gurobi license https://www.gurobi.com/
* Python 3.8
* pip install numpy pandas torch pytorch-lightning neuralforecast \
            onnx onnxruntime matplotlib termcolor tensorflow

Repository structure
Verify_nn_prod_plan/
            models_20260123/
                        domain_coverage_analysis.py 
                        empirical_manygraphs_aggregated_families.py 
                        lipschitz_empirical.py 
                        time_series_sensitivity.py

                        nhits_20260123/
                                    data/               # Preprocessed UCI demand data
                                    lightning_logs/     # Forecast model training logs
                                    models/             # Trained forecasters and policies
                                                policy_baseline/
                                                policy_robust/
                        
                                    scripts/            # Training scripts
                                    src/                # Forecasting + verification code                        
                        nbeats_20260123/
                                    data/
                                    lightning_logs/
                                    models/
                                    scripts/
                                    src/


# Commands (end-to-end pipeline):
```bash
1) Train Models

NHITS
cd models_20260123/nhits_20260123/
python3 -m src.forecast_model --kind nhits --workspace models/nhits \
  | tee "output_forecastmodel_nhits_$(date +%Y%m%d_%H%M%S).txt"

NBEATS Similarly:
cd models_20260123/nbeats_20260123/
python3 -m src.forecast_model --kind nbeats --workspace models/nbeats | tee "output_forecastmodel_nbeats_$(date +%Y%m%d_%H%M%S).txt"


2) Train both baseline+robust policies (writes models/scaling.json)
cd models_20260123/nhits_20260123/
python3 scripts/train_policy_two_onnx_both.py --forecast nhits --k 7 --pert-radius 1.0 --eps-q 0.1 | tee "output_trainpolicy_nhits_$(date +%Y%m%d_%H%M%S).txt"

cd models_20260123/nbeats_20260123/
python3 scripts/train_policy_two_onnx_both.py --forecast nbeats --k 7 --pert-radius 1.0 --eps-q 0.1 | tee "output_trainpolicy_nbeats_$(date +%Y%m%d_%H%M%S).txt"


3) Quick sanity checks in each controller directory :

3)1)
cd models_20260123/nhits_20260123/
cat models/scaling.json

cd models_20260123/nbeats_20260123/
cat models/scaling.json

You should see models/scaling.json containing:

I_MIN/I_MAX and F_MIN/F_MAX
i_scale_raw, f_scale_raw
last_forecast_raw_clipped_nonneg (length k)
 forecast_kind, seed, k

3)2)
cd models_20260123/nhits_20260123/
ls -la models/policy_baseline models/policy_robust

cd models_20260123/nbeats_20260123/
ls -la models/policy_baseline models/policy_robust

You should see similar output to

-rw-rw-r-- 1 user   351 jan.  29 01:44 lipschitz.json
-rw-rw-r-- 1 user  3431 jan.  29 01:44 policy.onnx
-rw-rw-r-- 1 user  4801 jan.  29 01:44 policy_two_copy.onnx


4) VERIFY POLICIES

cd models_20260123

4)1) NHITS BASELINE

python3 -m nhits_20260123.src.verify_robust_marabou \
  --models-dir nhits_20260123/models \
  --model-path nhits_20260123/models/policy_baseline/policy_two_copy.onnx \
  --eps-q 0.2 \
  --epsf-scaled 1.0 \
  --i-max-scaled 1.0 \
  --f-max-scaled 2.0 \
  --timeout 300 \
  --verbose  | tee "output_verify_nhits_baseline_$(date +%Y%m%d_%H%M%S).txt"

Expected output: ROBUSTNESS NOT PROVED: SAT/UNKNOWN in at least one direction.

4)2) NHITS ROBUST
python3 -m nhits_20260123.src.verify_robust_marabou \
  --models-dir nhits_20260123/models \
  --model-path nhits_20260123/models/policy_robust/policy_two_copy.onnx \
  --eps-q 0.2 \
  --epsf-scaled 1.0 \
  --i-max-scaled 1.0 \
  --f-max-scaled 2.0 \
  --timeout 300 \
  --verbose  | tee "output_verify_nhits_robust_$(date +%Y%m%d_%H%M%S).txt"

Expected output: ROBUSTNESS PROVED: UNSAT in both directions.

4)3) NBEATS BASELINE
python3 -m nbeats_20260123.src.verify_robust_marabou \
  --models-dir nbeats_20260123/models \
  --model-path nbeats_20260123/models/policy_baseline/policy_two_copy.onnx \
  --eps-q 0.2 \
  --epsf-scaled 1.0 \
  --i-max-scaled 1.0 \
  --f-max-scaled 2.0 \
  --timeout 300 \
  --verbose | tee "output_verify_nbeats_baseline_$(date +%Y%m%d_%H%M%S).txt"

Expected output: ROBUSTNESS NOT PROVED: SAT/UNKNOWN in at least one direction.

4)4) NBEATS ROBUST
python3 -m nbeats_20260123.src.verify_robust_marabou \
  --models-dir nbeats_20260123/models \
  --model-path nbeats_20260123/models/policy_robust/policy_two_copy.onnx \
  --eps-q 0.2 \
  --epsf-scaled 1.0 \
  --i-max-scaled 1.0 \
  --f-max-scaled 2.0 \
  --timeout 300 \
  --verbose  | tee "output_verify_nbeats_robust_$(date +%Y%m%d_%H%M%S).txt"

Expected output: ROBUSTNESS PROVED: UNSAT in both directions.


5) GENERATE EMPIRICAL EXPERIMENTS AND GRAPHS

cd Verify_nn_prod_plan

(1) GRAPHS WITH DOMAIN FAMILIES  
python3 -m models_20260123.empirical_manygraphs_aggregated_families \
  --controllers nhits_baseline nhits_robust nbeats_baseline nbeats_robust \
  --policy-onnx \
    nhits_baseline=models_20260123/nhits_20260123/models/policy_baseline/policy_two_copy.onnx \
    nhits_robust=models_20260123/nhits_20260123/models/policy_robust/policy_two_copy.onnx \
    nbeats_baseline=models_20260123/nbeats_20260123/models/policy_baseline/policy_two_copy.onnx \
    nbeats_robust=models_20260123/nbeats_20260123/models/policy_robust/policy_two_copy.onnx \
  --scaling-json models_20260123/nhits_20260123/models/scaling.json \
  --epsf-scaled 1.0 \
  --epsq-list 0.01 0.05 0.1 0.2 0.5 1.0 2.0 \
  --n-pairs 3000 \
  --paper-epsq 0.1 \
| tee "output_empirical_manygraphs_$(date +%Y%m%d_%H%M%S).txt"


[done] outputs in: models_20260123/empirical_out_20260129_022647

(2) lipschitz_empirical.py

python3 -m models_20260123.lipschitz_empirical \
  --controllers nhits_baseline nhits_robust nbeats_baseline nbeats_robust \
  --policy-onnx \
    nhits_baseline=models_20260123/nhits_20260123/models/policy_baseline/policy_two_copy.onnx \
    nhits_robust=models_20260123/nhits_20260123/models/policy_robust/policy_two_copy.onnx \
    nbeats_baseline=models_20260123/nbeats_20260123/models/policy_baseline/policy_two_copy.onnx \
    nbeats_robust=models_20260123/nbeats_20260123/models/policy_robust/policy_two_copy.onnx \
  --scaling-json models_20260123/nhits_20260123/models/scaling.json \
  --epsf-scaled 1.0 \
  --n-pairs 3000 \
  --outdir models_20260123/empirical_out \
  | tee "output_empirical_lipschitz_empirical_$(date +%Y%m%d_%H%M%S).txt"
  
Outputs
[done] wrote: models_20260123/empirical_out/lipschitz_out_20260129_022912/lipschitz_validation_table.csv
[done] wrote: models_20260123/empirical_out/lipschitz_out_20260129_022912/lipschitz_vs_empirical.png
[done] wrote: models_20260123/empirical_out/lipschitz_out_20260129_022912/Lemp_vs_Lhat.png

It will create a timestamped folder like:

models_20260123/empirical_out/lipschitz_out_YYYYMMDD_HHMMSS/
  lipschitz_validation_table.csv
  lipschitz_vs_empirical.png
  Lemp_vs_Lhat.png
  lipschitz_vs_empirical.tex


(3) time_series_sensitivity.py

python3 -m models_20260123.time_series_sensitivity \
  --episodes-pkl models_20260123/nhits_20260123/models/episodes_seed.pkl \
  --scaling-json  models_20260123/nhits_20260123/models/scaling.json \
  --controllers nhits_baseline nhits_robust nbeats_baseline nbeats_robust \
  --policy-onnx \
    nhits_baseline=models_20260123/nhits_20260123/models/policy_baseline/policy_two_copy.onnx \
    nhits_robust=models_20260123/nhits_20260123/models/policy_robust/policy_two_copy.onnx \
    nbeats_baseline=models_20260123/nbeats_20260123/models/policy_baseline/policy_two_copy.onnx \
    nbeats_robust=models_20260123/nbeats_20260123/models/policy_robust/policy_two_copy.onnx \
  --T 60 \
  --episode-idx 0 \
  --epsf-scaled 1.0 \
  --epsq 0.1 \
  --delta-mode uniform \
  --outroot models_20260123 \
  --seed 0 \
| tee "models_20260123/output_timeseries_sensitivity_epsq01_$(date +%Y%m%d_%H%M%S).txt"


This will create:
(models_20260123/empirical_out_20260125_022254/...)
models_20260123/empirical_out_TIMESTAMP/timeseries/fig_timeseries_nominal_vs_perturbed.png
models_20260123/empirical_out_TIMESTAMP/timeseries/fig_timeseries_sensitivity.png
models_20260123/empirical_out_TIMESTAMP/timeseries/timeseries_summary.csv

(4) domain_coverage_analysis.py

Please use the directory from experiment output (1), e.g. 'empirical_out_20260129_022647' in the parameter '--domain-table-csv'

(4.1) NHITS — Domain Coverage Analysis
Baseline + Robust (single run, same domains)
$python3 -m models_20260123.domain_coverage_analysis \
  --domain-table-csv models_20260123/empirical_out_20260129_022647/tables/domain_table_nhits_baseline.csv \
  --controllers nhits_baseline nhits_robust \
  --policy-onnx \
    nhits_baseline=models_20260123/nhits_20260123/models/policy_baseline/policy_two_copy.onnx \
    nhits_robust=models_20260123/nhits_20260123/models/policy_robust/policy_two_copy.onnx \
  --epsf-scaled 1.0 \
  --epsq 0.2 \
  --num-samples 5000 \
  --seed 0 \
  --outroot empirical_results \
  | tee "models_20260123/output_domain_coverage_analysis_epsq02_$(date +%Y%m%d_%H%M%S).txt"
  
[ok] wrote outputs to: empirical_results/empirical_out_20260129_023639/domain_coverage
     - empirical_results/empirical_out_20260129_023639/domain_coverage/domain_coverage_per_domain.csv
     - empirical_results/empirical_out_20260129_023639/domain_coverage/domain_coverage_by_family.csv
     - empirical_results/empirical_out_20260129_023639/domain_coverage/domain_coverage_table.tex
  

(4.2) NBEATS — Domain Coverage Analysis
python3 -m models_20260123.domain_coverage_analysis \
  --domain-table-csv models_20260123/empirical_out_20260129_022647/tables/domain_table_nbeats_baseline.csv \
  --controllers nbeats_baseline nbeats_robust \
  --policy-onnx \
    nbeats_baseline=models_20260123/nbeats_20260123/models/policy_baseline/policy_two_copy.onnx \
    nbeats_robust=models_20260123/nbeats_20260123/models/policy_robust/policy_two_copy.onnx \
  --epsf-scaled 1.0 \
  --epsq 0.2 \
  --num-samples 5000 \
  --seed 0 \
  --outroot empirical_results
  
[ok] wrote outputs to: empirical_results/empirical_out_20260129_023708/domain_coverage
     - empirical_results/empirical_out_20260129_023708/domain_coverage/domain_coverage_per_domain.csv
     - empirical_results/empirical_out_20260129_023708/domain_coverage/domain_coverage_by_family.csv
     - empirical_results/empirical_out_20260129_023708/domain_coverage/domain_coverage_table.tex


################
################
 
 
VIRTUAL ENVIRONMENT SET UP
 # Replace 'myenv' with your preferred environment name
python3 -m venv myenv
source myenv/bin/activate

Install packages: Use pip while the environment is active: 
pip install <package_name>.

Deactivate: Type "deactivate" to return to the system Python.

Delete: To remove the environment, simply delete the folder: 
rm -rf myenv


################
################
 
 
