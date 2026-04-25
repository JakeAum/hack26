# Hack26 — run log (AWS SageMaker / local)

This file captures **CLI transcripts** from data pulls and training so results stay reviewable in-repo. Paste new blocks under the appropriate section as runs complete.

---

## Iowa dataset — `hack26-dataset` (2018–2022)

**When:** 2026-04-25  
**Host:** `sagemaker-user@default` (paths under `~/hack26`)  
**Command:**

```text
hack26-dataset --states Iowa --start 2018 --end 2022 -v --allow-download
```

**Rotating log (engine):** `~/hack26/data/derived/logs/dataset_20260425_172016.log`

**Console output:**

```text
(hack26) sagemaker-user@default:~/hack26$ hack26-dataset --states Iowa --start 2018 --end 2022 -v --allow-download
2026-04-25 17:20:17 INFO     engine.dataset  environment:                                                                                             
                    INFO     engine.dataset    python:       3.12.13 (Linux-6.1.166-197.305.amzn2023.x86_64-x86_64-with-glibc2.39)                    
                    INFO     engine.dataset    torch:        2.8.0                                                                                    
                    INFO     engine.dataset    cuda:         torch installed, CUDA unavailable (CPU run)                                              
                    INFO     engine.dataset    data_root:    /home/sagemaker-user/hack26/data  (free=9223371962.2 GB)                                 
                    INFO     engine.dataset    HACK26_CDL_DATA_DIR=<unset>                                                                            
                    INFO     engine.dataset    HACK26_CACHE_DIR=<unset>                                                                               
                    INFO     engine.dataset    NASS_API_KEY=C44***90B  (len=36)                                                                       
                    INFO     engine.dataset    git:          991ade5-dirty                                                                            
                    INFO     engine.dataset    argv:         /opt/conda/bin/hack26-dataset --states Iowa --start 2018 --end 2022 -v --allow-download  
                    INFO     engine.dataset  rotated log file: /home/sagemaker-user/hack26/data/derived/logs/dataset_20260425_172016.log              
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  STEP 1/5  Loading county catalog                                                                         
                    INFO     engine.dataset  start: 2026-04-25 17:20:17  |  pid: 128208  |  host: default                                             
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  counties loaded: n=99  states=['19']                                                                     
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  STEP 2/5  Pulling weather (POWER+SMAP) for 2018-2022                                                     
                    INFO     engine.dataset  start: 2026-04-25 17:20:17  |  pid: 128208  |  host: default                                             
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  weather frame: rows=180774  cols=45  index=['date', 'geoid']                                             
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  STEP 3/5  Pulling CDL annual snapshots for 2018-2022                                                     
                    INFO     engine.dataset  start: 2026-04-25 17:20:17  |  pid: 128208  |  host: default                                             
                    INFO     engine.dataset  ==============================================================================                           
[cdl] downloading https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2018_30m_cdls.zip
[cdl] extracting 2018_30m_cdls.zip → /home/sagemaker-user/hack26/data
[cdl] 25/99 counties processed
[cdl] 50/99 counties processed
[cdl] 75/99 counties processed
[cdl] 99/99 counties processed
2026-04-25 17:23:19 INFO     engine.dataset  cdl-years: 1/5 years (20.0%, 182.4s, 0.0 years/s)  year=2018 rows=99                                     
[cdl] downloading https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2019_30m_cdls.zip
[cdl] extracting 2019_30m_cdls.zip → /home/sagemaker-user/hack26/data
[cdl] 25/99 counties processed
[cdl] 50/99 counties processed
[cdl] 75/99 counties processed
[cdl] 99/99 counties processed
2026-04-25 17:26:31 INFO     engine.dataset  cdl-years: 2/5 years (40.0%, 373.6s, 0.0 years/s)  year=2019 rows=99                                     
[cdl] downloading https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2020_30m_cdls.zip
[cdl] extracting 2020_30m_cdls.zip → /home/sagemaker-user/hack26/data
[cdl] 25/99 counties processed
[cdl] 50/99 counties processed
[cdl] 75/99 counties processed
[cdl] 99/99 counties processed
2026-04-25 17:29:33 INFO     engine.dataset  cdl-years: 3/5 years (60.0%, 555.5s, 0.0 years/s)  year=2020 rows=99                                     
[cdl] downloading https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2021_30m_cdls.zip
[cdl] extracting 2021_30m_cdls.zip → /home/sagemaker-user/hack26/data
[cdl] 25/99 counties processed
[cdl] 50/99 counties processed
[cdl] 75/99 counties processed
[cdl] 99/99 counties processed
2026-04-25 17:32:54 INFO     engine.dataset  cdl-years: 4/5 years (80.0%, 757.1s, 0.0 years/s)  year=2021 rows=99                                     
[cdl] downloading https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip
[cdl] extracting 2022_30m_cdls.zip → /home/sagemaker-user/hack26/data
[cdl] 25/99 counties processed
[cdl] 50/99 counties processed
[cdl] 75/99 counties processed
[cdl] 99/99 counties processed
2026-04-25 17:35:55 INFO     engine.dataset  cdl-years: 5/5 years (100.0%, 938.3s, 0.0 years/s)  year=2022 rows=99                                    
                    INFO     engine.dataset  cdl frame: rows=495  years=5  geoids=99                                                                  
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  STEP 4/5  Pulling NASS county yields (labels) for 2018-2022                                              
                    INFO     engine.dataset  start: 2026-04-25 17:35:55  |  pid: 128208  |  host: default                                             
                    INFO     engine.dataset  ==============================================================================                           
2026-04-25 17:35:57 INFO     engine.dataset  nass frame: rows=617  geoids=99  years=2018..2024                                                        
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  STEP 5/5  Assembling Darts TimeSeries                                                                    
                    INFO     engine.dataset  start: 2026-04-25 17:35:57  |  pid: 128208  |  host: default                                             
                    INFO     engine.dataset  ==============================================================================                           
                    INFO     engine.dataset  past-covariate columns (27): ['PRECTOTCORR', 'PRECTOTCORR_7d_avg', 'PRECTOTCORR_30d_avg', 'T2M',         
                             'T2M_7d_avg', 'T2M_30d_avg', 'T2M_MAX', 'T2M_MAX_7d_avg', 'T2M_MAX_30d_avg', 'T2M_MIN', 'T2M_MIN_7d_avg',                
                             'T2M_MIN_30d_avg', 'GWETROOT', 'GWETROOT_7d_avg', 'GWETROOT_30d_avg', 'GWETTOP', 'GWETTOP_7d_avg', 'GWETTOP_30d_avg',    
                             'GWETPROF', 'GWETPROF_7d_avg', 'GWETPROF_30d_avg', 'GDD', 'GDD_7d_avg', 'GDD_30d_avg', 'GDD_cumulative',                 
                             'GDD_cumulative_7d_avg', 'GDD_cumulative_30d_avg']                                                                       
                    INFO     engine.dataset  future-covariate columns (6): ['doy_sin', 'doy_cos', 'week_sin', 'week_cos', 'month',                    
                             'days_until_end_of_season']                                                                                              
                    INFO     engine.dataset  static-covariate columns  (13): ['state_08', 'state_19', 'state_29', 'state_31', 'state_55',             
                             'corn_pct_of_county', 'corn_pct_of_cropland', 'soybean_pct_of_cropland', 'log_corn_area_m2', 'log_land_area_m2',         
                             'centroid_lat', 'centroid_lon', 'historical_mean_yield_bu_acre']                                                         
2026-04-25 17:35:59 INFO     engine.dataset  series-built: 200/495 series (40.4%, 1.9s, 104.2 series/s)                                               
2026-04-25 17:36:00 INFO     engine.dataset  series-built: 400/495 series (80.8%, 3.0s, 133.7 series/s)                                               
2026-04-25 17:36:01 INFO     engine.dataset  series-built: 495/495 series (100.0%, 3.5s, 142.5 series/s)                                              
                    INFO     engine.dataset  bundle assembled: n_series=457  dropped(label)=38  dropped(cov)=0  dropped(no-wx)=0  past_cols=27        
                             future_cols=6  static_cols=13                                                                                            
                    INFO     engine.dataset  series per year: {2018: 93, 2019: 88, 2020: 95, 2021: 84, 2022: 97}                                      
                    INFO     engine.dataset  [2025-leak-guard] max year in bundle = 2022 (must be ≤ 2024)                                             
                    INFO     engine.dataset  n_series:            457                                                                                 
                    INFO     engine.dataset  past_cov columns:   27 -> ['PRECTOTCORR', 'PRECTOTCORR_7d_avg', 'PRECTOTCORR_30d_avg', 'T2M',            
                             'T2M_7d_avg', 'T2M_30d_avg']...                                                                                          
                    INFO     engine.dataset  future_cov columns: 6 -> ['doy_sin', 'doy_cos', 'week_sin', 'week_cos', 'month',                         
                             'days_until_end_of_season']                                                                                              
                    INFO     engine.dataset  static columns:     13 -> ['state_08', 'state_19', 'state_29', 'state_31', 'state_55',                   
                             'corn_pct_of_county', 'corn_pct_of_cropland', 'soybean_pct_of_cropland', 'log_corn_area_m2', 'log_land_area_m2',         
                             'centroid_lat', 'centroid_lon', 'historical_mean_yield_bu_acre']                                                         
                    INFO     engine.dataset  series per year:    {2018: 93, 2019: 88, 2020: 95, 2021: 84, 2022: 97}                                   
                    INFO     engine.dataset  series per state:   {'19': 457}                                                                          
                    INFO     engine.dataset  label range:        min=132.7  max=234.7  mean=191.4  sd=20.4                                            
(hack26) sagemaker-user@default:~/hack26$ 
```

**Summary (from log):**

| Metric | Value |
| ------ | ----- |
| Counties | 99 (Iowa) |
| CDL years | 2018–2022, ~938 s total for CDL step |
| NASS rows | 617, geoids 99, year span in frame 2018..2024 (cache / pull wider than 2018–2022 build) |
| `n_series` (after drops) | 457 |
| Dropped (missing label) | 38 |
| `past_cols` / `future_cols` / `static_cols` | 27 / 6 / 13 |

---

## Training runs — `hack26-train`

Paste each run below: **command line**, path to the **rotating log** (and optional per-epoch CSV + `test_*_metrics.csv`), plus any notable **GPU / wall time / loss** lines.

### Pass 1 (measurement) — *pending*

```text
# Example (replace with your actual command and paste output)
# hack26-train --forecast-date all --train-years 2008-2022 --val-year 2023 --test-year 2024 --out-dir measurement -v --allow-download --log-file ~/hack26/data/derived/logs/train_pass1.log
```

### Pass 2 (deliverable retrain) — *pending*

```text
# hack26-train --forecast-date all --train-years 2008-2024 --val-year 2023 --no-test --out-dir final -v --allow-download --log-file ~/hack26/data/derived/logs/train_pass2.log
```

---

## Forecast — `hack26-forecast` — *optional*

```text
# hack26-forecast --year 2025 --all-dates --states ... --model-dir ... --out ... -v --allow-download --log-file ~/hack26/data/derived/logs/run_2025.log
```

---

*Last updated: 2026-04-25 (dataset log captured). Training sections filled when runs complete.*
