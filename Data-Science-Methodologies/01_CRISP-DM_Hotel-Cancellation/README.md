# Project 1: CRISP-DM, Hotel Booking Cancellation

| | |
|---|---|
| **Question** | Which bookings are likely to cancel, and what should the hotel do about it? |
| **Dataset** | [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), CC BY 4.0, about 119k bookings from a city and a resort hotel, 2015 to 2017 |
| **Source to cite** | Antonio, Almeida and Nunes, "Hotel booking demand datasets", *Data in Brief*, 2019 |
| **Target** | `is_canceled` |
| **Deployment** | Streamlit app on a public URL |
| **Colab** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olegm-sjsu/Data-Mining/blob/main/Data-Science-Methodologies/01_CRISP-DM_Hotel-Cancellation/notebooks/crisp_dm_hotel_bookings.ipynb) (works after the push) |
| **Medium** | _add link_ |
| **YouTube** | _add link_ |

## Status

All six CRISP-DM phases are drafted, run and reviewed (see `critic_reviews/`, 15 review rounds from
GPT-5.6 Terra, final scores 9.1 to 9.4). Read [REPORT.md](REPORT.md) first.

**Still to do by you** (the assignment requires it or it needs your accounts):

- [ ] Read the notebook and be able to explain each cell; recompute two or three numbers by hand
- [ ] Push to GitHub, then run the notebook clean in Colab (needs a `KAGGLE_API_TOKEN` secret)
- [ ] Deploy `deployment/app.py` to Streamlit Community Cloud and add the URL here
- [ ] Publish `article/medium_article.md` on Medium (fill the links) and add the URL here
- [ ] Record the YouTube walkthrough (`article/presentation.html` is the slide deck) and add the URL here
- [ ] Optionally rotate the Kaggle token that was shared in chat


## Pitfalls to hand the LLM and check yourself

- **Direct leakage:** `reservation_status` and `reservation_status_date` reveal the outcome. Remove both.
- **Point-in-time leakage:** ask of every column, "is this known when the prediction is made?" Check `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, and `country`, which the hotel may only confirm at check-in. Read the source paper.
- **Random splits flatter the model.** Split by time. Derive the booking date as arrival date minus `lead_time`.
- **Duplicates:** there is no booking ID, and many rows are exact copies. They may be real group bookings. Investigate before dropping, and keep identical rows from straddling train and test.
- **Odd records:** negative `adr`, zero guests, and heavy missingness in `company` and `agent`. Missing `agent` may mean "no agent", so it may carry signal.

## Checklist

Tasks and outputs follow CRISP-DM 1.0 (Chapman et al., 2000). Write each output
in `phase_notes/`, and save at least two critic rounds per phase.

### 1. Business Understanding, `01_business_understanding.md`

- [x] **Determine business objectives:** background, objectives, business success criteria
- [x] **Assess situation:** resources, requirements, assumptions, constraints, risks and contingencies, terminology, costs and benefits
- [x] Build a cost matrix for false positives (wrongly chasing a good guest, or overbooking) and false negatives (an empty room). Label the dollar figures as assumptions
- [x] **Determine data mining goals:** data mining goals and measurable success criteria, set before modeling
- [x] **Produce project plan:** plan, plus initial assessment of tools and techniques

### 2. Data Understanding, `02_data_understanding.md`

- [x] **Collect initial data:** source, file, how it was fetched, known limits
- [x] **Describe data:** rows, columns, types, grain, data dictionary
- [x] **Explore data:** target rate by hotel, lead time, deposit type, segment, season, with a question answered by each chart
- [x] **Verify data quality:** missingness, duplicates, invalid values, outliers, and when each field gets recorded

### 3. Data Preparation, `03_data_preparation.md`

- [x] **Select data:** written rationale for each included or excluded row group and column
- [x] **Clean data:** cleaning report, every rule with a row count
- [x] **Construct data:** derived attributes such as total nights, total guests, booking date, seasonality
- [x] **Integrate data:** say what you did, or state that a single table needed none
- [x] **Format data:** one scikit-learn pipeline fit on training data only
- [x] Outlier analysis with a justified decision
- [x] Output: final dataset and dataset description

### 4. Modeling, `04_modeling.md`

- [x] **Select modeling techniques:** with the assumptions of each
- [x] **Generate test design:** time-based holdout, time-aware cross-validation, untouched final test
- [x] **Build model:** majority baseline, logistic regression, tree model, boosted model, with parameters and descriptions
- [x] Cluster guests into segments with a visualization, and use it if it helps
- [x] **Assess model:** comparison table with fold variation, then revised parameters

### 5. Evaluation, `05_evaluation.md`

- [x] **Evaluate results** against the business success criteria, including expected cost at the chosen threshold
- [x] Calibration, error slices by hotel, segment, and season, and SHAP read as association only
- [x] Fairness risk of `country`
- [x] **Review process:** what was missed or should be repeated
- [x] **Determine next steps:** list of possible actions and a go or no-go decision

### 6. Deployment, `06_deployment.md`

- [x] **Plan deployment:** build the Streamlit app in `deployment/`, with inputs, outputs, and a recommended action
- [x] **Plan monitoring and maintenance:** drift checks, retraining triggers, rollback
- [x] **Produce final report:** `REPORT.md` and a final presentation
- [x] **Review project:** experience documentation, what you would change
