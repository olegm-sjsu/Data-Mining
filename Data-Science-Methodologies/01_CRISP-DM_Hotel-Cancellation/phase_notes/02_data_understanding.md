# Phase 2: Data Understanding

Draft v2 (after critic round 1). Numbers come from
`notebooks/crisp_dm_hotel_bookings.ipynb` (section 2), `artifacts/metrics_phase2.json`
and `artifacts/figures/02_eda_rates.png`.

**Sealed test rows.** The final test period is fixed by date before any outcome
is inspected: the 80th percentile of arrival dates, **2017-04-22**. Every
target-based statistic in this note uses the **95,349 development rows**
(arrivals before that date). The 24,041 test rows (24,018 after cleaning) are
not used for any target-based analysis. Row counts of the whole file
(sections 2.1, 2.2) and label-free counts are marked "whole file". See the
exposure log in `03_data_preparation.md`.

## 2.1 Collect initial data

- **Source:** Kaggle `jessemostipak/hotel-booking-demand`, CC BY 4.0, from
  Antonio, de Almeida and Nunes (2019), *Data in Brief* 22.
- **How fetched:** Kaggle CLI with an API token into `data/hotel_bookings.csv`
  (16.9 MB, never edited). The string `NULL` in `agent` and `company` is read as
  missing.
- **Known limits:** no booking ID; one table; arrivals only from 2015-07-01 to
  2017-08-31; values are a day-before-arrival snapshot (Phase 1); two hotels in
  Portugal; anonymized.

## 2.2 Describe data

119,390 rows (whole file), 32 columns, one row per booking. Dictionary (dtype,
missing and unique counts are for the whole file; the audit set is A, B or
excluded):

| Column | Meaning | Set | Dtype | Missing | Unique |
|---|---|---|---|---|---|
| `hotel` | City Hotel (H2, Lisbon) or Resort Hotel (H1, Algarve) | A | object | 0 | 2 |
| `is_canceled` | Target: 1 = cancelled or no-show | target | object | 0 | 2 |
| `lead_time` | Days between booking and arrival | A | object | 0 | 479 |
| `arrival_date_year` | Arrival year (used to order data in time, not as a feature) | split only | object | 0 | 3 |
| `arrival_date_month` | Arrival month | A | object | 0 | 12 |
| `arrival_date_week_number` | Arrival week number | A | object | 0 | 53 |
| `arrival_date_day_of_month` | Arrival day of month | A | object | 0 | 31 |
| `stays_in_weekend_nights` | Weekend nights booked | A | object | 0 | 17 |
| `stays_in_week_nights` | Week nights booked | A | object | 0 | 35 |
| `adults` | Number of adults | A | object | 0 | 14 |
| `children` | Number of children (4 missing) | A | object | 4 | 5 |
| `babies` | Number of babies | A | object | 0 | 5 |
| `meal` | Meal plan (Undefined = SC) | A | object | 0 | 5 |
| `country` | Guest country (can be unknown until check-in) | B | object | 488 | 177 |
| `market_segment` | Market segment | A | object | 0 | 8 |
| `distribution_channel` | Booking channel | A | object | 0 | 5 |
| `is_repeated_guest` | 1 if returning guest | A | object | 0 | 2 |
| `previous_cancellations` | Earlier cancellations by the guest | A | object | 0 | 15 |
| `previous_bookings_not_canceled` | Earlier non-cancelled bookings | A | object | 0 | 73 |
| `reserved_room_type` | Room type booked | A | object | 0 | 10 |
| `assigned_room_type` | Room type assigned (assigned late) | B | object | 0 | 12 |
| `booking_changes` | Changes made to the booking | B | object | 0 | 21 |
| `deposit_type` | No Deposit / Refundable / Non Refund (suspect, see 2.4) | B (flagged) | object | 0 | 3 |
| `agent` | Travel agent ID (missing = no agent) | A | object | 16,340 | 333 |
| `company` | Company ID (94% missing) | A | object | 112,593 | 352 |
| `days_in_waiting_list` | Days on waiting list | B | object | 0 | 128 |
| `customer_type` | Contract / Group / Transient / Transient-Party | A | object | 0 | 4 |
| `adr` | Average daily rate | B | object | 0 | 8879 |
| `required_car_parking_spaces` | Parking spaces required | B | object | 0 | 5 |
| `total_of_special_requests` | Number of special requests | B | object | 0 | 6 |
| `reservation_status` | Check-Out / Canceled / No-Show (encodes the target) | X | object | 0 | 3 |
| `reservation_status_date` | Date of last status (encodes the outcome) | X (split logic only) | object | 0 | 926 |

Summary: 20 Set A columns (plus `arrival_date_year`, an input used only to order rows and derive the arrival weekday, so the app asks for 21 raw inputs), 7 Set B columns plus `deposit_type` (flagged),
2 excluded status columns, 1 target.

| Fact | Value |
|---|---|
| Development rows | 95,349 |
| Non-arrival rate (dev) | 36.05% (cancellations 33,341; no-shows 1,035; stays 60,973) |
| City / Resort (dev) | 41.28% / 25.94% |
| Arrival range (whole file) | 2015-07-01 to 2017-08-31 |
| Derived booking date range | 2013-06-24 to 2017-08-31; 2,637 before 2015 |
| Lead time (whole file) | median 69 days, mean 104, max 737 |

## 2.3 Explore data

Each chart (`02_eda_rates.png`) answers one question, on development rows.

| Question | Finding |
|---|---|
| Q1 Does the hotel matter? | Yes: 41.3% (city) vs 25.9% (resort) |
| Q2 Do earlier bookings cancel more? | Strongly. By lead time: 0-7 days 9.5%, 8-30 27.3%, 31-90 37.4%, 91-180 43.4%, 181-365 57.5%, 366+ 80.6% (2,052 rows) |
| Q3 Why is Non Refund about 99%? | See 2.4. Non Refund 99.2% (12,235 rows), No Deposit 26.8% (82,974), Refundable 10.7% (140) |
| Q4 Which segments cancel most? | Groups 60.3%, Online TA 34.7%, Offline TA/TO 34.1%; Direct 14.5%, Corporate 18.0%, Complementary 12.9%, Aviation 22.1% |
| Q5 Seasonality by arrival month? | City 37% to 45% across months. Resort from 14.8% (January) to 32.4% (September) |
| Q6 Drift across arrival quarters? | Quarterly rates run from 30.8% (2016Q1) to 42.1% (2015Q3); 2017Q2 (first 21 days) is 42.3%. By hotel, the resort moves between 21% and 32% and the city between 37% and 50%, so the rate is not stable over time |

## 2.4 Verify data quality (development rows unless stated)

| Check | Result | Handling (Phase 3) |
|---|---|---|
| Missing | `children` 4; `country` 483; `agent` 13,780 (14.5%); `company` 89,423 (93.8%) | children = 0; missing agent kept as "no agent"; company becomes a flag |
| Zero guests | 157 rows, non-arrival rate 13.4% | drop |
| Zero nights | 665 rows, 4.7% | keep, flag |
| Negative / extreme `adr` | 1 row at negative, 1 row at 5,400 | drop |
| Long tails | lead time to 737; 12 rows with more than 10 adults | keep rows, clip values at the training 99.9th percentile |
| `meal` Undefined | 1,055 rows | merge into SC |
| Full-row duplicates | 26,936 rows (28.2%). 33,551 rows sit in a duplicate group, with non-arrival rate 57.9% vs 24.2% outside; mostly Groups (12,788), Offline TA/TO (8,928), Online TA (3,534) | keep, since without an ID they cannot be told from group bookings; sensitivity in Phase 4 |

### The `deposit_type` anomaly

12,235 development bookings are Non Refund and 99.25% do not arrive. They are
96.7% Portuguese guests, mostly in the city hotel (10,627 rows, 99.8%), in the
Groups (7,631, 99.2%) and Offline TA/TO (4,241, 99.9%) segments, and 89.1% have
an agent. The five busiest agents cover 50.8% of them (agent 1 alone 4,052). The
same five agents have a 26.4% non-arrival rate on other deposit types, so it
is the deposit label, not the agents, that carries the near-certainty. Only 92
Non Refund bookings arrived.

A non-refundable rate that almost never turns into a stay is commercially
implausible. The most likely explanation is a recording effect (the type set or
changed when a cancelled booking was handled), but **this is a hypothesis, not a
finding**. The data cannot separate it from an unusual operational pattern,
and the paper's own definition of the field has not been read by us.

**Decision:** `deposit_type` moves out of Set A into the diagnostic Set B. The
Phase 4 comparison measures what it adds.

### Evidence on when fields are recorded

| Field | Arrived | Did not arrive |
|---|---|---|
| Assigned room differs from reserved | 21.0% | 2.0% |
| Mean `booking_changes` | 0.284 | 0.093 |
| Required parking space | 10.1% | **0.0%** |
| Mean special requests | 0.67 | 0.29 |
| Mean `days_in_waiting_list` | 1.94 | 4.49 |
| Mean `adr` | 92.37 | 95.64 |
| Country missing | 0.69% | 0.18% |

A parking space is recorded for 0.0% of non-arrivals, so the field is only
filled in for guests who show up. A room can only be assigned to a guest who is
expected, and changes accumulate until check-in. These fields behave as
post-booking and stay in Set B.

### Availability evidence for Set A: what this project can and cannot show

The data has no booking-time log, so availability of Set A fields at booking
cannot be proven from it. The classification rests on definition, with two
levels, and Phase 4 tests how much it matters:

| Level | Fields | Basis |
|---|---|---|
| **A1, fixed by the reservation request** | hotel, lead time, arrival month/week/day, nights, reserved room type, meal, market segment, distribution channel, agent, company, customer type, adults, children, babies | Part of what a booking is. Party size and meal can still be revised later, a residual risk |
| **A2, history counters** | `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled` | Counts from the guest's other bookings. In a day-before-arrival snapshot they may include bookings made after this one, so they are the least certain Set A fields |

Phase 4 compares A1 alone, A1+A2 (= Set A) and Set A + B. The deployment claim
stays conditional: a retrospective benchmark, not proof that every Set A field
is available at the moment of booking.

## Field audit result and decision gate

Set A, Set B and the excluded columns are as in the dictionary above.

**The formal gate (is Set A strong enough?) cannot be settled by univariate
signal.** Univariate signal is encouraging (lead time 9.5% to 80.6%, segment
12.9% to 60.3%, hotel 25.9% vs 41.3%) and is recorded as such. The gate is
decided in Phase 4 on validation-period results only (Phase 1,
Amendment A2), never on the test period. Until then Phase 3 proceeds on Set A. If Set A fails, the
project continues as an explicitly retrospective study.

## Open items carried forward

1. Read the source paper's field definitions (not done).
2. The deposit anomaly remains a hypothesis.
3. Time drift in the target rate (Q6) is expected to appear in calibration.
