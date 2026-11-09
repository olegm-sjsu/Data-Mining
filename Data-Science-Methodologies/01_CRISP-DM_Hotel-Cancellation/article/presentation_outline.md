# Final presentation outline (10 slides, about 10 minutes)

1. **Question and stakes:** 37% of bookings do not arrive; what should a hotel do? Dataset and decision.
2. **Define success first:** cost model, break-even 0.53, ten pre-registered criteria (B1-B4, D1-D6).
3. **Data understanding:** lead-time and segment charts (`02_eda_rates.png`); duplicates; the Non Refund anomaly.
4. **The leak:** parking spaces 0.0% for non-arrivals; Set A vs Set B; PR-AUC 0.72 to 0.85 with Set B.
5. **Preparation and validation:** arrival-date split, sealed test, one pipeline fit on training only; amendment A1.
6. **Modeling:** baselines vs logistic regression vs LightGBM; folds with net benefit; D1 failed honestly.
7. **Test results:** table (precision 0.72, recall 0.39, net benefit 1,533 vs -160); criteria scoreboard (4 fail).
8. **What breaks:** cost sensitivity (break-even save rate 0.19), July-August collapse, country false-positive gap.
9. **Decision and pilot design:** no-go for automation; randomized pilot, about 710 per arm; challenger LightGBM.
10. **Deployment, monitoring, lessons:** app demo, PSI and precision alerts, the critic loop and what I would change.
