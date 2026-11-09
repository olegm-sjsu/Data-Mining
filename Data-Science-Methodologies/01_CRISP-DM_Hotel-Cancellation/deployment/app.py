"""Hotel booking non-arrival risk: Streamlit decision-support app (CRISP-DM project, Phase 6)."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hotel_features as hf  # noqa: E402
import scoring  # noqa: E402

st.set_page_config(page_title="Booking non-arrival risk", page_icon="🏨", layout="wide")


@st.cache_resource
def get_models():
    return scoring.load_models(), scoring.load_policy()


(champion, challenger), policy = get_models()
THR = policy["threshold"]

st.title("Booking non-arrival risk")
st.caption("Decision support for hotel reservations. Retrospective model trained on 2015-2017 bookings of two Portuguese hotels. "
           "It is meant for a randomized pilot, not for automated decisions.")
pilot = st.sidebar.toggle("Pilot mode (randomized hold-out, Groups excluded)", value=True,
                          help="Phase 5 approved only a randomized pilot on non-Group bookings. Turn off only for demos.")

tab1, tab2, tab3 = st.tabs(["Score one booking", "Score a CSV", "About this model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        arrival = st.date_input("Arrival date", dt.date(2017, 6, 15))
        lead_time = st.number_input("Lead time (days between booking and arrival)", 0, 800, 90)
        wk_nights = st.number_input("Week nights", 0, 50, 2)
        we_nights = st.number_input("Weekend nights", 0, 20, 1)
    with c2:
        adults = st.number_input("Adults", 0, 20, 2)
        children = st.number_input("Children", 0, 10, 0)
        babies = st.number_input("Babies", 0, 10, 0)
        meal = st.selectbox("Meal plan", ["BB", "HB", "SC", "FB"])
        room = st.selectbox("Reserved room type", list("ABCDEFGHLP"), index=0)
    with c3:
        segment = st.selectbox("Market segment", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"])
        channel = st.selectbox("Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS"])
        customer = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
        repeated = st.checkbox("Returning guest")
        prev_c = st.number_input("Previous cancellations", 0, 30, 0)
        prev_nc = st.number_input("Previous bookings not cancelled", 0, 80, 0)
        agent = st.number_input("Agent ID (0 = none)", 0, 600, 9)
        company = st.number_input("Company ID (0 = none)", 0, 600, 0)

    row = pd.DataFrame([{
        "hotel": hotel, "lead_time": lead_time, "arrival_date_year": arrival.year, "arrival_date_month": hf.MONTHS[arrival.month - 1],
        "arrival_date_week_number": arrival.isocalendar()[1], "arrival_date_day_of_month": arrival.day,
        "stays_in_weekend_nights": we_nights, "stays_in_week_nights": wk_nights, "adults": adults, "children": children, "babies": babies,
        "meal": meal, "market_segment": segment, "distribution_channel": channel, "is_repeated_guest": int(repeated),
        "previous_cancellations": prev_c, "previous_bookings_not_canceled": prev_nc, "reserved_room_type": room, "customer_type": customer,
        "agent": float(agent) if agent else float("nan"), "company": float(company) if company else float("nan")}])
    res = scoring.score_frame(row, champion, challenger, policy, apply_cap=False)
    p = float(res.p_champion.iloc[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of non-arrival", f"{p:.1%}")
    m2.metric("Outreach threshold", f"{THR:.0%}")
    m3.metric("Recommendation", "Flag" if p >= THR else "No action")
    if pilot:
        pc = scoring.pilot_columns(row, [p >= THR and segment != "Groups"], above=[p >= THR])
        (st.warning if pc.action.iloc[0].startswith("contact") else st.info)(pc.action.iloc[0])
        st.caption(f"Pilot arm: {pc.pilot_arm.iloc[0] or 'not applicable'}. Assignment is a fixed function of the booking details, "
                   "so the same booking always lands in the same arm. Log every outcome in the hotel system.")
    else:
        (st.warning if p >= THR else st.success)(scoring.action_text(p, THR))
    if challenger is not None:
        st.caption(f"Challenger model (LightGBM, shadow only, does not drive the action): {float(res.p_challenger.iloc[0]):.1%}")
    if arrival.month in (7, 8):
        st.info("July and August arrivals: the model was clearly weaker in this period on held-out 2017 data. Treat the score with extra care.")

with tab2:
    st.write("Upload a CSV with the booking columns below (agent, company and children may be empty; an optional `booking_id` column makes "
             f"pilot assignment stable). Flags use the {policy['cap']:.0%} weekly cap: within each arrival week, only the highest-scoring bookings above the threshold are flagged.")
    st.warning("The cap is applied **within this file only**. Upload every booking arriving in each week in one file, "
               "or the cap will not reflect real weekly capacity.")
    st.code(", ".join(hf.RAW_A_COLS))
    up = st.file_uploader("Bookings CSV", type="csv")
    if up is not None:
        try:
            df_in = pd.read_csv(up, na_values=["NULL"])
            elig = (df_in["market_segment"].ne("Groups").values if pilot else None)
            out = scoring.score_frame(df_in, champion, challenger, policy, apply_cap=True, eligible=elig)
            wsz = scoring.week_sizes(df_in)
            if (wsz < 200).any():
                st.warning(f"{int((wsz < 200).sum())} arrival week(s) have fewer than 200 bookings in this file "
                           f"({', '.join(map(str, wsz[wsz < 200].index[:6]))}). The weekly cap may not match real capacity.")
            pcols = scoring.pilot_columns(df_in, out.flag_outreach.values, above=out.above_threshold.values) if pilot else pd.DataFrame(index=df_in.index)
            res_df = pd.concat([df_in.reset_index(drop=True), out.reset_index(drop=True), pcols.reset_index(drop=True)], axis=1)
            st.write(f"{len(res_df):,} bookings, {int(out.flag_outreach.sum()):,} flagged for outreach ({out.flag_outreach.mean():.1%})"
                     + (" (Group bookings never take a slot in pilot mode)." if pilot else "."))
            st.dataframe(res_df.head(200))
            st.download_button("Download scored CSV", res_df.to_csv(index=False).encode(), "scored_bookings.csv", "text/csv")
        except Exception as e:  # show a readable message instead of a traceback
            st.error(f"Could not score this file: {e}")

with tab3:
    st.markdown(f"""
**Model:** logistic regression on 27 features that we *assume* are known when a booking is made (availability is not verified: the source data is a day-before-arrival snapshot) (frozen policy hash `{policy['model_sha256']}`).
**Threshold:** {THR:.2f}. **Weekly cap:** {policy['cap']:.0%} of bookings.
**What it was tested on:** 24,018 bookings arriving 22 April to 31 August 2017. Precision of flagged bookings was 0.72, recall 0.39.

**Known limits (see REPORT.md):**
- Profit depends on the share of flagged guests that outreach actually keeps (break-even 19%), which is unmeasured. Run a randomized pilot first.
- Performance fell in July and August 2017 (flagged precision about 0.52 to 0.54).
- German guests were contacted wrongly about three times as often as Portuguese guests. Monitor this.
- Booking-time availability of the history counters is assumed, not proven.
""")
