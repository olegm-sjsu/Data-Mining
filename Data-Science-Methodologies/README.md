# Data Science Methodologies: CRISP-DM, SEMMA, KDD

## Video Walkthroughs (YouTube)

- **Project 1 — CRISP-DM (Hotel Booking Cancellation):** [https://youtu.be/sFLikESPdSA](https://youtu.be/sFLikESPdSA)
- **Project 2 — SEMMA (Lending Club Loan Default):** [https://youtu.be/sDaF-TPgXS8](https://youtu.be/sDaF-TPgXS8)
- **Project 3 — KDD (NSL-KDD Network Intrusion):** [https://youtu.be/Y5Y1Hvx6ipc](https://youtu.be/Y5Y1Hvx6ipc)

---

Three end-to-end projects, one per methodology, each on its own dataset. Every
phase is drafted with an LLM, reviewed by an LLM critic through several
revisions, and verified by hand. Each project ends with a Colab notebook, a
report, a deployed artifact, a Medium article, and a YouTube walkthrough.

## The three projects

| Folder | Methodology | YouTube Walkthrough | Dataset (Kaggle slug) | Question | Why this pairing | Deployment |
|---|---|---|---|---|---|---|
| [01_CRISP-DM_Hotel-Cancellation](01_CRISP-DM_Hotel-Cancellation/README.md) | CRISP-DM | [Watch Video](https://youtu.be/sFLikESPdSA) | `jessemostipak/hotel-booking-demand` | Which bookings will cancel, and what should the hotel do about it? | Clear business decision and costs, which is what Business Understanding needs | Streamlit app on a public URL |
| [02_SEMMA_LendingClub-Default](02_SEMMA_LendingClub-Default/README.md) | SEMMA | [Watch Video](https://youtu.be/sDaF-TPgXS8) | `wordsforthewise/lending-club` | Which loans will charge off? | The file is too large to use whole, so the Sample phase does real work. Credit scoring is also classic SAS territory | Batch scoring script plus SAS score code if available |
| [03_KDD_NSL-KDD-Intrusion](03_KDD_NSL-KDD-Intrusion/README.md) | KDD | [Watch Video](https://youtu.be/Y5Y1Hvx6ipc) | `hassan06/nslkdd` | What patterns separate attacks from normal traffic, and do they hold on attack types never seen in training? | KDD is about discovering knowledge, and this set supports clustering, rules, and anomaly detection | FastAPI service in Docker |

These are recommendations, so swap any of them if you prefer. Each must stay
different from the others and from what you already used in HW1 (PetFinder, NYC
taxi, credit card fraud, online retail).

Papers with Code no longer works as a source. `paperswithcode.com/datasets` now
redirects to Hugging Face, so stick with Kaggle.

## Folder layout (same in every project)

```
<project>/
├── README.md         phase-by-phase checklist for this project
├── notebooks/        the one Colab notebook, runnable top to bottom
├── phase_notes/      one Markdown file per phase: findings, decisions, evidence
├── critic_reviews/   every critique round, saved as evidence
├── artifacts/        figures, tables, trained models, metrics JSON
├── deployment/       the app or service plus how to run it
├── article/          Medium draft and images
├── REPORT.md         the final detailed report (write this in step 7)
└── data/             downloaded data, ignored by git
```

The shared prompts are in [CRITIC_PROMPTS.md](CRITIC_PROMPTS.md).

## What you have to do, step by step

Do the projects in order. CRISP-DM goes first because you already did one in
HW1, and steps 2 to 10 repeat for each project.

### Setup, once

**Step 1. Get your accounts and tools ready.**

- [x] Add your Kaggle credentials to Colab Secrets, so notebooks can download data.
- [x] Confirm you have access to two LLMs, ideally Claude and GPT-5. Use one to draft and the other to critique, so no model grades its own work.
- [x] Apply for SAS Viya for Learners with your SJSU email (needed for SEMMA). Do this first in case access takes a while. After login, check whether Model Studio is included. If SAS does not work out, the assignment says "if possible", so do SEMMA in Python and state why in the report.
- [x] Have a Medium account and a YouTube channel ready.

Open Interpreter, MetaGPT, and IBM SPSS Modeler are optional. Claude Code or
Codex already cover what the first two would do, so you can skip them.

### For each project

**Step 2. Open the project's README and set up the notebook.** Create one
notebook in `notebooks/` with a section heading per phase. First cells: pin
package versions, set a random seed, download the data into `data/`. Keep the
raw data untouched.

**Step 3. Draft phase 1.** Use the generator prompt from `CRITIC_PROMPTS.md`.
Ask for one small chunk at a time. Run the code yourself and write down what the
output actually says in `phase_notes/`.

**Step 4. Run the critic loop on that phase.** Paste the phase's text, code, and
real outputs into a fresh chat using the critic prompt with the "world renowned
authority" persona. Fix what is valid. Re-submit. Do at least two rounds per
phase. Save each round in `critic_reviews/` with your notes on what you accepted
and rejected, because this log is your proof that you did the required revisions.

**Step 5. Verify by hand before moving on.** The assignment requires this, and
no LLM can do it for you. For each phase:

- [x] Read every line of generated code and be able to explain it on video.
- [x] Recompute two or three key numbers a different way (for example, a groupby by hand against the reported rate).
- [x] Confirm nothing fit on the full data leaks into test (imputers, encoders, scalers, sampling).
- [x] Confirm every claim in the text matches a real output in the notebook.

**Step 6. Repeat steps 3 to 5 for every remaining phase.** The project README
lists every subtask. Go back to an earlier phase when you learn something that
breaks it, and record why. Those loops are a strength, so show them.

**Step 7. Write the report and finish the deployment.** Build the deployment in
`deployment/` and test it from a clean environment. Write `REPORT.md`: objective,
data, every phase with its evidence, model comparison, limitations, and next
actions. Then run the critic loop on the whole report.

SEMMA and KDD officially stop before deployment. The assignment still wants
"all the way to deployment", so add a clearly labeled extension section instead
of pretending it is part of the method.

**Step 8. Make it a proper Colab and push it.**

- [x] Restart and run all in Colab with a clean runtime. Save the notebook with its outputs.
- [x] Add an Open in Colab badge to the project README. Your repo is public, so this link pattern works: `https://colab.research.google.com/github/olegm-sjsu/Data-Mining/blob/main/Data-Science-Methodologies/<project>/notebooks/<file>.ipynb`
- [x] Commit everything except `data/`. Keep model files small enough for GitHub (under 100 MB each).

**Step 9. Write and publish the Medium article.** Lead with the question and
result, walk through each phase with your charts, name the mistakes the critic
caught, and link the Colab and repo. Run the article critic prompt, then publish.
Put the link in the project README.

**Step 10. Record the YouTube walkthrough.** Cover both the code and the
artifacts: notebook, critic log, report, and the live deployment. Upload it and
put the link in the project README.

### Final submission

**Step 11. Check all of this before you submit.**

| Item | CRISP-DM | SEMMA | KDD |
|---|---|---|---|
| Every phase and subtask checked off in the project README | [x] | [x] | [x] |
| At least 2 critic rounds per phase saved | [x] | [x] | [x] |
| Colab runs clean top to bottom | [x] | [x] | [x] |
| `REPORT.md` critiqued and finished | [x] | [x] | [x] |
| Deployment works, with run instructions | [x] | [x] | [x] |
| Medium article drafted in `article/medium_article.md` | [x] | [x] | [x] |
| YouTube video uploaded and linked | [x] ([Video](https://youtu.be/sFLikESPdSA)) | [x] ([Video](https://youtu.be/sDaF-TPgXS8)) | [x] ([Video](https://youtu.be/Y5Y1Hvx6ipc)) |
| Pushed to GitHub | [x] | [x] | [x] |

Also for SEMMA: SAS demo and report included, or the reason it was not possible.

## Worth doing in every project

The course deck's example prompt asks for more than a single model: clustering
with visualizations, outlier analysis, associative patterns, LSH, and a final
summary with recommendations. The assignment text does not list these, but
clustering and outlier analysis are cheap and fit all three projects, so include
them. Association rules and LSH fit best in KDD.
