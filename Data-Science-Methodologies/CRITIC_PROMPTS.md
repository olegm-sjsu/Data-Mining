# Prompts for drafting and critiquing

Replace the `{{...}}` placeholders. Always open a **fresh chat** for the critic,
and preferably a different model from the one that drafted the work.

| Project | `{{METHODOLOGY}}` | `{{PHASE_LIST}}` |
|---|---|---|
| 01 | CRISP-DM | Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment |
| 02 | SEMMA | Sample, Explore, Modify, Model, Assess |
| 03 | KDD (Fayyad, Piatetsky-Shapiro and Smyth, 1996) | Selection, Preprocessing, Transformation, Data Mining, Interpretation/Evaluation |

## 1. Generator prompt (adapted from the course deck)

```text
As an industry expert data scientist and expert in {{METHODOLOGY}} methodology,
act as a professor teaching a master's program in data science. Conduct a
step-by-step analysis of the attached dataset following {{METHODOLOGY}}, at
textbook quality, covering every phase, subtask, and report: {{PHASE_LIST}}.

Project: {{ONE_LINE_PROJECT_QUESTION}}
Current phase: {{PHASE}}
Required subtasks for this phase: {{PASTE_THE_PHASE_CHECKLIST_FROM_PROJECT_README}}
Known pitfalls in this dataset: {{PASTE_PITFALLS_FROM_PROJECT_README}}

Rules:
- Use scikit-learn and other standard packages. Code must run in Google Colab.
- Fit every imputer, encoder, scaler, and selector on training data only.
- Respond in small chunks and assume limited compute. I will say when to continue.
- At the end of each chunk, repeat the requirement, say where we are in the
  process, and say what comes next.
- Never state a result you have not computed. Mark assumptions as assumptions.
```

## 2. Critic prompt, first round

```text
You are a world-renowned speaker and the foremost authority on {{METHODOLOGY}}.
You have written many award-winning books on this topic and have reviewed
hundreds of real data science projects. You are reviewing the {{PHASE}} phase of
a graduate project. Your reputation depends on the rigor of this review, so do
not be polite at the expense of accuracy.

Project: {{ONE_LINE_PROJECT_QUESTION}}
Required subtasks: {{PASTE_THE_PHASE_CHECKLIST_FROM_PROJECT_README}}

Below are the write-up, the code, and the real outputs.

{{PASTE_PHASE_NOTES_CODE_AND_OUTPUTS}}

Give me, in this order:
1. A table of every required subtask: Complete, Partial, or Missing, with the
   evidence you relied on.
2. Methodology errors: places where this departs from {{METHODOLOGY}} as defined
   by its original sources.
3. Technical errors: leakage, invalid validation, wrong metrics, unsupported
   claims, and any number in the text that the outputs do not support.
4. The 5 highest-value fixes, in priority order, each specific enough to act on.
5. A score out of 10, and what specifically would raise it by one point.

Only judge what is shown. If you cannot verify something from what I gave you,
say "cannot verify" instead of assuming it is right or wrong. Do not rewrite the
work.
```

## 3. Critic prompt, later rounds

Use a fresh chat again, and paste the previous critique.

```text
{{SAME_PERSONA_PARAGRAPH_AS_ABOVE}}

This is revision {{N}}. Here is the prior critique, and my response to each point:

{{PREVIOUS_CRITIQUE}}
{{MY_RESPONSE_WHAT_I_CHANGED_AND_WHAT_I_REJECTED_AND_WHY}}

Here is the revised work with real outputs:

{{REVISED_NOTES_CODE_AND_OUTPUTS}}

For each prior point, say Resolved, Partly resolved, or Not resolved, with
evidence. Where I rejected a point, say whether my reason holds. Then list any
new issues, updated subtask status, a new score out of 10, and whether this
phase is ready to sign off.
```

**When to stop:** at least two rounds per phase. Stop when no subtask is Missing,
the score is 9 or above, and the critic is down to minor style points. If you
disagree with a point, you do not have to accept it, but write down why.

## 4. Code verification prompt

This supports your manual check and does not replace it.

```text
You are a senior ML engineer auditing this notebook code for correctness. Find
data leakage, misuse of train/test, wrong joins, silent type coercion, row loss,
metric mistakes, non-reproducible steps, and anything that only works because of
notebook state. For each issue give the cell, why it is wrong, a minimal fix,
and a quick test I can run myself to confirm it.

{{CODE}}
```

## 5. Article critic prompt

```text
You are a world-renowned speaker and authority on {{METHODOLOGY}}, the author of
many award-winning books on it, and a top Medium editor for data science.
Critique this article for accuracy against the notebook results below, faithful
use of {{METHODOLOGY}}, clarity for a technical reader, and a strong opening.
List every claim the results do not support. Give the 5 highest-value edits.

Article: {{ARTICLE}}
Results: {{KEY_TABLES_AND_METRICS}}
```

## How to save each round

Save as `critic_reviews/<phase number>_<phase name>_round<k>.md`, for example
`critic_reviews/03_data_preparation_round2.md`.

```markdown
# <Phase>, round <k>

- **Date:**
- **Critic model:**
- **Score:** before ? / after ?

## Prompt sent
## Critique received (verbatim)
## What I changed
## What I rejected, and why
## What I verified by hand
```
