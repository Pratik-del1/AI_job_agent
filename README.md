# 💼 AI Job Recommendation & Application Assistant

An AI-powered, human-in-the-loop job search and application assistant that discovers job opportunities, evaluates them against a candidate's profile, ranks the best matches, and provides an interactive dashboard for approving applications.

The system can automatically populate application forms while keeping **CAPTCHA solving, final review, and submission under human control**.

---

## 🚀 Overview

The project combines machine learning, job discovery, recommendation scoring, Streamlit, Docker, and browser automation into an end-to-end workflow.

```text
Candidate Resume
       ↓
Resume Parsing
       ↓
AI Matching Model
       ↓
Job Discovery
       ↓
Job Scoring & Ranking
       ↓
Streamlit Dashboard
       ↓
Human Approval
       ↓
Application Queue
       ↓
Browser Automation
       ↓
Form Autofill
       ↓
Human CAPTCHA / Review
       ↓
Manual Submission
```

---

## ✨ Features

- 📄 Resume-based job matching
- 🤖 AI-powered job recommendation
- 🎯 Skill matching
- 💼 Role matching
- 📊 Experience matching
- 🧠 Semantic job-resume similarity
- 🔄 Job refresh and recommendation updates
- 🆕 New job detection
- 🖥️ Streamlit dashboard
- 🔍 Match-score filtering
- 🏷️ Recommendation-category filtering
- ✅ Human approval workflow
- 📋 Application queue
- 🌐 ATS detection
- 📝 Automated application form autofill
- 🌍 Playwright browser automation
- 👤 Human-in-the-loop CAPTCHA handling
- 👀 Manual application review
- ✋ Manual final submission
- 🐳 Dockerized Streamlit environment
- 💻 Host-based visible browser automation

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      Job Sources     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Job Updater      │
                         │                      │
                         │ • Fetch jobs         │
                         │ • Process jobs       │
                         │ • Load ML model      │
                         │ • Score jobs         │
                         │ • Generate rankings  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ recommended_jobs.csv │
                         └──────────┬───────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │           Streamlit Dashboard          │
              │                                        │
              │ • View recommended jobs                │
              │ • Filter by score                      │
              │ • View match breakdown                 │
              │ • View job posting                     │
              │ • Approve / skip applications          │
              └──────────────────────┬─────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ application_queue.csv│
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────┐
                  │     Application Worker         │
                  │                                │
                  │ • Monitor approved jobs       │
                  │ • Detect ATS                  │
                  │ • Open application page       │
                  │ • Autofill candidate details  │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   Visible Chromium   │
                       │                      │
                       │ • Form autofill      │
                       │ • Human CAPTCHA      │
                       │ • Human review       │
                       │ • Manual submission  │
                       └──────────────────────┘
```

---

## 📁 Project Structure

```text
ai_job_agent/
│
├── app/
│   └── app.py
│
├── automation/
│   ├── application_state.py
│   ├── application_worker.py
│   ├── ats_detector.py
│   ├── field_mapper.py
│   ├── form_handler.py
│   └── job_updater.py
│
├── data/
│   ├── recommended_jobs.csv
│   ├── application_queue.csv
│   ├── candidate_profile.json
│   └── ...
│
├── notebook/
│   ├── resume_parsing.ipynb
│   └── models/
│       └── resume-job-matcher-v1/
│
├── checklist.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 🧠 Machine Learning Pipeline

The recommendation system uses the candidate's resume/profile and job descriptions to calculate how well each job matches the candidate.

### 1. Resume Processing

The candidate's resume is processed to extract relevant information such as:

- Skills
- Experience
- Roles
- Technical background
- Relevant keywords

### 2. Job Processing

Job postings are collected and processed to extract information such as:

- Job title
- Company
- Location
- Job description
- Required skills
- Experience requirements

### 3. Matching

Each job is evaluated using multiple matching signals.

#### Skill Match

Measures how closely the candidate's skills align with the skills required by the job.

#### Role Match

Measures how closely the job role matches the candidate's profile.

#### Experience Match

Measures compatibility between the candidate's experience and the experience expected by the job.

#### Semantic Similarity

The matching model evaluates semantic similarity between the candidate profile and the job description.

### 4. Final Score

The individual matching signals are combined into a final recommendation score.

```text
Skill Match
     +
Role Match
     +
Experience Match
     +
Semantic Similarity
     ↓
Final Match Score
```

The jobs are then categorized into recommendation levels:

```text
HIGH PRIORITY
GOOD MATCH
CONSIDER
LOW MATCH
```

---

## 🖥️ Streamlit Dashboard

The Streamlit interface provides a central place to review recommended jobs.

For every job, the dashboard displays:

- Job title
- Company
- Location
- Match score
- Recommendation category
- Skill match
- Role match
- Experience match
- Job posting link
- Application status

Users can:

```text
View Job
    ↓
Review AI recommendation
    ↓
Approve & Apply
```

or:

```text
Skip
```

---

## 🔄 Refreshing Job Recommendations

The dashboard contains a:

**🔄 Refresh Jobs**

button.

When pressed, the system:

```text
Fetch latest jobs
       ↓
Process job data
       ↓
Load matching model
       ↓
Calculate recommendation scores
       ↓
Generate recommended_jobs.csv
       ↓
Reload Streamlit dashboard
```

This allows the recommendation list to be updated without manually running the complete recommendation pipeline.

---

## 📋 Application Queue

Approved jobs are stored in:

```text
data/application_queue.csv
```

The queue allows the application worker to process approved applications separately from the Streamlit interface.

Typical application states include:

```text
USER_APPROVED
IN_PROGRESS
FORM_FILLED
READY_FOR_REVIEW
USER_CONFIRMED
SUBMITTED
FAILED
```

The workflow is:

```text
USER_APPROVED
      ↓
IN_PROGRESS
      ↓
FORM_FILLED
      ↓
READY_FOR_REVIEW
      ↓
Human Review
      ↓
Manual Submission
```

---

## 🤖 Application Automation

Once a user approves a job, the application is added to the queue.

The application worker continuously monitors the queue.

When it finds a:

```text
USER_APPROVED
```

application, it:

1. Detects the ATS.
2. Opens the application URL.
3. Launches a visible Chromium browser.
4. Uploads the resume.
5. Fills candidate information.
6. Fills supported profile links.
7. Leaves unsupported fields for manual completion.
8. Leaves CAPTCHA handling to the user.
9. Leaves final submission to the user.

---

## 👤 Human-in-the-Loop

The system intentionally keeps critical parts of the application process under human control.

The AI does **not**:

- Solve CAPTCHA
- Make decisions on ambiguous job-specific questions
- Click the final Submit button

Instead:

```text
AI
 ↓
Autofill application
 ↓
Human
 ↓
Complete CAPTCHA
 ↓
Review information
 ↓
Answer additional questions
 ↓
Submit application
```

This keeps the final application decision with the user.

---

## 🌐 ATS Detection

Before opening an application, the system detects the ATS platform associated with the job posting.

The automation architecture separates ATS detection from form handling so additional ATS platforms can be added independently.

Example workflow:

```text
Job URL
   ↓
ATS Detection
   ↓
ATS-specific Form Handler
   ↓
Application Autofill
```

---

## 🐳 Docker

The Streamlit application and recommendation system can run inside Docker.

Build the Docker image:

```bash
docker compose build
```

Start the container:

```bash
docker compose up
```

The Streamlit dashboard will be available at:

```text
http://localhost:8501
```

---

## 🖥️ Browser Automation and Docker

The visible browser automation runs on the host machine rather than inside the Docker container.

This is intentional because the application worker launches a visible Chromium browser for human interaction.

The architecture is:

```text
Docker
│
├── Streamlit
├── Job updater
└── Matching model
```

and:

```text
Host Machine
│
└── application_worker.py
        │
        ▼
   Playwright
        │
        ▼
 Visible Chromium
```

This separation allows the Streamlit application to run inside Docker while preserving a visible browser for CAPTCHA and human review.

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Enter the project directory:

```bash
cd ai_job_agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright Chromium

```bash
playwright install chromium
```

---

## ▶️ Running the Application

### Start Streamlit

From the project root:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

### Start the Application Worker

Open a second terminal.

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python automation/application_worker.py
```

The worker continuously monitors the application queue for approved applications.

---

## 🧾 Structured Resume Profile

The `jobagent` package extracts the resume into a validated, structured profile: contact details, skills (with category and evidence), experience, education, projects, technologies, seniority and job preferences.

Extraction uses an LLM with a Pydantic schema. The default provider is Google Gemini: copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (the model name is set there too, in `JOBAGENT_LLM_MODEL`). To use OpenAI instead, install `langchain-openai` and set `JOBAGENT_LLM_PROVIDER=openai`, `JOBAGENT_LLM_MODEL` and `OPENAI_API_KEY`. Without a key, or if the call fails, it falls back to regex parsing, which fills contact details and skills only.

```bash
python -m jobagent.resume.cli
```

The profile is written to `data/resume_profile.json`. Useful options: `--resume PATH`, `--output PATH`, `--no-llm`.

The resume read by default is `JOBAGENT_RESUME_FILE` in `.env`. The job updater (`RESUME_FILE` in `automation/job_updater.py`) and the worker's upload (`resume_path` in `data/candidate_profile.json`) each name the resume separately, so change all three together.

LLM requests time out after 60 seconds, and transient errors (503, 429, timeouts) are retried twice with exponential backoff before falling back to regex parsing. These are adjustable in `.env`; see `.env.example`.

To set your own job preferences, create `data/preferences.json`. Any field you supply replaces what was read from the resume:

```json
{
  "preferred_roles": ["Machine Learning Engineer", "AI Engineer"],
  "preferred_locations": ["Bengaluru", "Remote"],
  "work_modes": ["hybrid"],
  "excluded_roles": ["Sales"]
}
```

The hybrid matcher scores jobs against this profile. The legacy matcher still uses its own resume parser, and form filling still uses `data/candidate_profile.json`.

---

## 🎯 Hybrid Explainable Matching

Two matchers are available, chosen with `JOBAGENT_MATCHER` in `.env`:

- `legacy`: the original scorer in `automation/job_updater.py`, unchanged. This is the default when the variable is not set.
- `hybrid`: the scorer in `jobagent/matching/`. Every score is computed deterministically from the resume profile, the posting and `jobagent/matching/resources/taxonomy.json`. No LLM is involved.

The hybrid score is a weighted mean of five components:

| Component | Default weight | How it is computed |
|---|---|---|
| Skill | 0.30 | Whole-term taxonomy matching; required skills count 1.0, unlabelled 0.7, nice-to-have 0.4 |
| Role | 0.30 | Job title mapped to a role family and compared with your stated roles |
| Experience | 0.20 | Years the posting asks for (or implied by a seniority word) against yours |
| Semantic | 0.10 | Embedding similarity between resume facets and chunks of the cleaned posting |
| Preference | 0.10 | Location, work mode, employment type and excluded roles from `data/preferences.json` |

A component that cannot be scored for a job is shown as `n/a` and its weight is shared among the others. Weights, thresholds, label caps and the embedding model are settings; see `jobagent/config.py` and `.env.example`.

The hybrid matcher needs `data/resume_profile.json` (run `python -m jobagent.resume.cli` first). On each refresh it scores every fetched job into `data/scored_jobs.csv`, then writes the top 10 not already in the application workflow to `data/recommended_jobs.csv`. Jobs are ranked by score; a job being new only breaks ties between scores within one point.

To add a skill, alias or role family, edit `taxonomy.json`. No code change is needed.

### Comparing the matchers

```bash
python -m jobagent.matching.evaluate snapshot                 # freeze the live feed
python -m jobagent.matching.evaluate template eval/snapshots/NAME.json
python -m jobagent.matching.evaluate compare eval/snapshots/NAME.json
```

Fill the `label` column of `eval/labels.csv` with `relevant`, `maybe` or `not_relevant`. The comparison then reports Precision@5, NDCG@10, pairwise ordering accuracy and each component's contribution for both matchers. Without labels it reports no quality metrics.

### Tests

```bash
python -m pytest
```

---

## 🔐 Privacy

The project contains personal candidate information.

The following types of files should **not** be committed to a public repository:

- Personal resume
- Candidate profile
- Application history
- Generated screenshots
- Personal application data

These files should be excluded through `.gitignore`.

Before publishing the repository, verify:

```bash
git status
```

and ensure that no personal information is staged.

---

## 🛡️ Security Considerations

Do not commit:

```text
.env
API keys
Passwords
Authentication tokens
Personal resumes
Private candidate information
```

If credentials are required in the future, use environment variables or a secrets manager instead of hardcoding them.

---

## 📊 Example Workflow

A typical session looks like this:

```text
1. Start Docker
        ↓
2. Open Streamlit
        ↓
3. Click "Refresh Jobs"
        ↓
4. Latest jobs are fetched
        ↓
5. AI scores jobs
        ↓
6. Recommendations appear
        ↓
7. User reviews jobs
        ↓
8. User clicks "APPROVE & APPLY"
        ↓
9. Job enters application queue
        ↓
10. Host worker detects approval
        ↓
11. Browser opens
        ↓
12. Candidate information is autofilled
        ↓
13. User completes CAPTCHA
        ↓
14. User reviews application
        ↓
15. User manually submits
```

---

## 🧩 Main Components

| Component | Purpose |
|---|---|
| `app/app.py` | Streamlit dashboard |
| `automation/job_updater.py` | Fetches jobs and generates recommendations |
| `automation/application_worker.py` | Processes approved applications |
| `automation/application_state.py` | Manages application states |
| `automation/ats_detector.py` | Detects ATS platforms |
| `automation/field_mapper.py` | Maps application fields |
| `automation/form_handler.py` | Handles form autofill |
| `notebook/resume_parsing.ipynb` | Resume processing and model development |
| `recommended_jobs.csv` | Current job recommendations |
| `application_queue.csv` | Approved application queue |

---

## 🚧 Current Limitations

### CAPTCHA

CAPTCHA is handled manually by the user.

### Job-Specific Questions

Questions that require contextual or subjective answers are left for human review.

### Browser Automation

Visible browser automation runs on the host machine rather than inside Docker.

### ATS Coverage

Different ATS platforms may require separate form handlers.

---

## 🔮 Future Improvements

Possible future improvements include:

- Additional job sources
- More ATS integrations
- Improved job deduplication
- Persistent application history
- Application analytics
- Better location matching
- Resume variants for different job categories
- Improved semantic matching
- Interview tracking
- Application success tracking
- Job expiration detection
- Cloud deployment of the recommendation dashboard

---

## 📌 Project Status

### Core Project: Complete

The current system provides an end-to-end workflow:

```text
Resume
   ↓
Job Discovery
   ↓
AI Matching
   ↓
Job Ranking
   ↓
Streamlit Recommendations
   ↓
Human Approval
   ↓
Application Queue
   ↓
Browser Automation
   ↓
Form Autofill
   ↓
Human CAPTCHA
   ↓
Human Review
   ↓
Manual Submission
```

The project is designed as a personal AI-powered job search and application assistant with a human-controlled final application process.

---

## 👨‍💻 Author

**Pratik Srivastava**

AI / ML • Data • Automation • Web Development
