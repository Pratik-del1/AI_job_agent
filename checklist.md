# Referral Recommendation System — Project Checklist

## Phase 1 — Dataset Understanding

- [x] Load `all_job_post.csv`
- [x] Load `resume_data_for_ranking.csv`
- [x] Check number of rows and columns
- [x] Check column names
- [x] Check data types
- [x] Check missing values
- [x] Check duplicate rows
- [x] Understand what each column represents

## Phase 2 — Data Cleaning

- [x] Remove duplicate rows
- [x] Standardize column names
- [x] Handle missing values
- [ ] Normalize text
- [ ] Clean list-like columns
- [ ] Validate cleaned dataset

## Phase 3 — EDA

- [ ] Analyze common skills
- [ ] Analyze job titles
- [ ] Analyze education
- [ ] Analyze experience
- [ ] Analyze locations
- [ ] Create visualizations

## Phase 4 — Candidate Profile

- [ ] Create `candidate_profile`
- [ ] Combine relevant resume fields
- [ ] Inspect sample profiles

## Phase 5 — Job Profile

- [ ] Create `job_profile`
- [ ] Combine relevant job fields
- [ ] Inspect sample job profiles

## Phase 6 — TF-IDF Baseline

- [ ] Create TF-IDF vectors
- [ ] Calculate cosine similarity
- [ ] Rank jobs
- [ ] Return Top-10 recommendations

## Phase 7 — Skill Matching

- [ ] Extract candidate skills
- [ ] Extract job skills
- [ ] Normalize skills
- [ ] Calculate skill overlap

## Phase 8 — Hybrid Recommendation

- [ ] Add text similarity
- [ ] Add skill score
- [ ] Add experience score
- [ ] Add education score
- [ ] Add location score
- [ ] Create final ranking score

## Phase 9 — Semantic Matching

- [ ] Choose embedding model
- [ ] Generate resume embeddings
- [ ] Generate job embeddings
- [ ] Calculate embedding similarity
- [ ] Compare against TF-IDF

## Phase 10 — Evaluation

- [ ] Create validation dataset
- [ ] Define relevance
- [ ] Calculate Precision@K
- [ ] Calculate Recall@K
- [ ] Calculate NDCG@K
- [ ] Compare baseline vs hybrid model

## Phase 11 — ML Ranking

- [ ] Create resume-job pair features
- [ ] Train baseline model
- [ ] Try XGBoost/LightGBM
- [ ] Evaluate ranking performance

## Phase 12 — API

- [ ] Create FastAPI backend
- [ ] Create `/recommend` endpoint
- [ ] Return Top-N jobs

## Phase 13 — Frontend

- [ ] Create resume input/upload
- [ ] Display recommended jobs
- [ ] Display match percentage
- [ ] Display matched skills
- [ ] Display missing skills

## Phase 14 — Production

- [ ] Add logging
- [ ] Add tests
- [ ] Create requirements.txt
- [ ] Dockerize
- [ ] Write README
- [ ] Deploy