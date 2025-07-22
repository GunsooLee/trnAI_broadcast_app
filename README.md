# Home Shopping Broadcast Recommender 📺

AI-driven system that predicts expected sales for each broadcast time-slot and suggests the best product **or** product-category to schedule.

* Python 3.13 + XGBoost
* PostgreSQL (table `broadcast_training_dataset`)
* Streamlit web UI
* Docker-ready (single service)  

---

## 1. Quick Start (server)

```bash
# clone or pull the latest code
cd /opt
git clone https://github.com/<YOUR_ORG>/broadcast_recommender.git
cd broadcast_recommender        # repo root

# build & run container (first time or after updates)
docker compose up -d --build
```

* The container image is built from `Dockerfile` and starts Streamlit on **port 8501**.
* Open `http://SERVER_IP:8501` in your browser ➜ fill in date / weather / slots ➜ click **🚀 추천 실행**.

### 1.1 Stopping / restarting
```bash
docker compose down            # stop
# pull new code & redeploy
git pull
docker compose up -d --build
```

---

## 2. Environment Variables

| Name   | Description | Default (in code) |
|--------|-------------|--------------------|
| `DB_URI` | SQLAlchemy-style PostgreSQL URI | `postgresql://TRN_AI:TRN_AI@localhost:5432/TRNAI_DB` |

Set in `docker-compose.yml`:
```yaml
environment:
  - DB_URI=postgresql://TIKITAKA:TIKITAKA@TIKITAKA_postgres:5432/TIKITAKA_DB
```
Make sure the Streamlit container and DB container share the same Docker network.

---

## 3. Local Development (without Docker)

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model (loads data from DB and saves xgb_broadcast_sales.joblib)
python broadcast_recommender.py train

# Run Streamlit UI
streamlit run streamlit_app.py
```

---

## 4. Training & Recommendation via CLI

```bash
# Train
python broadcast_recommender.py train

# Recommend (product codes)
python broadcast_recommender.py recommend \
    --date 2025-07-24 \
    --time_slots "아침,점심,저녁" \
    --products "P1001,P2002,P3003"

# Recommend (category mode)
python broadcast_recommender.py recommend \
    --date 2025-07-24 \
    --time_slots "아침,점심,저녁" \
    --category
```

---

## 5. Project Structure

```
├─ broadcast_recommender.py   # core training & recommend logic
├─ streamlit_app.py           # web UI
├─ requirements.txt           # python deps
├─ Dockerfile                 # container build
├─ docker-compose.yml         # one-click deploy
└─ README.md                  # this file
```

---

## 6. Troubleshooting

* **`OperationalError: connection refused`**  
  – Confirm `DB_URI` host matches the PostgreSQL container/service name.  
  – Check both containers are on the same Docker network.
* **Model not found**  
  – Run `python broadcast_recommender.py train` inside the container **once** or copy the `.joblib` file.

---

Happy broadcasting! 🎉
