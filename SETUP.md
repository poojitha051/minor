# Tomato Guard — setup

This project has a **Flask** backend and an **Expo (React Native)** frontend. Do **not** commit real API keys or MongoDB passwords; use a local `backend/.env` file (see below).

## 1. Backend (Flask)

### Commands (Windows PowerShell)

```powershell
cd c:\placements\minor\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment variables

Copy `backend/.env.example` to `backend/.env` and set:

- `MONGO_URI` — your MongoDB connection string (Atlas or local).
- `WEATHER_API_KEY` — OpenWeatherMap API key (optional; mock weather is used if empty).

### Model file

Place your trained Keras model at:

`backend/model/best_tomato_model.keras`

It should match **11 classes** in `class_names.json`, **224×224** RGB input, trained with **pixel values in \[0, 1\]** (divide by 255), consistent with `app.py` preprocessing.

### Run the API

```powershell
# From backend/ with venv active
python app.py
```

The server listens on `http://0.0.0.0:5000` (reachable from other devices on your LAN using your PC’s IP).

Health check: `GET http://<YOUR_IP>:5000/health`

## 2. Frontend (Expo)

### Commands

```powershell
cd c:\placements\minor\frontend
npm install
npx expo start
```

### API base URL

Edit `frontend/utils/api.js`:

- Set `API_HOST` to your computer’s **LAN IP** (e.g. `192.168.1.10`) when using a **physical phone** with Expo Go on the same Wi‑Fi.
- For **Android Emulator** with Flask on the same PC, set `USE_ANDROID_EMULATOR_HOST` to `true` (uses `10.0.2.2`).

### Run on a physical device

1. Install **Expo Go** from the App Store or Play Store.
2. Start the bundler with `npx expo start`.
3. Scan the QR code (Android: Expo Go; iOS: Camera app).

Allow **camera**, **photos**, and **location** when prompted (weather and capture).

## 3. Summary of environment variables

| Variable            | Where        | Purpose                          |
|---------------------|-------------|----------------------------------|
| `MONGO_URI`         | `backend/.env` | Users + prediction history    |
| `WEATHER_API_KEY`   | `backend/.env` | OpenWeatherMap (optional)     |
| `PORT`              | `backend/.env` | Flask port (default 5000)       |
| `API_HOST`          | `frontend/utils/api.js` | Backend IP for the app |

## 4. Production notes

- Serve Flask behind **gunicorn** + HTTPS reverse proxy.
- Replace `http://YOUR_IP:5000` with your **HTTPS API URL** in `api.js`.
- Rotate any credentials that were ever pasted into chat or committed by mistake.
