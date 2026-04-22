/**
 * API helpers — set API_HOST to your computer's LAN IP when testing on a physical device.
 * Android emulator: set USE_ANDROID_EMULATOR_HOST to true to use 10.0.2.2.
 */
import { Platform } from "react-native";
import axios from "axios";


/** Replace with your machine's IP (e.g. 192.168.1.10) for real devices on the same Wi‑Fi. */
export const API_HOST = "192.168.0.7";

/** Your PC's LAN IP — same Wi‑Fi as the phone. Change if your IP changes. */
export const API_HOST = "10.100.13.70";


export const API_PORT = "5000";

/** Set true only when running the app on Android Emulator and Flask on the same PC. */
export const USE_ANDROID_EMULATOR_HOST = false;

const ANDROID_EMULATOR_HOST = "10.0.2.2";

export function getApiBaseUrl() {
  if (Platform.OS === "android" && USE_ANDROID_EMULATOR_HOST) {
    return `http://${ANDROID_EMULATOR_HOST}:${API_PORT}`;
  }
  return `http://${API_HOST}:${API_PORT}`;
}

export const API_BASE_URL = getApiBaseUrl();

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

/**
 * Upload image for /predict (multipart field name must be "image").
 */
export async function predictDisease(imageUri) {
  const form = new FormData();
  const name = imageUri.split("/").pop() || "leaf.jpg";
  const ext = name.split(".").pop()?.toLowerCase();
  const type =
    ext === "png"
      ? "image/png"
      : ext === "webp"
        ? "image/webp"
        : "image/jpeg";

  form.append("image", {
    uri: imageUri,
    name: name.endsWith(".") ? "leaf.jpg" : name,
    type,
  });

  const res = await client.post("/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function fetchWeather(lat, lon) {
  const res = await client.get("/weather", { params: { lat, lon } });
  return res.data;
}

export async function signupUser({ name, phone, password }) {
  const res = await client.post("/signup", { name, phone, password });
  return res.data;
}

export async function loginUser({ phone, password }) {
  const res = await client.post("/login", { phone, password });
  return res.data;
}

export async function saveHistory(payload) {
  const res = await client.post("/history", payload);
  return res.data;
}

export async function getHistory(phone) {
  const res = await client.get(`/history/${encodeURIComponent(phone)}`);
  return res.data;
}
