// StainScan Offline-First Service Worker
const CACHE_NAME = "stainscan-v1.0";
const ASSETS = [
  "./",
  "./index.html",
  "./style.css?v=3.7",
  "./app.js?v=2.0",
  "./stainscan_logo.png",
  "./manifest.json"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((k) => {
          if (k !== CACHE_NAME) return caches.delete(k);
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  // Only cache GET requests for local assets, never intercept backend POST/predict APIs
  if (e.request.method !== "GET" || e.request.url.includes("/predict") || e.request.url.includes("/api/")) {
    return;
  }
  e.respondWith(
    caches.match(e.request).then((res) => {
      return res || fetch(e.request).catch(() => caches.match("./index.html"));
    })
  );
});
