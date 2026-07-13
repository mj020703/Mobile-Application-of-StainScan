/* ==========================================================================
   STAINSCAN MOBILE APP CLIENT-SIDE CONTROLLER
   Simulates Convolutional Neural Network (CNN) feature extraction models,
   runs interactive cleaning checklists for users.
   ========================================================================== */

// --- Backend API URL Configuration ---
// Modify this URL to point to your live cloud server (e.g., Render) when deployed.
const CONFIG = {
    API_BASE_URL: "https://stainscan-backend-gmbw.onrender.com"
};

// --- Default Knowledge Base Recommendations ---
const defaultKnowledgeBase = {
    "Used Cooking Oil": {
        materials: ["Liquid Dish Soap", "Warm Water", "Microfiber Cloth", "Baking Soda"],
        steps: [
            "Blot the excess oil immediately using a clean paper towel. Do not rub, as this spreads the oil.",
            "Apply a generous amount of liquid dish soap directly to the stained area. Dish soap is designed to cut grease.",
            "Gently work the soap into the cotton fabric fibers with a soft cloth or toothbrush in circular motions.",
            "Let it stand for 5-10 minutes to allow the soap to break down the oil structure.",
            "Rinse the area thoroughly with warm water to flush out the grease-soap emulsion.",
            "Launder standardly at the highest safe temperature for the garment, then check the area before machine drying."
        ]
    },
    "Black Ballpen Ink": {
        materials: ["Isopropyl Alcohol", "Cotton Balls", "Absorbent Towels", "Liquid Detergent"],
        steps: [
            "Place an absorbent paper towel directly underneath the stained layer of the fabric to catch bleeding ink.",
            "Dab the stain generously using a cotton ball saturated with isopropyl alcohol.",
            "Blot repeatedly, switching to fresh cotton balls as they absorb the ink. Do not scrub, blot only.",
            "Rinse the stained fabric section thoroughly with cold water to remove the alcohol.",
            "Rub a small amount of liquid detergent into any remaining faint ink outline.",
            "Wash immediately in a regular laundry cycle, verifying the stain is gone before applying heat drying."
        ]
    },
    "Mud": {
        materials: ["Laundry Brush", "Liquid Laundry Detergent", "Warm Water", "White Vinegar"],
        steps: [
            "Allow the mud to dry completely. Attempting to clean wet mud will rub dirt deeper into cotton fibers.",
            "Scrape or brush off dry mud crust using a stiff-bristled brush.",
            "Pre-treat the remaining dirt spots with a small amount of liquid laundry detergent.",
            "Rub the fabric together gently under warm running water to release dirt particles.",
            "For stubborn brown mud stains, mix equal parts warm water and white vinegar, sponge the area, and let sit for 10 minutes.",
            "Rinse clean and launder normally in a warm wash cycle."
        ]
    }
};

// --- Live Cloud API Integration Datastore ---
class DataStore {
    static getApiUrl(endpoint) {
        return `${CONFIG.API_BASE_URL}${endpoint}`;
    }

    static async init() {
        try {
            const res = await fetch(this.getApiUrl("/api/kb"));
            if (res.ok) {
                const kb = await res.json();
                localStorage.setItem("stainscan_kb", JSON.stringify(kb));
            }
        } catch (e) {
            console.warn("Could not fetch KB from cloud on init, using local storage cache:", e);
        }
    }

    static getKB() {
        return JSON.parse(localStorage.getItem("stainscan_kb")) || defaultKnowledgeBase;
    }

    static async saveKB(kb) {
        localStorage.setItem("stainscan_kb", JSON.stringify(kb));
        try {
            await fetch(this.getApiUrl("/api/kb"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(kb)
            });
        } catch (e) {
            console.error("Failed to save KB to cloud:", e);
        }
    }

    static async getHistoryCloud(email) {
        try {
            const res = await fetch(this.getApiUrl(`/api/scans?email=${encodeURIComponent(email)}`));
            if (res.ok) {
                const scans = await res.json();
                localStorage.setItem(`stainscan_history_${email}`, JSON.stringify(scans));
                return scans;
            }
        } catch (e) {
            console.warn("Failed to get scans from cloud, using offline cache:", e);
        }
        return JSON.parse(localStorage.getItem(`stainscan_history_${email}`)) || [];
    }

    static async saveHistoryCloud(scanItem) {
        const email = scanItem.email;
        const localHist = JSON.parse(localStorage.getItem(`stainscan_history_${email}`)) || [];
        localHist.unshift(scanItem);
        localStorage.setItem(`stainscan_history_${email}`, JSON.stringify(localHist.slice(0, 10)));

        try {
            await fetch(this.getApiUrl("/api/scans"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(scanItem)
            });
        } catch (e) {
            console.error("Failed to upload scan history to cloud:", e);
        }
    }

    static async addLog(type, text) {
        console.log(`[LOG - ${type}] ${text}`);
        try {
            await fetch(this.getApiUrl("/api/logs"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ type, text })
            });
        } catch (e) {
            console.warn("Could not post log to cloud:", e);
        }
    }

    static getUsers() {
        return JSON.parse(localStorage.getItem("stainscan_users")) || [];
    }

    static saveUsers(users) {
        localStorage.setItem("stainscan_users", JSON.stringify(users));
    }

    static getHistory(email) {
        const user = email || (JSON.parse(sessionStorage.getItem("stainscan_current_user")) || {}).email;
        if (!user) return [];
        return JSON.parse(localStorage.getItem(`stainscan_history_${user}`)) || [];
    }

    static saveHistory(hist, email) {
        const user = email || (JSON.parse(sessionStorage.getItem("stainscan_current_user")) || {}).email;
        if (!user) return;
        localStorage.setItem(`stainscan_history_${user}`, JSON.stringify(hist));
    }

    static async updateScanStatusCloud(scanId, status) {
        try {
            const res = await fetch(this.getApiUrl(`/api/scans/${scanId}`), {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status })
            });
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.error(`Failed to update scan status for ${scanId} in cloud:`, e);
        }
        return null;
    }
}

// --- Application Core Control ---
document.addEventListener("DOMContentLoaded", () => {
    DataStore.init();
    updateClock();
    setInterval(updateClock, 60000);

    // Current State variables
    let currentUser = JSON.parse(sessionStorage.getItem("stainscan_current_user")) || null;
    let selectedImageBase64 = null;
    let scanResultData = null;
    let activeGuideScanId = null;

    // --- DOM Elements Cache ---
    const appHeader = document.getElementById("appHeader");
    const appNavBar = document.getElementById("appNavBar");
    const headerAvatar = document.getElementById("headerAvatar");
    const dashboardUserName = document.getElementById("dashboardUserName");
    const headerProfileBtn = document.getElementById("headerProfileBtn");

    // Screens
    const screenAuth = document.getElementById("screen-auth");
    const screenDashboard = document.getElementById("screen-dashboard");
    const screenScan = document.getElementById("screen-scan");
    const screenHistory = document.getElementById("screen-history");
    const screenProfile = document.getElementById("screen-profile");
    const screens = [screenAuth, screenDashboard, screenScan, screenHistory, screenProfile];

    // Auth cards
    const authLoginCard = document.getElementById("auth-login-card");
    const authRegisterCard = document.getElementById("auth-register-card");
    const authRecoveryCard = document.getElementById("auth-recovery-card");
    const authCards = [authLoginCard, authRegisterCard, authRecoveryCard];

    // --- Toast Notifications ---
    function showToast(message, type = "success") {
        const container = document.getElementById("toastContainer");
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;

        let icon = "fa-circle-check";
        if (type === "error") icon = "fa-circle-exclamation";
        if (type === "warning") icon = "fa-triangle-exclamation";

        toast.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        `;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(-10px)";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // --- Clock Status bar ---
    function updateClock() {
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        hours = hours < 10 ? '0' + hours : hours;
        minutes = minutes < 10 ? '0' + minutes : minutes;
        const timeEl = document.getElementById("statusTime");
        if (timeEl) timeEl.textContent = `${hours}:${minutes}`;
    }

    // --- Navigation / Routing Router ---
    function navigateTo(screenId) {
        // Enforce Authentication
        if (!currentUser && screenId !== "screen-auth") {
            navigateTo("screen-auth");
            return;
        }

        // Hide/Show Header and Navigation Bar based on screen
        if (appHeader) appHeader.style.display = "none";
        if (screenId === "screen-auth") {
            appNavBar.style.display = "none";
        } else {
            appNavBar.style.display = "flex";
        }

        screens.forEach(screen => {
            if (screen.id === screenId) {
                screen.classList.add("active");
            } else {
                screen.classList.remove("active");
            }
        });

        // Update Bottom Nav active state
        const navItems = document.querySelectorAll(".nav-item");
        navItems.forEach(item => {
            if (item.getAttribute("data-screen") === screenId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        // If landing on specific screens, reload lists/content
        if (screenId === "screen-dashboard") {
            DataStore.getHistoryCloud(currentUser.email).then(scans => {
                renderDashboard(scans);
            });
        } else if (screenId === "screen-history") {
            DataStore.getHistoryCloud(currentUser.email).then(scans => {
                renderHistory(scans);
            });
        } else if (screenId === "screen-profile") {
            renderProfile();
        }
    }

    // Bind Navigation bottom buttons
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => {
            const targetScreen = item.getAttribute("data-screen");
            navigateTo(targetScreen);
        });
    });

    if (headerProfileBtn) {
        headerProfileBtn.addEventListener("click", () => navigateTo("screen-profile"));
    }

    // --- AUTHENTICATION MODULE ---
    // Toggle card tabs
    document.getElementById("btnGoRegister").addEventListener("click", () => {
        authCards.forEach(c => c.classList.remove("active"));
        authRegisterCard.classList.add("active");
        document.getElementById("auth-subtitle").textContent = "Register a new Account";
    });

    document.getElementById("btnGoLogin").addEventListener("click", () => {
        authCards.forEach(c => c.classList.remove("active"));
        authLoginCard.classList.add("active");
        document.getElementById("auth-subtitle").textContent = "AI-Powered Cotton Fabric Stain Care";
    });

    document.getElementById("btnGoForgot").addEventListener("click", () => {
        authCards.forEach(c => c.classList.remove("active"));
        authRecoveryCard.classList.add("active");
        document.getElementById("auth-subtitle").textContent = "Recover Account Password";
    });

    document.getElementById("btnBackToLogin").addEventListener("click", () => {
        authCards.forEach(c => c.classList.remove("active"));
        authLoginCard.classList.add("active");
        document.getElementById("auth-subtitle").textContent = "AI-Powered Cotton Fabric Stain Care";
    });

    // Login Form Submit
    document.getElementById("loginForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const email = document.getElementById("login-email").value.trim().toLowerCase();
        const password = document.getElementById("login-password").value;

        fetch(DataStore.getApiUrl("/api/login"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        })
        .then(res => {
            const contentType = res.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error("Unable to connect to the cloud service. Please verify that your backend server has been successfully redeployed on Render.");
            }
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || "Login failed"); });
            }
            return res.json();
        })
        .then(matchedUser => {
            currentUser = matchedUser;
            sessionStorage.setItem("stainscan_current_user", JSON.stringify(matchedUser));
            if (headerAvatar) {
                headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${matchedUser.avatar || matchedUser.name}`;
            }
            dashboardUserName.textContent = matchedUser.name;
            showToast(`Welcome back, ${matchedUser.name}!`);
            navigateTo("screen-dashboard");
            document.getElementById("loginForm").reset();
        })
        .catch(err => {
            showToast(err.message || "Invalid email credentials or password.", "error");
        });
    });

    // Registration Form Submit
    document.getElementById("registerForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const name = document.getElementById("reg-name").value.trim();
        const email = document.getElementById("reg-email").value.trim().toLowerCase();
        const password = document.getElementById("reg-password").value;
        const terms = document.getElementById("reg-terms").checked;

        if (password.length < 6) {
            showToast("Password must be at least 6 characters.", "error");
            return;
        }

        fetch(DataStore.getApiUrl("/api/register"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email, password })
        })
        .then(res => {
            const contentType = res.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error("Unable to connect to the cloud service. Please verify that your backend server has been successfully redeployed on Render.");
            }
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || "Registration failed"); });
            }
            return res.json();
        })
        .then(newUser => {
            showToast("Account created successfully! Please Sign In.");
            document.getElementById("registerForm").reset();
            document.getElementById("btnGoLogin").click();
        })
        .catch(err => {
            showToast(err.message || "Registration failed.", "error");
        });
    });

    // Password Recovery Submit
    document.getElementById("recoveryForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const email = document.getElementById("recover-email").value.trim().toLowerCase();
        const users = DataStore.getUsers();

        if (users.some(u => u.email === email)) {
            showToast("Password recovery link sent to your email!");
            DataStore.addLog("info", `Recovery code requested for: ${email}`);
            document.getElementById("recoveryForm").reset();
            document.getElementById("btnBackToLogin").click();
        } else {
            showToast("Email address not found.", "error");
        }
    });

    // --- USER DASHBOARD MODULE ---
    function renderDashboard(historyList) {
        const history = historyList || JSON.parse(localStorage.getItem(`stainscan_history_${currentUser.email}`)) || [];

        // Update stats
        document.getElementById("statTotalScans").textContent = history.length;
        const savedCount = history.filter(h => h.status === "Treated").length;
        document.getElementById("statSavedGarments").textContent = savedCount;

        // Render recent scans list
        const dashboardRecentList = document.getElementById("dashboardRecentList");
        dashboardRecentList.innerHTML = "";

        const recents = history.slice(0, 3); // show last 3

        if (recents.length === 0) {
            dashboardRecentList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>No scans performed yet. Try scanning a stain!</p>
                </div>
            `;
            return;
        }

        recents.forEach(item => {
            const itemEl = document.createElement("div");
            itemEl.className = "scan-item";

            let classType = "unknown";
            let icon = "fa-question";

            if (item.stain === "Mud") { classType = "mud"; icon = "fa-mound"; }
            else if (item.stain === "Used Cooking Oil") { classType = "oil"; icon = "fa-bottle-droplet"; }
            else if (item.stain === "Black Ballpen Ink") { classType = "ink"; icon = "fa-pen-clip"; }

            const formattedDate = new Date(item.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            const statusClass = item.status.toLowerCase();

            itemEl.innerHTML = `
                <div class="scan-item-left">
                    <div class="scan-item-badge ${classType}"><i class="fas ${icon}"></i></div>
                    <div class="scan-item-details">
                        <span class="scan-title">${item.stain}</span>
                        <span class="scan-meta">${formattedDate} • ${item.fabric}</span>
                    </div>
                </div>
                <div class="scan-item-right">
                    <span class="scan-conf">${item.confidence}% AI</span>
                    <span class="scan-status-tag ${statusClass}">${item.status}</span>
                </div>
            `;

            itemEl.addEventListener("click", () => {
                openCleaningGuide(item);
            });

            dashboardRecentList.appendChild(itemEl);
        });
    }

    document.getElementById("btnQuickScan").addEventListener("click", () => navigateTo("screen-scan"));
    document.getElementById("btnViewAllHistory").addEventListener("click", () => navigateTo("screen-history"));

    // --- INTELLIGENT SCANNER & CNN SIMULATOR MODULE ---
    const dropArea = document.getElementById("dropArea");
    const fileInput = document.getElementById("fileInput");
    const uploadEmptyState = document.getElementById("uploadEmptyState");
    const uploadPreviewState = document.getElementById("uploadPreviewState");
    const uploadedImagePreview = document.getElementById("uploadedImagePreview");
    const btnClearPreview = document.getElementById("btnClearPreview");
    const scanControls = document.getElementById("scanControls");
    const scanOverlayLine = document.getElementById("scanOverlayLine");
    const scannerLoader = document.getElementById("scannerLoader");
    const resultsCard = document.getElementById("resultsCard");

    // File selection triggers
    dropArea.addEventListener("click", (e) => {
        if (e.target === fileInput) return; // Prevent double trigger from bubbling
        if (!selectedImageBase64 && e.target !== btnClearPreview && !btnClearPreview.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });

    // Drag and Drop files
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.style.borderColor = "var(--color-primary)";
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.style.borderColor = "rgba(255, 255, 255, 0.12)";
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.style.borderColor = "rgba(255, 255, 255, 0.12)";
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });

    function handleUploadedFile(file) {
        if (!file.type.startsWith("image/")) {
            showToast("Please upload an image file.", "error");
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement("canvas");
                const max_size = 300;
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > max_size) {
                        height *= max_size / width;
                        width = max_size;
                    }
                } else {
                    if (height > max_size) {
                        width *= max_size / height;
                        height = max_size;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, width, height);

                selectedImageBase64 = canvas.toDataURL("image/jpeg", 0.8);
                uploadedImagePreview.src = selectedImageBase64;
                uploadEmptyState.style.display = "none";
                uploadPreviewState.style.display = "block";
                scanControls.style.display = "block";
                resultsCard.style.display = "none";
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Clear Preview
    btnClearPreview.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedImageBase64 = null;
        fileInput.value = "";
        uploadPreviewState.style.display = "none";
        uploadEmptyState.style.display = "flex";
        scanControls.style.display = "none";
        resultsCard.style.display = "none";
        scanOverlayLine.style.display = "none";
    });

    // Analyze Stain action
    document.getElementById("btnAnalyzeStain").addEventListener("click", () => {
        if (!selectedImageBase64) return;

        // Hide controls, trigger laser scanning lines
        scanControls.style.display = "none";
        scanOverlayLine.style.display = "block";
        scannerLoader.style.display = "block";

        const loaderStatusText = document.getElementById("loaderStatusText");
        const loaderSubtext = document.getElementById("loaderSubtext");
        const progressBarFill = document.getElementById("progressBarFill");

        // Grab CNN settings from simulator
        const simStain = document.getElementById("sim-stain").value;
        const simFabric = document.getElementById("sim-fabric").value;
        const simConfidence = parseInt(document.getElementById("sim-confidence").value) || 85;
        const simError = document.getElementById("sim-error-trigger").value;

        // Contact live prediction server if "live" mode is active
        let livePredictionPromise = null;
        if (simStain === "live" && simError === "none") {
            const targetUrl = `${CONFIG.API_BASE_URL}/predict`;

            livePredictionPromise = fetch(targetUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ image: selectedImageBase64 })
            })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(errData => {
                            throw new Error(errData.error || `Server status ${res.status}`);
                        }).catch(() => {
                            throw new Error(`Server status ${res.status}`);
                        });
                    }
                    return res.json();
                });
        }

        let progress = 0;
        progressBarFill.style.width = "0%";

        const statusSteps = [
            { text: "Preprocessing Image...", sub: "Reducing background noise and adjusting contrast maps", limit: 25 },
            { text: "Detecting Material...", sub: "Verifying fabric weave orientation matches Cotton rules", limit: 50 },
            { text: "Running CNN Classifier...", sub: "Extracting convolution layers & stain features maps", limit: 75 },
            { text: "Validating Results...", sub: "Computing Softmax classification confidence scores", limit: 100 }
        ];

        let currentStepIndex = 0;
        let isPredictionFinished = false;
        let predictionResult = null;
        let predictionError = null;

        if (simStain === "live" && simError === "none") {
            if (livePredictionPromise) {
                livePredictionPromise.then(result => {
                    predictionResult = result;
                    isPredictionFinished = true;
                }).catch(err => {
                    predictionError = err;
                    isPredictionFinished = true;
                });
            } else {
                predictionError = new Error("Prediction server offline or host unreachable");
                isPredictionFinished = true;
            }
        } else {
            isPredictionFinished = true;
        }

        const interval = setInterval(() => {
            progress += 2;
            
            if (progress > 100) {
                if (!isPredictionFinished) {
                    progress = 0;
                    currentStepIndex = 0;
                } else {
                    clearInterval(interval);
                    scanOverlayLine.style.display = "none";
                    scannerLoader.style.display = "none";

                    if (simError !== "none") {
                        handleSimulationError(simError);
                    } else if (simStain === "live") {
                        if (predictionError) {
                            processScanningFailure(predictionError.message || "Connection refused");
                        } else if (predictionResult && !predictionResult.error) {
                            processScanningSuccess(predictionResult.stain, predictionResult.fabric, predictionResult.confidence);
                        } else {
                            const errMsg = (predictionResult && predictionResult.error) ? predictionResult.error : "Unknown backend error";
                            processScanningFailure(errMsg);
                        }
                    } else {
                        processScanningSuccess(simStain, simFabric, simConfidence);
                    }
                    return;
                }
            }

            progressBarFill.style.width = `${progress}%`;

            const activeStep = statusSteps[currentStepIndex];
            if (activeStep && progress >= activeStep.limit) {
                loaderStatusText.textContent = activeStep.text;
                loaderSubtext.textContent = activeStep.sub;
                currentStepIndex++;
            }
        }, 50);
    });

    function handleSimulationError(errorType) {
        let msg = "A system error occurred during analysis.";
        let code = "SYS_ERR_305";

        if (errorType === "low_light") {
            msg = "Low light conditions. Enhance lighting or flash and try scanning again.";
            code = "WARN_LOW_LIGHT";
        } else if (errorType === "blurry") {
            msg = "Blurry fabric details. Keep camera steady and capture sharp details.";
            code = "WARN_BLURRY_FRAME";
        } else if (errorType === "network_error") {
            msg = "Connection to CNN backend database timed out. Check network logs.";
            code = "ERR_NET_TIMEOUT";
        }

        showToast(msg, errorType === "network_error" ? "error" : "warning");
        DataStore.addLog(errorType === "network_error" ? "error" : "warning", `Scan failure [${code}]: ${msg}`);
        scanControls.style.display = "block";
    }

    function processScanningSuccess(stain, fabric, confidence) {
        const isCotton = fabric.includes("Cotton");
        const isStainSupported = stain !== "Unknown Stain";

        // Reset badge styles in case a previous scan failed
        const resultsBadge = document.querySelector(".results-card .results-badge");
        resultsBadge.innerHTML = `<i class="fas fa-check-circle"></i> Detection Successful`;
        resultsBadge.style.backgroundColor = "";
        resultsBadge.style.color = "";
        document.getElementById("btnViewCleaningGuide").style.display = "inline-flex";

        const resultStainType = document.getElementById("resultStainType");
        const resultFabricType = document.getElementById("resultFabricType");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultConfidenceBar = document.getElementById("resultConfidenceBar");
        const safetyBanner = document.getElementById("safetyBanner");

        // --- 70% Confidence Threshold Guardrail ---
        if (confidence < 70) {
            resultsBadge.innerHTML = `<i class="fas fa-circle-exclamation"></i> Stain Not Detected`;
            resultsBadge.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
            resultsBadge.style.color = "#ef4444";

            resultStainType.textContent = "Unrecognized Input";
            resultFabricType.textContent = "N/A";
            resultConfidence.textContent = `${confidence}%`;
            resultConfidenceBar.style.width = `${confidence}%`;

            safetyBanner.className = "safety-warning-banner";
            safetyBanner.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
            safetyBanner.style.borderColor = "rgba(239, 68, 68, 0.2)";
            safetyBanner.style.color = "#ef4444";
            safetyBanner.innerHTML = `<i class="fas fa-triangle-exclamation"></i> <span>Unrecognized Input - The system could not confidently identify a fabric stain. Please ensure the area is flat and well-lit, then try again.</span>`;
            
            document.getElementById("btnViewCleaningGuide").style.display = "none";
            
            DataStore.addLog("warning", `Low confidence scan rejected (${confidence}% confidence)`);
            resultsCard.style.display = "block";
            showToast("Unrecognized Input", "warning");
            return;
        }

        scanResultData = {
            id: "h_" + Math.random().toString(36).substr(2, 9),
            email: currentUser.email,
            stain: stain,
            fabric: fabric,
            confidence: confidence,
            timestamp: new Date().toISOString(),
            status: "Pending",
            image: ""
        };

        resultStainType.textContent = stain;
        resultFabricType.textContent = fabric;
        resultConfidence.textContent = `${confidence}%`;
        resultConfidenceBar.style.width = `${confidence}%`;

        if (!isCotton) {
            safetyBanner.className = "safety-warning-banner";
            safetyBanner.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
            safetyBanner.style.borderColor = "rgba(239, 68, 68, 0.2)";
            safetyBanner.style.color = "#ef4444";
            safetyBanner.innerHTML = `<i class="fas fa-circle-exclamation"></i> <span>Fabric Mismatch: Treatment plans are calibrated for 100% Cotton only. Do not wash polyester/silk with this recipe.</span>`;
            document.getElementById("btnViewCleaningGuide").style.display = "none";
            DataStore.addLog("warning", `Unsupported fabric detected: ${fabric}`);
        } else if (!isStainSupported) {
            safetyBanner.className = "safety-warning-banner";
            safetyBanner.style.backgroundColor = "rgba(245, 158, 11, 0.08)";
            safetyBanner.style.borderColor = "rgba(245, 158, 11, 0.2)";
            safetyBanner.style.color = "#f59e0b";
            safetyBanner.innerHTML = `<i class="fas fa-triangle-exclamation"></i> <span>Unsupported Stain: The CNN model was unable to classify stain parameters. Ask an expert.</span>`;
            document.getElementById("btnViewCleaningGuide").style.display = "none";
            DataStore.addLog("warning", `Unknown stain classified.`);
        } else {
            safetyBanner.className = "safety-warning-banner";
            safetyBanner.style.backgroundColor = "rgba(16, 185, 129, 0.08)";
            safetyBanner.style.borderColor = "rgba(16, 185, 129, 0.2)";
            safetyBanner.style.color = "#10b981";
            safetyBanner.innerHTML = `<i class="fas fa-shield-heart"></i> <span>Fabric Safety: Verified safe to treat using water-based/neutral methods.</span>`;
            document.getElementById("btnViewCleaningGuide").style.display = "inline-flex";
        }

        DataStore.saveHistoryCloud(scanResultData);
        DataStore.addLog("info", `Successful CNN scan: ${stain} on ${fabric} (${confidence}% confidence)`);

        resultsCard.style.display = "block";
        showToast("Fabric Scan Complete!");
    }

    function processScanningFailure(errorMessage) {
        const resultStainType = document.getElementById("resultStainType");
        const resultFabricType = document.getElementById("resultFabricType");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultConfidenceBar = document.getElementById("resultConfidenceBar");
        const safetyBanner = document.getElementById("safetyBanner");
        const resultsBadge = document.querySelector(".results-card .results-badge");
        const btnViewCleaningGuide = document.getElementById("btnViewCleaningGuide");

        resultsBadge.innerHTML = `<i class="fas fa-circle-exclamation"></i> Analysis Failed`;
        resultsBadge.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
        resultsBadge.style.color = "#ef4444";

        resultStainType.innerHTML = `<span style="color:#ef4444; font-weight: normal;">Backend Error: ${errorMessage}</span>`;
        resultFabricType.textContent = "N/A";
        resultConfidence.textContent = "0%";
        resultConfidenceBar.style.width = "0%";

        safetyBanner.className = "safety-warning-banner";
        safetyBanner.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
        safetyBanner.style.borderColor = "rgba(239, 68, 68, 0.2)";
        safetyBanner.style.color = "#ef4444";
        safetyBanner.innerHTML = `<i class="fas fa-triangle-exclamation"></i> <span>Connection or processing error. Please check server status.</span>`;

        btnViewCleaningGuide.style.display = "none";

        resultsCard.style.display = "block";
    }

    document.getElementById("btnRestartScan").addEventListener("click", () => {
        btnClearPreview.click();
    });

    document.getElementById("btnViewCleaningGuide").addEventListener("click", () => {
        if (scanResultData) {
            openCleaningGuide(scanResultData);
        }
    });

    // --- SCAN HISTORY SCREEN MODULE ---
    const historyListContainer = document.getElementById("historyListContainer");
    const historySearch = document.getElementById("historySearch");
    let activeFilter = "all";

    function renderHistory(historyList) {
        const history = historyList || JSON.parse(localStorage.getItem(`stainscan_history_${currentUser.email}`)) || [];
        const searchQuery = historySearch.value.trim().toLowerCase();

        historyListContainer.innerHTML = "";

        const filtered = history.filter(item => {
            const matchesSearch = item.stain.toLowerCase().includes(searchQuery) || item.fabric.toLowerCase().includes(searchQuery);
            const matchesFilter = activeFilter === "all" || item.stain === activeFilter;
            return matchesSearch && matchesFilter;
        });

        if (filtered.length === 0) {
            historyListContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-magnifying-glass"></i>
                    <p>No scans found matching details.</p>
                </div>
            `;
            return;
        }

        filtered.forEach(item => {
            const itemEl = document.createElement("div");
            itemEl.className = "scan-item";

            let classType = "unknown";
            let icon = "fa-question";

            if (item.stain === "Mud") { classType = "mud"; icon = "fa-mound"; }
            else if (item.stain === "Used Cooking Oil") { classType = "oil"; icon = "fa-bottle-droplet"; }
            else if (item.stain === "Black Ballpen Ink") { classType = "ink"; icon = "fa-pen-clip"; }

            const formattedDate = new Date(item.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            const statusClass = item.status.toLowerCase();

            itemEl.innerHTML = `
                <div class="scan-item-left">
                    <div class="scan-item-badge ${classType}"><i class="fas ${icon}"></i></div>
                    <div class="scan-item-details">
                        <span class="scan-title">${item.stain}</span>
                        <span class="scan-meta">${formattedDate} • ${item.fabric}</span>
                    </div>
                </div>
                <div class="scan-item-right">
                    <span class="scan-conf">${item.confidence}% AI</span>
                    <span class="scan-status-tag ${statusClass}">${item.status}</span>
                </div>
            `;

            itemEl.addEventListener("click", () => {
                openCleaningGuide(item);
            });

            historyListContainer.appendChild(itemEl);
        });
    }

    historySearch.addEventListener("input", renderHistory);

    document.querySelectorAll(".filter-chips .chip").forEach(chip => {
        chip.addEventListener("click", () => {
            document.querySelectorAll(".filter-chips .chip").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            activeFilter = chip.getAttribute("data-filter");
            renderHistory();
        });
    });

    // --- INTERACTIVE RECOMMENDER & CHECKLIST MODAL ---
    const guideModal = document.getElementById("guideModal");
    const modalStainName = document.getElementById("modalStainName");
    const modalMaterialsList = document.getElementById("modalMaterialsList");
    const modalStepsList = document.getElementById("modalStepsList");
    const modalProgressPercent = document.getElementById("modalProgressPercent");
    const modalProgressBarFill = document.getElementById("modalProgressBarFill");
    const btnCompleteTreatment = document.getElementById("btnCompleteTreatment");

    function openCleaningGuide(historyItem) {
        activeGuideScanId = historyItem.id;
        const kb = DataStore.getKB();
        const recipe = kb[historyItem.stain];

        if (!recipe) {
            showToast("No active cleaning guide registered for this stain.", "error");
            return;
        }

        modalStainName.textContent = `${historyItem.stain} Stain Care`;

        modalMaterialsList.innerHTML = "";
        recipe.materials.forEach(mat => {
            const chip = document.createElement("li");
            chip.textContent = mat;
            modalMaterialsList.appendChild(chip);
        });

        modalStepsList.innerHTML = "";
        recipe.steps.forEach((stepText, idx) => {
            const stepEl = document.createElement("div");
            stepEl.className = "step-checklist-item";
            stepEl.innerHTML = `
                <div class="step-checkbox"><i class="fas fa-check"></i></div>
                <div class="step-text-wrap">
                    <span class="step-num-title">Step ${idx + 1}</span>
                    <span class="step-instr">${stepText}</span>
                </div>
            `;

            stepEl.addEventListener("click", () => {
                stepEl.classList.toggle("checked");
                updateChecklistProgress();
            });

            modalStepsList.appendChild(stepEl);
        });

        if (historyItem.status === "Treated") {
            document.querySelectorAll(".step-checklist-item").forEach(item => item.classList.add("checked"));
            btnCompleteTreatment.disabled = true;
            btnCompleteTreatment.innerHTML = `<i class="fas fa-check-double"></i> <span>Completed & Saved</span>`;
        } else {
            btnCompleteTreatment.disabled = false;
            btnCompleteTreatment.innerHTML = `<i class="fas fa-circle-check"></i> <span>Mark Treatment Completed</span>`;
        }

        updateChecklistProgress();
        guideModal.style.display = "flex";
    }

    function updateChecklistProgress() {
        const total = document.querySelectorAll(".step-checklist-item").length;
        const checked = document.querySelectorAll(".step-checklist-item.checked").length;
        const percent = total > 0 ? Math.round((checked / total) * 100) : 0;

        modalProgressPercent.textContent = `${percent}%`;
        modalProgressBarFill.style.width = `${percent}%`;

        if (percent === 100) {
            btnCompleteTreatment.classList.add("active");
        } else {
            btnCompleteTreatment.classList.remove("active");
        }
    }

    function closeGuideModal() {
        guideModal.style.display = "none";
        activeGuideScanId = null;
        renderDashboard();
        renderHistory();
    }

    document.getElementById("btnCloseModal").addEventListener("click", closeGuideModal);
    document.getElementById("btnCancelModal").addEventListener("click", closeGuideModal);

    btnCompleteTreatment.addEventListener("click", async () => {
        console.log("Complete Treatment button clicked.");
        if (!activeGuideScanId) {
            console.error("Click handler triggered but no activeGuideScanId set.");
            showToast("Failed to complete treatment: active guide scan ID not found.", "error");
            return;
        }

        try {
            // 1. Update local storage cache first for instant feedback
            const history = DataStore.getHistory();
            const itemIdx = history.findIndex(h => (h.id === activeGuideScanId || h._id === activeGuideScanId));

            if (itemIdx !== -1) {
                history[itemIdx].status = "Treated";
                DataStore.saveHistory(history);
                console.log(`Local storage updated: marked scan ID ${activeGuideScanId} as Treated.`);
            } else {
                console.warn(`Scan ID ${activeGuideScanId} not found in local history cache.`);
            }

            // 2. Update cloud database via API
            try {
                const response = await DataStore.updateScanStatusCloud(activeGuideScanId, "Treated");
                if (response && response.success) {
                    console.log(`Cloud database updated successfully for scan ID: ${activeGuideScanId}`);
                } else {
                    console.warn(`Cloud update returned non-success for scan ID: ${activeGuideScanId}`, response);
                }
            } catch (apiError) {
                console.warn("Could not sync treatment status to API (running local fallback):", apiError);
            }

            // Log activity to user actions list
            DataStore.addLog("info", `Treatment marked complete for scan ID: ${activeGuideScanId}`);

            // 3. Update the UI state
            showToast("Stain treated and logged successfully!");
            
            // Redirect / close modal
            closeGuideModal();

        } catch (err) {
            console.error("Error executing Complete Treatment click handler:", err);
            showToast("An unexpected error occurred while completing the treatment.", "error");
        }
    });

    // --- USER PROFILE & SETTINGS MODULE ---
    const editProfileModal = document.getElementById("editProfileModal");
    const changePasswordModal = document.getElementById("changePasswordModal");

    function renderProfile() {
        document.getElementById("profileAvatar").src = `https://api.dicebear.com/7.x/bottts/svg?seed=${currentUser.avatar || currentUser.name}`;
        document.getElementById("profileName").textContent = currentUser.name;
        document.getElementById("profileEmail").textContent = currentUser.email;
    }

    document.getElementById("btnChangeAvatar").addEventListener("click", () => {
        const seed = prompt("Enter a word to generate a new AI avatar seed:", currentUser.avatar || currentUser.name);
        if (seed) {
            const users = DataStore.getUsers();
            const index = users.findIndex(u => u.email === currentUser.email);
            if (index !== -1) {
                users[index].avatar = seed;
                DataStore.saveUsers(users);
                currentUser.avatar = seed;
                sessionStorage.setItem("stainscan_current_user", JSON.stringify(currentUser));
                renderProfile();
                if (headerAvatar) {
                    headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${seed}`;
                }
                showToast("Avatar seed updated!");
                DataStore.addLog("info", `Avatar changed for: ${currentUser.email}`);
            }
        }
    });

    document.getElementById("btnLogout").addEventListener("click", () => {
        sessionStorage.removeItem("stainscan_current_user");
        currentUser = null;
        showToast("Signed out successfully.");
        navigateTo("screen-auth");
    });

    document.getElementById("btnEditProfile").addEventListener("click", () => {
        document.getElementById("edit-profile-name").value = currentUser.name;
        document.getElementById("edit-profile-email").value = currentUser.email;
        editProfileModal.style.display = "flex";
    });

    document.getElementById("btnCloseEditProfileModal").addEventListener("click", () => editProfileModal.style.display = "none");
    document.getElementById("btnCancelEditProfile").addEventListener("click", () => editProfileModal.style.display = "none");

    document.getElementById("editProfileForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const newName = document.getElementById("edit-profile-name").value.trim();
        const newEmail = document.getElementById("edit-profile-email").value.trim().toLowerCase();

        const users = DataStore.getUsers();
        const emailExists = users.find(u => u.email === newEmail && u.email !== currentUser.email);
        if (emailExists) {
            showToast("Email address is already in use.", "error");
            return;
        }

        const idx = users.findIndex(u => u.email === currentUser.email);
        if (idx !== -1) {
            users[idx].name = newName;
            users[idx].email = newEmail;
            DataStore.saveUsers(users);

            const oldEmail = currentUser.email;
            const history = DataStore.getHistory(oldEmail);
            history.forEach(item => {
                if (item.email === oldEmail) {
                    item.email = newEmail;
                }
            });
            DataStore.saveHistory(history, newEmail);
            if (oldEmail !== newEmail) {
                localStorage.removeItem(`stainscan_history_${oldEmail}`);
            }

            currentUser.name = newName;
            currentUser.email = newEmail;
            sessionStorage.setItem("stainscan_current_user", JSON.stringify(currentUser));

            renderProfile();
            dashboardUserName.textContent = newName;

            editProfileModal.style.display = "none";
            showToast("Personal details updated!");
            DataStore.addLog("info", `Profile updated details for: ${newEmail}`);
        }
    });

    document.getElementById("btnChangePassword").addEventListener("click", () => {
        changePasswordModal.style.display = "flex";
    });

    document.getElementById("btnCloseChangePasswordModal").addEventListener("click", () => changePasswordModal.style.display = "none");
    document.getElementById("btnCancelChangePassword").addEventListener("click", () => changePasswordModal.style.display = "none");

    document.getElementById("changePasswordForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const oldPass = document.getElementById("old-password").value;
        const newPass = document.getElementById("new-password").value;

        if (oldPass !== currentUser.password) {
            showToast("Current password matches incorrectly.", "error");
            return;
        }

        if (newPass.length < 6) {
            showToast("New password must be at least 6 characters.", "error");
            return;
        }

        const users = DataStore.getUsers();
        const idx = users.findIndex(u => u.email === currentUser.email);
        if (idx !== -1) {
            users[idx].password = newPass;
            DataStore.saveUsers(users);

            currentUser.password = newPass;
            sessionStorage.setItem("stainscan_current_user", JSON.stringify(currentUser));

            changePasswordModal.style.display = "none";
            document.getElementById("changePasswordForm").reset();
            showToast("Password updated successfully!");
            DataStore.addLog("info", `Password change for: ${currentUser.email}`);
        }
    });

    // --- SIMULATOR TABS CONTROLLER ACCENT ---
    const btnToggleSim = document.getElementById("btnToggleSim");
    const simulationPanel = document.getElementById("simulationPanel");

    btnToggleSim.addEventListener("click", () => {
        simulationPanel.classList.toggle("minimized");
        if (simulationPanel.classList.contains("minimized")) {
            btnToggleSim.innerHTML = `<i class="fas fa-plus"></i>`;
        } else {
            btnToggleSim.innerHTML = `<i class="fas fa-minus"></i>`;
        }
    });

    // Clear browser storage reset button
    document.getElementById("btnClearStorage").addEventListener("click", () => {
        if (confirm("Are you sure you want to clear browser storage? This will clear all scan history logs and reset the database.")) {
            localStorage.clear();
            sessionStorage.clear();
            showToast("Storage reset! Reloading page...", "warning");
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        }
    });

    // Initialize View on Page Load
    if (currentUser) {
        if (headerAvatar) {
            headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${currentUser.avatar || currentUser.name}`;
        }
        dashboardUserName.textContent = currentUser.name;
        navigateTo("screen-dashboard");
    } else {
        navigateTo("screen-auth");
    }
});
