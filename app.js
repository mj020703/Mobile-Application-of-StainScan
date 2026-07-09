/* ==========================================================================
   STAINSCAN MOBILE APP CLIENT-SIDE CONTROLLER
   Simulates Convolutional Neural Network (CNN) feature extraction models,
   runs interactive cleaning checklists for users.
   ========================================================================== */

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

// --- Mock Datastore Initializer ---
class DataStore {
    static init() {
        let currentKB = null;
        try {
            currentKB = JSON.parse(localStorage.getItem("stainscan_kb"));
        } catch (e) {
            currentKB = null;
        }

        if (!currentKB || typeof currentKB !== "object" || !currentKB["Used Cooking Oil"] || !currentKB["Black Ballpen Ink"] || !currentKB["Mud"]) {
            localStorage.setItem("stainscan_kb", JSON.stringify(defaultKnowledgeBase));
        }
        if (!localStorage.getItem("stainscan_users")) {
            const defaultUsers = [
                { name: "Admin Manager", email: "admin@stainscan.com", password: "admin123", role: "admin", avatar: "admin" },
                { name: "John Doe", email: "user@stainscan.com", password: "user123", role: "user", avatar: "John" }
            ];
            localStorage.setItem("stainscan_users", JSON.stringify(defaultUsers));
        }
        if (!localStorage.getItem("stainscan_history")) {
            const defaultHistory = [
                {
                    id: "h_1",
                    email: "user@stainscan.com",
                    stain: "Mud",
                    fabric: "Cotton Fabric (100%)",
                    confidence: 94,
                    timestamp: new Date(Date.now() - 3600000 * 24).toISOString(), // 1 day ago
                    status: "Treated",
                    image: "https://images.unsplash.com/photo-1595079676339-1534801ad6cf?w=400&q=80"
                },
                {
                    id: "h_2",
                    email: "user@stainscan.com",
                    stain: "Used Cooking Oil",
                    fabric: "Cotton Fabric (100%)",
                    confidence: 85,
                    timestamp: new Date(Date.now() - 3600000 * 4).toISOString(), // 4 hours ago
                    status: "Pending",
                    image: "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?w=400&q=80"
                }
            ];
            localStorage.setItem("stainscan_history", JSON.stringify(defaultHistory));
        }
        if (!localStorage.getItem("stainscan_logs")) {
            const defaultLogs = [
                { time: new Date().toISOString(), type: "info", text: "Database initialized successfully." },
                { time: new Date().toISOString(), type: "info", text: "CNN Stain classification models loaded (Ver 2.5)." },
                { time: new Date().toISOString(), type: "info", text: "Fabric classification safety models loaded (Ver 1.1)." }
            ];
            localStorage.setItem("stainscan_logs", JSON.stringify(defaultLogs));
        }
    }

    static getKB() { return JSON.parse(localStorage.getItem("stainscan_kb")); }
    static saveKB(kb) {
        try {
            localStorage.setItem("stainscan_kb", JSON.stringify(kb));
        } catch (e) {
            console.error("Failed to save KB to localStorage:", e);
        }
    }
    static getUsers() { return JSON.parse(localStorage.getItem("stainscan_users")); }
    static saveUsers(users) {
        try {
            localStorage.setItem("stainscan_users", JSON.stringify(users));
        } catch (e) {
            console.error("Failed to save users to localStorage:", e);
        }
    }
    static getHistory() { return JSON.parse(localStorage.getItem("stainscan_history")); }
    static saveHistory(hist) {
        try {
            localStorage.setItem("stainscan_history", JSON.stringify(hist));
        } catch (e) {
            console.error("Failed to save history: storage quota exceeded. Clearing older history items...", e);
            if (e.name === "QuotaExceededError" || e.name === "NS_ERROR_DOM_QUOTA_REACHED") {
                // Keep only the 5 most recent items
                if (hist.length > 5) {
                    const truncated = hist.slice(0, 5);
                    try {
                        localStorage.setItem("stainscan_history", JSON.stringify(truncated));
                        this.addLog("warning", "Storage quota exceeded. Older history items cleared.");
                    } catch (err) {
                        console.error("Storage still exceeded after truncation:", err);
                    }
                }
            }
        }
    }
    static getLogs() { return JSON.parse(localStorage.getItem("stainscan_logs")); }
    static saveLogs(logs) {
        try {
            localStorage.setItem("stainscan_logs", JSON.stringify(logs));
        } catch (e) {
            console.error("Failed to save logs to localStorage:", e);
        }
    }

    static addLog(type, text) {
        const logs = this.getLogs();
        logs.unshift({ time: new Date().toISOString(), type, text });
        this.saveLogs(logs.slice(0, 100)); // limit to last 100 logs
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
        if (screenId === "screen-auth") {
            appHeader.style.display = "none";
            appNavBar.style.display = "none";
        } else {
            appHeader.style.display = "flex";
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
            renderDashboard();
        } else if (screenId === "screen-history") {
            renderHistory();
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

    headerProfileBtn.addEventListener("click", () => navigateTo("screen-profile"));

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

        const users = DataStore.getUsers();
        const matchedUser = users.find(u => u.email === email && u.password === password);

        if (matchedUser) {
            currentUser = matchedUser;
            sessionStorage.setItem("stainscan_current_user", JSON.stringify(matchedUser));
            headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${matchedUser.avatar || matchedUser.name}`;
            dashboardUserName.textContent = matchedUser.name;
            showToast(`Welcome back, ${matchedUser.name}!`);
            DataStore.addLog("info", `User login successful: ${matchedUser.email}`);
            navigateTo("screen-dashboard");

            // clear form
            document.getElementById("loginForm").reset();
        } else {
            showToast("Invalid email credentials or password.", "error");
            DataStore.addLog("warning", `Failed login attempt for: ${email}`);
        }
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

        const users = DataStore.getUsers();
        if (users.find(u => u.email === email)) {
            showToast("Email address already registered.", "error");
            return;
        }

        // Add user
        const newUser = {
            name,
            email,
            password,
            role: "user",
            avatar: name
        };
        users.push(newUser);
        DataStore.saveUsers(users);
        DataStore.addLog("info", `New user registered: ${email}`);
        showToast("Account created successfully! Please Sign In.");

        // Go back to login
        document.getElementById("registerForm").reset();
        document.getElementById("btnGoLogin").click();
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
    function renderDashboard() {
        const history = DataStore.getHistory().filter(h => h.email === currentUser.email);

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
            selectedImageBase64 = event.target.result;
            uploadedImagePreview.src = selectedImageBase64;
            uploadEmptyState.style.display = "none";
            uploadPreviewState.style.display = "block";
            scanControls.style.display = "block";
            resultsCard.style.display = "none";
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

        // Contact live local backend server if "live" mode is active
        let livePredictionPromise = null;
        if (simStain === "live" && simError === "none") {
            const backendHost = !window.location.hostname || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
                ? "localhost"
                : window.location.hostname;
            livePredictionPromise = fetch(`http://${backendHost}:5000/predict`, {
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

        const interval = setInterval(() => {
            progress += 2;
            progressBarFill.style.width = `${progress}%`;

            const activeStep = statusSteps[currentStepIndex];
            if (progress >= activeStep.limit) {
                loaderStatusText.textContent = activeStep.text;
                loaderSubtext.textContent = activeStep.sub;
                currentStepIndex++;
            }

            if (progress >= 100) {
                clearInterval(interval);
                scanOverlayLine.style.display = "none";
                scannerLoader.style.display = "none";

                if (simError !== "none") {
                    handleSimulationError(simError);
                } else if (simStain === "live") {
                    if (livePredictionPromise) {
                        livePredictionPromise.then(result => {
                            if (result && !result.error) {
                                // Successful live classification
                                processScanningSuccess(result.stain, result.fabric, result.confidence);
                            } else {
                                // Backend error payload
                                const errMsg = (result && result.error) ? result.error : "Unknown backend error";
                                processScanningFailure(errMsg);
                            }
                        })
                            .catch(err => {
                                // Backend fetch error/connection failure
                                processScanningFailure(err.message || "Connection refused");
                            });
                    } else {
                        // Backend server offline
                        processScanningFailure("Prediction server offline or host unreachable");
                    }
                } else {
                    processScanningSuccess(simStain, simFabric, simConfidence);
                }
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

        const resultStainType = document.getElementById("resultStainType");
        const resultFabricType = document.getElementById("resultFabricType");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultConfidenceBar = document.getElementById("resultConfidenceBar");
        const safetyBanner = document.getElementById("safetyBanner");

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

        const history = DataStore.getHistory();
        history.unshift(scanResultData);
        DataStore.saveHistory(history);
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

    function renderHistory() {
        const history = DataStore.getHistory().filter(h => h.email === currentUser.email);
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

    btnCompleteTreatment.addEventListener("click", () => {
        if (!activeGuideScanId) return;

        const history = DataStore.getHistory();
        const itemIdx = history.findIndex(h => h.id === activeGuideScanId);

        if (itemIdx !== -1) {
            history[itemIdx].status = "Treated";
            DataStore.saveHistory(history);
            DataStore.addLog("info", `Treatment marked complete for scan ID: ${activeGuideScanId}`);
            showToast("Stain treated and logged successfully!");
            closeGuideModal();
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
                headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${seed}`;
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

            const history = DataStore.getHistory();
            history.forEach(item => {
                if (item.email === currentUser.email) {
                    item.email = newEmail;
                }
            });
            DataStore.saveHistory(history);

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
        headerAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${currentUser.avatar || currentUser.name}`;
        dashboardUserName.textContent = currentUser.name;
        navigateTo("screen-dashboard");
    } else {
        navigateTo("screen-auth");
    }
});
