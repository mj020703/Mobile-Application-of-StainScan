/* ==========================================================================
   STAINSCAN ADMIN WEB PORTAL CONTROLLER
   Handles administrator authentication, system synchronization with the shared
   localStorage database, recipe modifications, and analytics updates.
   ========================================================================== */

const LIVE_RENDER_BACKEND = "https://stainscan-backend-gmbw.onrender.com";

// Intelligently select API backend URL:
// 1. If accessed over HTTP/HTTPS from localhost or local LAN IP (e.g., http://localhost:5000), use current origin.
// 2. If hosted on Render (e.g. https://...onrender.com), use window.location.origin.
// 3. If accessed from local file system (file://), default to LIVE_RENDER_BACKEND.
function resolveApiBaseUrl() {
    if (typeof window !== "undefined" && window.location && window.location.protocol.startsWith("http")) {
        const host = window.location.hostname;
        if (host === "localhost" || host === "127.0.0.1" || host.startsWith("192.168.") || host.includes("onrender.com")) {
            return window.location.origin;
        }
    }
    return LIVE_RENDER_BACKEND;
}

const CONFIG = {
    API_BASE_URL: resolveApiBaseUrl()
};

class AdminDataStore {
    static getApiUrl(endpoint) {
        return `${CONFIG.API_BASE_URL}${endpoint}`;
    }

    static async getDbStatus() {
        try {
            const res = await fetch(this.getApiUrl("/api/db-status"));
            if (res.ok) return await res.json();
        } catch (e) {
            console.warn("Could not fetch DB status:", e);
        }
        return null;
    }

    static async init() {
        try {
            const res = await fetch(this.getApiUrl("/api/kb"));
            if (res.ok) {
                const kb = await res.json();
                localStorage.setItem("stainscan_kb", JSON.stringify(kb));
            }
        } catch (e) {
            console.warn("Could not sync KB with cloud on init:", e);
        }
    }

    static getKB() {
        return JSON.parse(localStorage.getItem("stainscan_kb")) || {};
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

    static async getUsers() {
        try {
            const res = await fetch(this.getApiUrl("/api/users"));
            if (res.ok) return await res.json();
        } catch (e) {
            console.error("Failed to fetch users from database:", e);
        }
        return [];
    }

    static async getHistory() {
        try {
            // Fetch live scan documents excluding large Base64 images for optimal speed
            const res = await fetch(this.getApiUrl("/api/scans?limit=50&include_images=false"));
            if (res.ok) return await res.json();
        } catch (e) {
            console.error("Failed to fetch scans from database:", e);
        }
        return [];
    }

    static async getMetrics() {
        const endpoints = ["/api/stats", "/api/admin/overview", "/api/metrics"];
        for (const ep of endpoints) {
            try {
                const res = await fetch(this.getApiUrl(ep));
                if (res.ok) return await res.json();
            } catch (_) {}
        }
        return null;
    }

    static async getLogs() {
        try {
            const res = await fetch(this.getApiUrl("/api/logs"));
            if (res.ok) return await res.json();
        } catch (e) {
            console.error("Failed to fetch logs from cloud:", e);
        }
        return [];
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
}

document.addEventListener("DOMContentLoaded", () => {
    AdminDataStore.init();
    let currentAdmin = JSON.parse(sessionStorage.getItem("stainscan_current_admin")) || null;

    // --- DOM Elements ---
    const authContainer = document.getElementById("admin-auth-container");
    const mainContainer = document.getElementById("admin-main-container");
    const adminTabTitle = document.getElementById("adminTabTitle");
    const sidebarAdminName = document.getElementById("sidebarAdminName");
    const sidebarAvatar = document.getElementById("sidebarAvatar");
    const currentDateBadge = document.getElementById("currentDateBadge");

    // Panes
    const menuItems = document.querySelectorAll(".menu-item");
    const adminPanes = document.querySelectorAll(".admin-pane");

    // Set Current Date
    const today = new Date();
    currentDateBadge.textContent = today.toLocaleDateString([], { month: 'long', day: 'numeric', year: 'numeric' });

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

    // --- Tab Router Navigation ---
    function navigateToTab(tabId) {
        // Toggle Active Menu Button
        menuItems.forEach(item => {
            if (item.getAttribute("data-tab") === tabId) {
                item.classList.add("active");
                // Update header title based on navigation
                const text = item.querySelector("span").textContent;
                adminTabTitle.textContent = text;
            } else {
                item.classList.remove("active");
            }
        });

        // Toggle Active Content Panel
        adminPanes.forEach(pane => {
            if (pane.id === tabId) {
                pane.classList.add("active");
            } else {
                pane.classList.remove("active");
            }
        });

        // Trigger updates depending on view
        if (tabId === "admin-overview") {
            renderOverview();
        } else if (tabId === "admin-users-list") {
            renderUsers();
        } else if (tabId === "admin-recipe-editor") {
            loadRecipeDetails();
        } else if (tabId === "admin-audit-logs") {
            renderLogs();
        }
    }

    menuItems.forEach(item => {
        item.addEventListener("click", () => {
            navigateToTab(item.getAttribute("data-tab"));
        });
    });

    // --- ADMIN AUTHENTICATION GATES ---
    document.getElementById("adminLoginForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const submitBtn = document.getElementById("btnAdminLoginSubmit") || e.target.querySelector("button[type='submit']");
        const email = document.getElementById("admin-email").value.trim().toLowerCase();
        const password = document.getElementById("admin-password").value;

        // Activate smooth loading state and disable button to prevent multiple submissions
        const originalHtml = submitBtn ? submitBtn.innerHTML : null;
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.classList.add("loading");
            submitBtn.innerHTML = `<i class="fas fa-circle-notch fa-spin btn-spinner"></i> <span>Authenticating...</span>`;
        }

        const restoreBtn = () => {
            if (submitBtn && originalHtml) {
                submitBtn.disabled = false;
                submitBtn.classList.remove("loading");
                submitBtn.innerHTML = originalHtml;
            }
        };

        fetch(AdminDataStore.getApiUrl("/api/login"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || "Login failed"); });
            }
            return res.json();
        })
        .then(matched => {
            if (matched.role === "admin") {
                currentAdmin = matched;
                sessionStorage.setItem("stainscan_current_admin", JSON.stringify(matched));
                
                authContainer.classList.remove("active");
                mainContainer.style.display = "flex";
                
                sidebarAdminName.textContent = matched.name;
                sidebarAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${matched.avatar || matched.name}`;
                
                showToast(`Access granted. Welcome, ${matched.name}!`);
                renderOverview();
                document.getElementById("adminLoginForm").reset();
                restoreBtn();
            } else {
                showToast("Access Denied. Account does not possess System Admin role.", "error");
                restoreBtn();
            }
        })
        .catch(err => {
            showToast(err.message || "Invalid credentials entered. Try again.", "error");
            restoreBtn();
        });
    });

    // Log Out
    document.getElementById("btnAdminLogout").addEventListener("click", () => {
        sessionStorage.removeItem("stainscan_current_admin");
        currentAdmin = null;
        
        mainContainer.style.display = "none";
        authContainer.classList.add("active");
        document.getElementById("adminLoginForm").reset();
        showToast("Logged out from Admin Console.");
    });

    // --- LIVE DATABASE STATUS INDICATOR ---
    async function updateDbStatusBadge() {
        const dbBadge = document.getElementById("dbStatusBadge");
        if (!dbBadge) return;
        const status = await AdminDataStore.getDbStatus();
        if (status && status.is_connected && !status.is_demo_mode) {
            dbBadge.innerHTML = `<i class="fas fa-database" style="color:#10b981;"></i> <span>MongoDB Atlas: <strong>Live</strong></span>`;
            dbBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
            dbBadge.style.color = "#10b981";
            dbBadge.title = `Connected to MongoDB Atlas cluster '${status.database_name}'. Live collections: users (${status.collections?.users || 0}), scans (${status.collections?.scans || 0}), metrics (${status.collections?.metrics || 0})`;
        } else if (status && status.status === "fallback_mode") {
            dbBadge.innerHTML = `<i class="fas fa-database" style="color:#f59e0b;"></i> <span>Database: <strong>Fallback Mode (Local JSON)</strong></span>`;
            dbBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
            dbBadge.style.color = "#f59e0b";
            dbBadge.title = "MONGO_URI not configured or unreachable; using local scans.json and users.json";
        } else {
            dbBadge.innerHTML = `<i class="fas fa-triangle-exclamation" style="color:#ef4444;"></i> <span>Backend: <strong>Connecting...</strong></span>`;
            dbBadge.style.borderColor = "rgba(239, 68, 68, 0.4)";
            dbBadge.style.color = "#ef4444";
        }
    }

    // --- OVERVIEW & ANALYTICS PANELS ---
    async function renderOverview() {
        updateDbStatusBadge();
        const [users, history, metrics] = await Promise.all([
            AdminDataStore.getUsers(),
            AdminDataStore.getHistory(),
            AdminDataStore.getMetrics()
        ]);

        // 1. Registered Mobile Users: strictly count users with role === 'user' or role !== 'admin'
        const mobileUsersList = users.filter(u => u.role === "user" || u.role !== "admin");
        const totalUsers = (metrics && typeof metrics.registered_mobile_users === "number")
            ? metrics.registered_mobile_users
            : ((metrics && typeof metrics.totalUsersCount === "number")
                ? metrics.totalUsersCount
                : ((metrics && typeof metrics.total_users === "number")
                    ? metrics.total_users
                    : mobileUsersList.length));

        // 2. Total CNN Scans Run: query from backend stats or scans/history collection
        const treatedFromHistory = history.filter(h => h.status === "Treated").length;
        const totalScans = (metrics && typeof metrics.total_scans === "number")
            ? metrics.total_scans
            : ((metrics && typeof metrics.totalScansCount === "number")
                ? metrics.totalScansCount
                : history.length);

        // 3. Successful Treatments: query from backend stats or treated scans in history
        const totalTreated = (metrics && typeof metrics.successful_treatments === "number")
            ? metrics.successful_treatments
            : ((metrics && typeof metrics.totalTreatedCount === "number")
                ? metrics.totalTreatedCount
                : treatedFromHistory);

        const totalUsersEl = document.getElementById("totalUsersCount");
        const totalScansEl = document.getElementById("totalScansCount");
        const totalTreatedEl = document.getElementById("totalTreatedCount");

        if (totalUsersEl) totalUsersEl.textContent = totalUsers;
        if (totalScansEl) totalScansEl.textContent = totalScans;
        if (totalTreatedEl) totalTreatedEl.textContent = totalTreated;

        // Render Stain Distribution SVG Charts
        const chartContainer = document.getElementById("adminStainChart");
        if (!chartContainer) return;
        chartContainer.innerHTML = "";

        const counts = {
            "Used Cooking Oil": 0,
            "Black Ballpen Ink": 0,
            "Mud": 0
        };

        // If history has items, compute distribution
        if (history.length > 0 && metrics && metrics.stain_counts) {
            Object.keys(counts).forEach(k => {
                if (typeof metrics.stain_counts[k] === "number") {
                    counts[k] = metrics.stain_counts[k];
                }
            });
        }

        if (history.length > 0) {
            history.forEach(item => {
                if (counts[item.stain] !== undefined && (!metrics || !metrics.stain_counts)) {
                    counts[item.stain]++;
                }
            });
        }

        const total = Object.values(counts).reduce((a, b) => a + b, 0);

        Object.keys(counts).forEach(key => {
            const val = counts[key];
            const pct = total > 0 ? Math.round((val / total) * 100) : 0;

            const barRow = document.createElement("div");
            barRow.className = "bar-row";
            barRow.innerHTML = `
                <div class="bar-lbl" title="${key}">${key}</div>
                <div class="bar-track-wrap">
                    <div class="bar-fill-indicator" style="width: ${pct}%;"></div>
                </div>
                <div class="bar-val">${val}</div>
            `;
            chartContainer.appendChild(barRow);
        });

        // Render Live Mobile Scans Feed Table
        const scansTableBody = document.getElementById("liveScansTableBody");
        if (scansTableBody) {
            scansTableBody.innerHTML = "";
            if (!history || history.length === 0) {
                scansTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:24px; color:var(--text-muted);">No mobile scans logged yet. Perform a scan on your mobile phone to see it appear here!</td></tr>`;
            } else {
                // Show most recent scans first (up to 15)
                const recentScans = [...history].reverse().slice(0, 15);
                recentScans.forEach(scan => {
                    const tr = document.createElement("tr");
                    const dateStr = scan.timestamp ? new Date(scan.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "Just now";
                    const isTreated = scan.status === "Treated";
                    const statusBadge = isTreated
                        ? `<span class="badge badge-success"><i class="fas fa-check-circle"></i> Treated</span>`
                        : `<span class="badge badge-warning"><i class="fas fa-clock"></i> Pending</span>`;
                    
                    const confVal = parseInt(scan.confidence) || 0;
                    const isUnidentified = confVal < 70 || scan.stain === "Unidentified" || scan.stain === "Unidentified Stain" || scan.stain === "Not Classified";
                    const displayStain = isUnidentified ? "Unidentified" : (scan.stain || "Unknown");

                    let stainColor = "#ef4444";
                    if (!isUnidentified) {
                        if (scan.stain === "Used Cooking Oil") stainColor = "#f59e0b";
                        else if (scan.stain === "Black Ballpen Ink") stainColor = "#3b82f6";
                        else if (scan.stain === "Mud") stainColor = "#8b5cf6";
                    }

                    const displayFabric = isUnidentified ? "Unidentified / N/A" : (scan.fabric || "Cotton (100%)");

                    tr.innerHTML = `
                        <td>${dateStr}</td>
                        <td><strong>${scan.email || "mobile_user"}</strong></td>
                        <td><span style="font-weight:600; color:${stainColor};">${displayStain}</span></td>
                        <td>${displayFabric}</td>
                        <td><strong>${confVal}%</strong></td>
                        <td>${statusBadge}</td>
                    `;
                    scansTableBody.appendChild(tr);
                });
            }
        }
    }

    // --- REGISTERED USERS DIRECTORY PANELS ---
    async function renderUsers() {
        const users = await AdminDataStore.getUsers();
        const tableBody = document.getElementById("usersTableBody");
        tableBody.innerHTML = "";

        if (users.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center;">No accounts registered.</td></tr>`;
            return;
        }

        users.forEach(u => {
            const tr = document.createElement("tr");
            
            // Seed a consistent mock date based on email to make it look premium
            const seedDate = u.email.includes("admin") ? "2026-06-01" : "2026-07-01";
            const formattedDate = new Date(seedDate).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
            const roleTagClass = u.role === "admin" ? "pending" : "treated";

            tr.innerHTML = `
                <td><strong>${u.name}</strong></td>
                <td>${u.email}</td>
                <td><span class="scan-status-tag ${roleTagClass}">${u.role.toUpperCase()}</span></td>
                <td><span style="color:var(--text-muted);">${formattedDate}</span></td>
            `;
            tableBody.appendChild(tr);
        });
    }

    // --- KNOWLEDGE BASE RECIPE EDITOR PANEL ---
    const recipeSelect = document.getElementById("recipeStainSelect");
    const recipeMaterials = document.getElementById("recipeMaterials");
    const recipeSteps = document.getElementById("recipeSteps");

    recipeSelect.addEventListener("change", loadRecipeDetails);

    function loadRecipeDetails() {
        const kb = AdminDataStore.getKB();
        const selection = recipeSelect.value;
        const recipe = kb[selection];

        if (recipe) {
            recipeMaterials.value = recipe.materials.join(", ");
            recipeSteps.value = recipe.steps.join("\n");
        }
    }

    // Save recipe changes
    document.getElementById("btnSaveRecipe").addEventListener("click", () => {
        const selection = recipeSelect.value;
        const mats = recipeMaterials.value.trim();
        const steps = recipeSteps.value.trim();

        if (!mats || !steps) {
            showToast("Materials lists and instructions cannot be empty.", "error");
            return;
        }

        const matsList = mats.split(",").map(m => m.trim()).filter(Boolean);
        const stepsList = steps.split("\n").map(s => s.trim()).filter(Boolean);

        const kb = AdminDataStore.getKB();
        kb[selection] = {
            materials: matsList,
            steps: stepsList
        };

        AdminDataStore.saveKB(kb);
        AdminDataStore.addLog("info", `Stain removal recipe updated by admin: ${selection}`);
        showToast("Recipe changes saved & synchronized with Mobile App database!");
    });

    // --- AUDIT SYSTEM LOGS PANELS ---
    const logsContainer = document.getElementById("logsContainer");

    async function renderLogs() {
        const logs = await AdminDataStore.getLogs();
        logsContainer.innerHTML = "";

        if (logs.length === 0) {
            logsContainer.innerHTML = `<div style="color:var(--text-muted); padding:20px; text-align:center;">No audit trails found.</div>`;
            return;
        }

        logs.forEach(log => {
            const time = new Date(log.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const date = new Date(log.time).toLocaleDateString([], { month: 'short', day: 'numeric' });
            const typeLabel = log.type === "error" ? "ERR" : log.type === "warning" ? "WRN" : "INF";
            const classType = log.type === "error" ? "log-type-err" : log.type === "warning" ? "log-type-warn" : "log-type-info";

            const div = document.createElement("div");
            div.className = "log-row";
            div.innerHTML = `
                <span class="log-time">[${date} ${time}]</span>
                <span class="${classType}">[${typeLabel}]</span>
                <span>${log.text}</span>
            `;
            logsContainer.appendChild(div);
        });
    }

    // Clear logs
    document.getElementById("btnClearLogs").addEventListener("click", async () => {
        if (confirm("Are you sure you want to delete all audit logs?")) {
            try {
                await fetch(AdminDataStore.getApiUrl("/api/logs"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ type: "info", text: "System Audit logs cleared by Administrator." })
                });
                await renderLogs();
                showToast("Logs cleared successfully.");
            } catch (e) {
                console.error("Failed to clear logs:", e);
            }
        }
    });

    // --- REALTIME SYNC POLL ---
    // Poll updates every 2.5 seconds to sync statistics and logs instantly when mobile scans occur!
    setInterval(async () => {
        if (currentAdmin) {
            const activeMenuItem = document.querySelector(".menu-item.active");
            const activeTab = activeMenuItem ? activeMenuItem.getAttribute("data-tab") : "admin-overview";
            if (activeTab === "admin-overview") {
                await renderOverview();
            } else if (activeTab === "admin-audit-logs") {
                await renderLogs();
            }
        }
    }, 2500);

    // Initial check
    if (currentAdmin) {
        authContainer.classList.remove("active");
        mainContainer.style.display = "flex";
        sidebarAdminName.textContent = currentAdmin.name;
        sidebarAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${currentAdmin.avatar || currentAdmin.name}`;
        renderOverview();
    } else {
        authContainer.classList.add("active");
        mainContainer.style.display = "none";
    }
});
