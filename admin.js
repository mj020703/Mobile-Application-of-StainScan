/* ==========================================================================
   STAINSCAN ADMIN WEB PORTAL CONTROLLER
   Handles administrator authentication, system synchronization with the shared
   localStorage database, recipe modifications, and analytics updates.
   ========================================================================== */

class AdminDataStore {
    static init() {
        if (!localStorage.getItem("stainscan_users")) {
            const defaultUsers = [
                { name: "Admin Manager", email: "admin@stainscan.com", password: "admin123", role: "admin", avatar: "admin" },
                { name: "John Doe", email: "user@stainscan.com", password: "user123", role: "user", avatar: "John" }
            ];
            localStorage.setItem("stainscan_users", JSON.stringify(defaultUsers));
        }
        let currentKB = null;
        try {
            currentKB = JSON.parse(localStorage.getItem("stainscan_kb"));
        } catch(e) {
            currentKB = null;
        }

        if (!currentKB || typeof currentKB !== "object" || !currentKB["Used Cooking Oil"] || !currentKB["Black Ballpen Ink"] || !currentKB["Mud"]) {
            const defaultKB = {
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
            localStorage.setItem("stainscan_kb", JSON.stringify(defaultKB));
        }
        if (!localStorage.getItem("stainscan_history")) {
            localStorage.setItem("stainscan_history", JSON.stringify([]));
        }
        if (!localStorage.getItem("stainscan_logs")) {
            localStorage.setItem("stainscan_logs", JSON.stringify([]));
        }
    }
    static getKB() { return JSON.parse(localStorage.getItem("stainscan_kb")) || {}; }
    static saveKB(kb) { localStorage.setItem("stainscan_kb", JSON.stringify(kb)); }
    static getUsers() { return JSON.parse(localStorage.getItem("stainscan_users")) || []; }
    static getHistory() { return JSON.parse(localStorage.getItem("stainscan_history")) || []; }
    static getLogs() { return JSON.parse(localStorage.getItem("stainscan_logs")) || []; }
    static saveLogs(logs) { localStorage.setItem("stainscan_logs", JSON.stringify(logs)); }

    static addLog(type, text) {
        const logs = this.getLogs();
        logs.unshift({ time: new Date().toISOString(), type, text });
        this.saveLogs(logs.slice(0, 100));
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
        const email = document.getElementById("admin-email").value.trim().toLowerCase();
        const password = document.getElementById("admin-password").value;

        const users = AdminDataStore.getUsers();
        const matched = users.find(u => u.email === email && u.password === password);

        if (matched) {
            if (matched.role === "admin") {
                currentAdmin = matched;
                sessionStorage.setItem("stainscan_current_admin", JSON.stringify(matched));
                
                // Show Admin portal UI
                authContainer.classList.remove("active");
                mainContainer.style.display = "flex";
                
                sidebarAdminName.textContent = matched.name;
                sidebarAvatar.src = `https://api.dicebear.com/7.x/bottts/svg?seed=${matched.avatar || matched.name}`;
                
                showToast(`Access granted. Welcome, ${matched.name}!`);
                AdminDataStore.addLog("info", `Admin dashboard unlocked by: ${matched.email}`);
                
                // Load Dashboard
                renderOverview();
            } else {
                showToast("Access Denied. Account does not possess System Admin role.", "error");
                AdminDataStore.addLog("warning", `Restricted access blocked to admin panel: ${email}`);
            }
        } else {
            showToast("Invalid credentials entered. Try again.", "error");
        }
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

    // --- OVERVIEW & ANALYTICS PANELS ---
    function renderOverview() {
        const users = AdminDataStore.getUsers();
        const history = AdminDataStore.getHistory();

        // Totals counting
        document.getElementById("totalUsersCount").textContent = users.length;
        document.getElementById("totalScansCount").textContent = history.length;
        
        const treatedCount = history.filter(h => h.status === "Treated").length;
        document.getElementById("totalTreatedCount").textContent = treatedCount;

        // Render Stain Distribution SVG Charts
        const chartContainer = document.getElementById("adminStainChart");
        chartContainer.innerHTML = "";

        const counts = {
            "Used Cooking Oil": 0,
            "Black Ballpen Ink": 0,
            "Mud": 0
        };

        history.forEach(item => {
            if (counts[item.stain] !== undefined) {
                counts[item.stain]++;
            }
        });

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
    }

    // --- REGISTERED USERS DIRECTORY PANELS ---
    function renderUsers() {
        const users = AdminDataStore.getUsers();
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

    function renderLogs() {
        const logs = AdminDataStore.getLogs();
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
    document.getElementById("btnClearLogs").addEventListener("click", () => {
        if (confirm("Are you sure you want to delete all audit logs?")) {
            localStorage.setItem("stainscan_logs", JSON.stringify([]));
            AdminDataStore.addLog("info", "System Audit logs cleared by Administrator.");
            renderLogs();
            showToast("Logs cleared successfully.");
        }
    });

    // --- REALTIME SYNC POLL ---
    // Poll updates every 4 seconds to sync statistics and logs instantly when mobile scans occur!
    setInterval(() => {
        if (currentAdmin) {
            const activeTab = document.querySelector(".menu-item.active").getAttribute("data-tab");
            if (activeTab === "admin-overview") {
                renderOverview();
            } else if (activeTab === "admin-audit-logs") {
                renderLogs();
            }
        }
    }, 4000);

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
