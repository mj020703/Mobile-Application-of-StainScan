/* ==========================================================================
   STAINSCAN MOBILE APP CLIENT-SIDE CONTROLLER
   Simulates Convolutional Neural Network (CNN) feature extraction models,
   runs interactive cleaning checklists for users.
   ========================================================================== */

// --- Backend API URL Configuration ---
const CONFIG = {
    API_BASE_URL: (typeof window !== "undefined" && window.location.origin && window.location.origin !== "null" && !window.location.origin.startsWith("file:")) ? window.location.origin : "http://localhost:5000"
};

// --- Default Knowledge Base Recommendations & Precaution Dataset ---
const defaultKnowledgeBase = {
    "Black Ballpen Ink (White Cotton)": {
        stain: "Black Ballpen Ink",
        fabric_color: "White Cotton",
        materials: ["Paper Towels / Clean Cloth", "70% Isopropyl Alcohol", "Cotton Swabs", "Liquid Laundry Detergent", "Oxygen Bleach"],
        steps: [
            "Place a paper towel or clean cloth underneath the stained area.",
            "Dab 70% Isopropyl Alcohol onto a cotton swab and blot gently from the outer edges inward.",
            "Apply liquid laundry detergent directly to the stain and let sit for 5-10 minutes.",
            "Rinse thoroughly with cold water, then machine wash in warm water with oxygen bleach."
        ],
        caution: "Do not use chlorine bleach directly on ink spots, as it can fix the pigment into cotton fibers."
    },
    "Black Ballpen Ink (Grey Cotton)": {
        stain: "Black Ballpen Ink",
        fabric_color: "Grey Cotton",
        materials: ["Paper Towels", "70% Isopropyl Alcohol", "Clean Cloth", "Color-Safe Liquid Detergent"],
        steps: [
            "Turn garment inside out and place paper towel under the front.",
            "Dab 70% Isopropyl Alcohol on a cloth and gently blot the reverse side to push ink out.",
            "Apply color-safe liquid detergent directly to the spot.",
            "Rinse with cool water and wash on a cool cycle with color-safe detergent."
        ],
        caution: "Do not use acetone or chlorine bleach on grey fabric, as it will strip the dye."
    },
    "Used Cooking Oil (White Cotton)": {
        stain: "Used Cooking Oil",
        fabric_color: "White Cotton",
        materials: ["Baking Soda / Cornstarch", "Concentrated Liquid Dish Soap", "Hot Water", "Clean Brush / Cloth"],
        steps: [
            "Sprinkle baking soda or cornstarch on fresh oil for 10-15 minutes to absorb surface lipid, then brush away.",
            "Apply concentrated liquid dish soap directly onto the stain.",
            "Gently work soap into fibers and let sit for 15 minutes.",
            "Rinse with hot water, then wash in warm water."
        ],
        caution: "Ensure stain is fully removed before machine drying; heat permanently sets oil."
    },
    "Used Cooking Oil (Grey Cotton)": {
        stain: "Used Cooking Oil",
        fabric_color: "Grey Cotton",
        materials: ["Mild Liquid Dish Soap", "Soft-Bristled Brush / Clean Cloth", "Lukewarm Water", "Color-Safe Detergent"],
        steps: [
            "Dab mild liquid dish soap directly onto the grease spot.",
            "Softly work soap into the fabric using a soft-bristled brush or clean cloth.",
            "Let sit for 10-15 minutes to break down grease.",
            "Rinse with lukewarm water and wash in a standard warm cycle with color-safe detergent."
        ],
        caution: "Avoid aggressive scrubbing to prevent color fading or fuzzing on grey cotton."
    },
    "Mud (White Cotton)": {
        stain: "Mud",
        fabric_color: "White Cotton",
        materials: ["Dull Knife or Brush", "Oxygen Bleach", "Warm Water", "Liquid Laundry Detergent"],
        steps: [
            "Allow mud to dry completely (never clean wet mud).",
            "Gently scrape off hardened top crust with a dull knife or brush.",
            "Pre-soak in warm water with oxygen bleach for 30 minutes.",
            "Apply liquid detergent directly to remaining marks and machine wash warm."
        ],
        caution: "Washing wet mud forces fine clay deep into fabric weave."
    },
    "Mud (Grey Cotton)": {
        stain: "Mud",
        fabric_color: "Grey Cotton",
        materials: ["Soft-Bristle Brush", "Color-Safe Liquid Detergent", "Cool Water"],
        steps: [
            "Allow mud to dry completely.",
            "Gently brush off dry surface crust using a soft-bristled brush.",
            "Pre-soak in cool water mixed with color-safe liquid detergent for 20-30 minutes.",
            "Lightly rub remaining marks with liquid detergent and wash on a cool cycle."
        ],
        caution: "Use cool water pre-soaks to prevent fine soil particles from setting into colored fabric."
    }
};

// Aliases for backward compatibility
defaultKnowledgeBase["Black Ballpen Ink"] = defaultKnowledgeBase["Black Ballpen Ink (White Cotton)"];
defaultKnowledgeBase["Used Cooking Oil"] = defaultKnowledgeBase["Used Cooking Oil (White Cotton)"];
defaultKnowledgeBase["Mud"] = defaultKnowledgeBase["Mud (White Cotton)"];

// --- Hiligaynon Knowledge Base Recommendations & Precaution Dataset ---
const hiligaynonKnowledgeBase = {
    "Black Ballpen Ink (White Cotton)": {
        stain: "Itom nga Tinta sang Ballpen",
        fabric_color: "Puti nga Bulak (White Cotton)",
        materials: ["Paper Towels / Matinlo nga Tela", "70% Isopropyl Alcohol", "Cotton Swabs", "Liquid Laundry Detergent", "Oxygen Bleach"],
        steps: [
            "Ibutang ang paper towel ukon matinlo nga panapton sa idalom sang may digo nga parte.",
            "Ibutang ang 70% Isopropyl Alcohol sa cotton swab kag ihaplas sing mahinay halin sa gwa pasulod.",
            "Ibutang ang liquid laundry detergent deretso sa digo kag pabay-i sang 5-10 ka minuto.",
            "Banlawi sing maayo sa matugnaw nga tubig, dayon labhan sa malig-on nga tubig upod ang oxygen bleach."
        ],
        caution: "Indi maggamit sang chlorine bleach deretso sa digo sang tinta, kay mahimo ini magpabilin sang kolor sa mga hilo sang bulak."
    },
    "Black Ballpen Ink (Grey Cotton)": {
        stain: "Itom nga Tinta sang Ballpen",
        fabric_color: "Abohon nga Bulak (Grey Cotton)",
        materials: ["Paper Towels", "70% Isopropyl Alcohol", "Matinlo nga Tela", "Color-Safe Liquid Detergent"],
        steps: [
            "Pabaliskara ang bayo kag ibutang ang paper towel sa idalom sang atubang.",
            "Ibutang ang 70% Isopropyl Alcohol sa tela kag mahinay nga ihaplas sa likod agud magwa ang tinta.",
            "Ibutang ang color-safe liquid detergent deretso sa digo.",
            "Banlawi sa matugnaw nga tubig kag labhan gamit ang color-safe detergent."
        ],
        caution: "Indi maggamit sang acetone ukon chlorine bleach sa abohon nga tela, kay magakakas ini sang tina."
    },
    "Used Cooking Oil (White Cotton)": {
        stain: "Gingamit nga Mantika sa Pagluto",
        fabric_color: "Puti nga Bulak (White Cotton)",
        materials: ["Baking Soda / Cornstarch", "Concentrated Liquid Dish Soap", "Mainit nga Tubig", "Matinlo nga Sipilyo / Tela"],
        steps: [
            "Ibudbod ang baking soda ukon cornstarch sa preska nga mantika sang 10-15 ka minuto agud masuyop ang mantika, dayon silhigi.",
            "Ibutang ang concentrated liquid dish soap deretso sa digo.",
            "Mahinay nga ikuskos ang habon sa tela kag pabay-i sang 15 ka minuto.",
            "Banlawi sa mainit nga tubig, dayon labhan sa malig-on nga tubig."
        ],
        caution: "Siguruhon nga nadula sing bug-os ang digo antes ipa-uga sa makina; ang init nagapabilin sang mantika sa tela."
    },
    "Used Cooking Oil (Grey Cotton)": {
        stain: "Gingamit nga Mantika sa Pagluto",
        fabric_color: "Abohon nga Bulak (Grey Cotton)",
        materials: ["Mahinay nga Liquid Dish Soap", "Mahumok nga Sipilyo / Matinlo nga Tela", "Malig-on nga Tubig", "Color-Safe Detergent"],
        steps: [
            "Ibutang ang mahinay nga liquid dish soap deretso sa parte nga may mantika.",
            "Mahinay nga ikuskos ang habon gamit ang mahumok nga sipilyo ukon matinlo nga tela.",
            "Pabay-i sang 10-15 ka minuto agud magkatunaw ang mantika.",
            "Banlawi sa malig-on nga tubig kag labhan sa normal nga cycle upod ang color-safe detergent."
        ],
        caution: "Likawi ang tuman nga pagkusod agud indi maglubad ang kolor ukon maghimulmol ang abohon nga bulak."
    },
    "Mud (White Cotton)": {
        stain: "Lapok",
        fabric_color: "Puti nga Bulak (White Cotton)",
        materials: ["Mapurot nga Kutsilyo ukon Sipilyo", "Oxygen Bleach", "Malig-on nga Tubig", "Liquid Laundry Detergent"],
        steps: [
            "Pabay-i nga mag-uga sing bug-os ang lapok (indi gid magtinlo sang basa nga lapok).",
            "Mahinay nga kagison ang nagtikang nga lapok gamit ang mapurot nga kutsilyo ukon sipilyo.",
            "Ibabad sa malig-on nga tubig upod ang oxygen bleach sang 30 ka minuto.",
            "Ibutang ang liquid detergent deretso sa nabilin nga marka kag labhan sa malig-on nga tubig."
        ],
        caution: "Ang paglaba sang basa nga lapok nagapasulod sang yab-ok kag duta sing madalom sa tela."
    },
    "Mud (Grey Cotton)": {
        stain: "Lapok",
        fabric_color: "Abohon nga Bulak (Grey Cotton)",
        materials: ["Mahumok nga Sipilyo", "Color-Safe Liquid Detergent", "Matugnaw nga Tubig"],
        steps: [
            "Pabay-i nga mag-uga sing bug-os ang lapok.",
            "Mahinay nga sipilyuha ang uga nga lapok gamit ang mahumok nga sipilyo.",
            "Ibabad sa matugnaw nga tubig nga may miksla nga color-safe detergent sang 20-30 ka minuto.",
            "Mahinay nga kuskuson ang nabilin nga marka gamit ang liquid detergent kag labhan sa matugnaw nga tubig."
        ],
        caution: "Maggamit sang matugnaw nga tubig sa pagbabad agud indi magpabilin ang pinino nga duta sa may kolor nga tela."
    }
};

hiligaynonKnowledgeBase["Black Ballpen Ink"] = hiligaynonKnowledgeBase["Black Ballpen Ink (White Cotton)"];
hiligaynonKnowledgeBase["Used Cooking Oil"] = hiligaynonKnowledgeBase["Used Cooking Oil (White Cotton)"];
hiligaynonKnowledgeBase["Mud"] = hiligaynonKnowledgeBase["Mud (White Cotton)"];

function getCleaningRecommendation(stainName, fabricColorOrType, lang = null) {
    const activeLang = lang || (typeof I18N !== "undefined" ? I18N.currentLang : "en");
    const kb = activeLang === "hil" ? hiligaynonKnowledgeBase : defaultKnowledgeBase;
    const stainStr = String(stainName || "").toLowerCase();
    const fabricStr = String(fabricColorOrType || "").toLowerCase();
    const isGrey = fabricStr.includes("grey") || fabricStr.includes("gray") || fabricStr.includes("abohon");

    if (stainStr.includes("ballpen") || stainStr.includes("ink") || stainStr.includes("tinta")) {
        return isGrey ? kb["Black Ballpen Ink (Grey Cotton)"] : kb["Black Ballpen Ink (White Cotton)"];
    } else if (stainStr.includes("cooking") || stainStr.includes("oil") || stainStr.includes("mantika")) {
        return isGrey ? kb["Used Cooking Oil (Grey Cotton)"] : kb["Used Cooking Oil (White Cotton)"];
    } else if (stainStr.includes("mud") || stainStr.includes("lapok")) {
        return isGrey ? kb["Mud (Grey Cotton)"] : kb["Mud (White Cotton)"];
    }
    return isGrey ? kb["Used Cooking Oil (Grey Cotton)"] : kb["Used Cooking Oil (White Cotton)"];
}

// --- INTERNATIONALIZATION (i18n) TRANSLATION ENGINE ---
const I18N = {
    currentLang: (typeof window !== "undefined" && window.localStorage)
        ? (localStorage.getItem("stainscan_lang") || "en")
        : "en",
    onLanguageChange: null,
    translations: {
        en: {
            app_title: "StainScan",
            auth_subtitle: "AI-Powered Cotton Fabric Stain Care",
            label_email: "Email Address",
            placeholder_email: "name@email.com",
            label_password: "Password",
            placeholder_password: "••••••••",
            link_forgot_password: "Forgot Password?",
            btn_sign_in: "Sign In",
            text_dont_have_account: "Don't have an account?",
            link_sign_up: "Sign Up",
            label_fullname: "Full Name",
            placeholder_fullname: "John Doe",
            placeholder_min_password: "Min 6 characters",
            label_terms: "I agree to the Terms & Privacy Policy",
            btn_create_account: "Create Account",
            text_already_have_account: "Already have an account?",
            text_recovery_info: "Enter your email address and we'll send you instructions to reset your password.",
            btn_send_reset: "Send Reset Link",
            btn_back_to_login: "Back to Login",
            dashboard_greeting: "Let's save your cotton fabrics today.",
            banner_title: "Identify Stains Instantly",
            banner_desc: "Scan cotton garments to identify Mud, Cooking Oil, or Pen Ink and get cleaning recipes.",
            btn_start_scanning: "Start Scanning",
            stat_total_scans: "Total Scans",
            stat_stains_treated: "Stains Treated",
            recent_activity: "Recent Activity",
            link_view_all: "View All",
            empty_recent: "No scans performed yet. Try scanning a stain!",
            pro_tip_title: "Pro Tip for Cotton Fabrics",
            pro_tip_desc: "Always treat stains as early as possible. Fresh stains are significantly easier to extract than dried-in stains.",
            scan_title: "Analyze Fabric Stain",
            scan_subtitle: "Upload a clear photo of your stained cotton garment.",
            upload_tap: "Tap to Select Photo",
            upload_drag: "Or drag and drop image here",
            upload_limits: "JPG, PNG up to 10MB (Cotton Fabrics)",
            btn_analyze: "Begin Intelligent CNN Analysis",
            loader_preprocessing: "Preprocessing Image...",
            loader_reducing_noise: "Reducing background noise and resizing fabric details",
            badge_detection_successful: "Detection Successful",
            stain_classification: "Stain Classification",
            fabric_material: "Fabric Color & Material",
            cnn_confidence: "CNN Confidence",
            safety_precaution: "SAFETY PRECAUTION",
            safety_precaution_label: "Safety Precaution:",
            recommended_procedure: "Recommended Cleaning Procedure",
            steps_unit: "Steps",
            btn_open_checklist: "Open Interactive Checklist",
            btn_scan_another: "Scan Another Garment",
            modal_materials: "Required Materials",
            modal_steps: "Interactive Cleaning Steps",
            modal_step_tip: "Check off steps as you complete them to track your treatment progress.",
            modal_restoration_progress: "Garment Restoration Progress",
            btn_close_guide: "Close Guide",
            btn_mark_treatment_completed: "Mark Treatment Completed",
            treatment_completed_status: "Completed & Saved",
            step_prefix: "Step",
            history_title: "Scan History Log",
            history_subtitle: "Review and track your previous stain treatment plans.",
            placeholder_history_search: "Search stains or fabric...",
            filter_all: "All",
            filter_oil: "Oil",
            filter_ink: "Ink",
            filter_mud: "Mud",
            account_options: "Account Options",
            btn_edit_profile: "Edit Personal Details",
            btn_change_password: "Change Password",
            system_settings: "System Settings",
            notification_alerts: "Notification Alerts",
            btn_sign_out: "Sign Out",
            nav_home: "Home",
            nav_scan: "Scan",
            nav_history: "History",
            nav_profile: "Profile",
            stain_cooking_oil: "Used Cooking Oil",
            stain_ballpen_ink: "Black Ballpen Ink",
            stain_mud: "Mud",
            stain_unknown: "Unknown Stain"
        },
        hil: {
            app_title: "StainScan",
            auth_subtitle: "Pag-atipan sang Digo sa Tela nga Bulak paagi sa AI",
            label_email: "Email Address",
            placeholder_email: "ngalan@email.com",
            label_password: "Password",
            placeholder_password: "••••••••",
            link_forgot_password: "Nalipatan ang Password?",
            btn_sign_in: "Mag-sulod (Sign In)",
            text_dont_have_account: "Wala ka pa account?",
            link_sign_up: "Mag-rehistro (Sign Up)",
            label_fullname: "Bilog nga Ngalan",
            placeholder_fullname: "Juan Dela Cruz",
            placeholder_min_password: "Indi magnubo sa 6 ka karakter",
            label_terms: "Nagapasugot ako sa Kasugtanan kag Palisiya sa Pribasidad",
            btn_create_account: "Maghimo sang Account",
            text_already_have_account: "May account ka na?",
            text_recovery_info: "Ibutang ang imo email address kag padal-an ka namon sang instruksyon sa pag-reset sang imo password.",
            btn_send_reset: "Ipadala ang Link sa Pag-reset",
            btn_back_to_login: "Balik sa Pag-sulod",
            dashboard_greeting: "Salbaron naton ang imo mga bayo nga bulak subong.",
            banner_title: "Kilalaha ang mga Digo Gilayon",
            banner_desc: "I-scan ang mga bayo nga bulak agud makilala ang Lapok, Mantika, ukon Tinta kag makuha ang mga pamaagi sa pagpaninlo.",
            btn_start_scanning: "Sugdan ang Pag-scan",
            stat_total_scans: "Kabilugan nga Scan",
            stat_stains_treated: "Digo nga Natinluan",
            recent_activity: "Bag-o nga Hilikuton",
            link_view_all: "Tan-awon Tanan",
            empty_recent: "Wala pa sang na-scan. Tilawi mag-scan sang digo!",
            pro_tip_title: "Maayo nga Pahanumdom sa Telang Bulak",
            pro_tip_desc: "Tinlui pirme ang digo sang temprano pa. Ang preska nga digo mas mahapos kuhaon sangsa uga na.",
            scan_title: "Usisaon ang Digo sa Tela",
            scan_subtitle: "Mag-upload sang maathag nga litrato sang imo may digo nga bayo nga bulak (cotton).",
            upload_tap: "Pinduta agud Magpili sang Litrato",
            upload_drag: "Ukon idul-ong kag ihulog ang litrato diri",
            upload_limits: "JPG, PNG tubtob 10MB (Tela nga Bulak)",
            btn_analyze: "Sugdan ang Maalam nga Pagusisa sang CNN",
            loader_preprocessing: "Ginatipon kag Ginausisa ang Litrato...",
            loader_reducing_noise: "Ginapanubo ang gahod sa background kag ginasibu ang tela",
            badge_detection_successful: "Nadihit sang Matuod",
            stain_classification: "Klassipikasyon sang Digo",
            fabric_material: "Kolor kag Materyales sang Tela",
            cnn_confidence: "Kumpirmasyon sang CNN",
            safety_precaution: "PAHAMANGNO SA KALIG-ONAN",
            safety_precaution_label: "Pahamangno sa Kalig-onan:",
            recommended_procedure: "Gintuytoyan nga Pagpaninlo",
            steps_unit: "Mga Tikang",
            btn_open_checklist: "Apreha ang Sunod-sunod nga Listahan",
            btn_scan_another: "Mag-scan sang Lain nga Bayo",
            modal_materials: "Mga Ginakinahanglan nga Gamit",
            modal_steps: "Sunod-sunod nga Pagpaninlo",
            modal_step_tip: "Tiktiki ang mga tikang samtang ginahimo mo ini agud masubaybayan ang imo pagpaninlo.",
            modal_restoration_progress: "Kauswagan sang Pagpaninlo sang Bayo",
            btn_close_guide: "Isira ang Giya",
            btn_mark_treatment_completed: "Tiktikan nga Tapos na ang Pagpaninlo",
            treatment_completed_status: "Tapos na kag Na-save",
            step_prefix: "Tikang",
            history_title: "Listahan sang Na-scan",
            history_subtitle: "Tan-awa kag subaybaya ang imo mga nauna nga plano sa pagpaninlo.",
            placeholder_history_search: "Pangitaa ang digo ukon tela...",
            filter_all: "Tanan",
            filter_oil: "Mantika",
            filter_ink: "Tinta",
            filter_mud: "Lapok",
            account_options: "Mga Pililian sang Account",
            btn_edit_profile: "Bag-uhon ang Personal nga Detalye",
            btn_change_password: "Ilisan ang Password",
            system_settings: "Mga Pagsulundan sang Sistema",
            notification_alerts: "Pahibalo kag Alerts",
            btn_sign_out: "Maggwa (Sign Out)",
            nav_home: "Balay",
            nav_scan: "Scan",
            nav_history: "Kasaysayan",
            nav_profile: "Profile",
            stain_cooking_oil: "Gingamit nga Mantika sa Pagluto",
            stain_ballpen_ink: "Itom nga Tinta sang Ballpen",
            stain_mud: "Lapok",
            stain_unknown: "Wala Mahibal-an nga Digo"
        }
    },
    t(key) {
        const lang = this.currentLang || "en";
        if (this.translations[lang] && this.translations[lang][key] !== undefined) {
            return this.translations[lang][key];
        }
        if (this.translations.en && this.translations.en[key] !== undefined) {
            return this.translations.en[key];
        }
        return key;
    },
    tStain(stain) {
        if (!stain) return "";
        if (this.currentLang !== "hil") return stain;
        const s = String(stain).toLowerCase();
        if (s.includes("ballpen") || s.includes("ink") || s.includes("tinta")) return this.t("stain_ballpen_ink");
        if (s.includes("oil") || s.includes("cooking") || s.includes("mantika")) return this.t("stain_cooking_oil");
        if (s.includes("mud") || s.includes("lapok")) return this.t("stain_mud");
        if (s.includes("unknown")) return this.t("stain_unknown");
        return stain;
    },
    tFabric(fabric) {
        if (!fabric) return "";
        if (this.currentLang !== "hil") return fabric;
        return fabric
            .replace("White Cotton", "Puti nga Bulak (White Cotton)")
            .replace("Grey Cotton", "Abohon nga Bulak (Grey Cotton)")
            .replace("Cotton Fabric (100%)", "Tela nga Bulak (Cotton 100%)")
            .replace("Cotton Fabric", "Tela nga Bulak");
    },
    setLanguage(lang) {
        if (lang !== "en" && lang !== "hil") return;
        this.currentLang = lang;
        try {
            localStorage.setItem("stainscan_lang", lang);
        } catch (e) {
            console.warn("Could not save language to localStorage:", e);
        }
        this.updateButtons();
        this.applyTranslations();
        if (typeof this.onLanguageChange === "function") {
            try {
                this.onLanguageChange(lang);
            } catch (err) {
                console.error("Error running onLanguageChange handler:", err);
            }
        }
    },
    updateButtons() {
        const btnEn = document.getElementById("btnLangEn");
        const btnHil = document.getElementById("btnLangHil");
        if (btnEn && btnHil) {
            if (this.currentLang === "hil") {
                btnHil.classList.add("active");
                btnEn.classList.remove("active");
            } else {
                btnEn.classList.add("active");
                btnHil.classList.remove("active");
            }
        }
    },
    applyTranslations() {
        // 1. Text elements with data-i18n
        document.querySelectorAll("[data-i18n]").forEach(el => {
            const key = el.getAttribute("data-i18n");
            const translation = this.t(key);
            if (translation && translation !== key) {
                el.textContent = translation;
            }
        });

        // 2. Placeholder elements with data-i18n-placeholder
        document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
            const key = el.getAttribute("data-i18n-placeholder");
            const translation = this.t(key);
            if (translation && translation !== key) {
                el.setAttribute("placeholder", translation);
            }
        });

        // 3. Dynamic results procedure step count
        const resultProcedureStepCount = document.getElementById("resultProcedureStepCount");
        if (resultProcedureStepCount && resultProcedureStepCount.textContent) {
            const numMatch = resultProcedureStepCount.textContent.match(/\d+/);
            if (numMatch) {
                resultProcedureStepCount.textContent = `${numMatch[0]} ${this.t("steps_unit")}`;
            }
        }

        // 4. Mark treatment completed button
        const btnCompleteTreatment = document.getElementById("btnCompleteTreatment");
        if (btnCompleteTreatment) {
            const span = btnCompleteTreatment.querySelector("span");
            if (span && !btnCompleteTreatment.disabled) {
                span.textContent = this.t("btn_mark_treatment_completed");
            }
        }
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

    // --- App Launch Splash Screen Sequence ---
    const splashScreen = document.getElementById("splash-screen");
    const splashStatus = document.getElementById("splash-status");

    if (splashScreen) {
        // a. Keep the splash screen visible for ~1.5s on app launch
        // b. At 0.7s, update #splash-status text to "Readying AI Model..."
        setTimeout(() => {
            if (splashStatus) {
                splashStatus.style.opacity = "0";
                setTimeout(() => {
                    splashStatus.textContent = "Readying AI Model...";
                    splashStatus.style.opacity = "1";
                }, 150);
            }
        }, 700);

        // c. At 1.5s, play smooth fade-out / scale-down exit transition and reveal main view
        setTimeout(() => {
            splashScreen.classList.add("fade-out");
            setTimeout(() => {
                splashScreen.style.display = "none";
            }, 400);
        }, 1500);
    }

    // --- Language Selector Buttons ---
    const btnLangEn = document.getElementById("btnLangEn");
    const btnLangHil = document.getElementById("btnLangHil");
    if (btnLangEn) {
        btnLangEn.addEventListener("click", () => I18N.setLanguage("en"));
    }
    if (btnLangHil) {
        btnLangHil.addEventListener("click", () => I18N.setLanguage("hil"));
    }

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

    // Login Form Submit (proceeds directly without repeating splash intro)
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
                throw new Error("Unable to connect to the backend service. Please check server connection.");
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

    // Form Reset State helper
    function resetScanFormState() {
        selectedImageBase64 = null;
        scanResultData = null;
        fileInput.value = "";
        uploadedImagePreview.src = "";
        uploadPreviewState.style.display = "none";
        uploadEmptyState.style.display = "flex";
        scanControls.style.display = "none";
        resultsCard.style.display = "none";
        scanOverlayLine.style.display = "none";
        if (scannerLoader) scannerLoader.style.display = "none";
    }

    // Clear Preview
    btnClearPreview.addEventListener("click", (e) => {
        e.stopPropagation();
        resetScanFormState();
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
                body: JSON.stringify({
                    image: selectedImageBase64
                })
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

        const statusSteps = I18N.currentLang === "hil" ? [
            { text: "Ginatipon kag Ginausisa ang Litrato...", sub: "Ginapanubo ang background noise kag ginasibu ang contrast", limit: 25 },
            { text: "Ginatan-aw ang Materyal sang Tela...", sub: "Ginasiguro nga ang tela nagasanto sa bulak (cotton)", limit: 50 },
            { text: "Ginatinguhaan sang CNN Classifier...", sub: "Ginakuha ang mga detalye sang digo kag pat-od nga layer", limit: 75 },
            { text: "Ginakumpirma ang Resulta...", sub: "Ginasukol ang kumpirmasyon sang Softmax score", limit: 100 }
        ] : [
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
                            const rawStain = predictionResult.stain_classification || predictionResult.stain || "Unknown Stain";
                            let displayStain = predictionResult.stain || rawStain;
                            if (rawStain.toLowerCase().includes("ballpen") || rawStain.toLowerCase().includes("ink")) {
                                displayStain = "Black Ballpen Ink";
                            } else if (rawStain.toLowerCase().includes("cooking") || rawStain.toLowerCase().includes("oil")) {
                                displayStain = "Used Cooking Oil";
                            } else if (rawStain.toLowerCase().includes("mud")) {
                                displayStain = "Mud";
                            }
                            const fabricMat = predictionResult.fabric || (predictionResult.fabric_color ? `${predictionResult.fabric_color} (100%)` : "Cotton Fabric (100%)");
                            processScanningSuccess(displayStain, fabricMat, predictionResult.confidence, predictionResult);
                        } else {
                            const errMsg = (predictionResult && predictionResult.error) ? predictionResult.error : "Unknown backend error";
                            processScanningFailure(errMsg);
                        }
                    } else {
                        const simRec = getCleaningRecommendation(simStain, simFabric, I18N.currentLang);
                        processScanningSuccess(simStain, simFabric, simConfidence, simRec);
                    }
                    return;
                }
            }

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

    function processScanningSuccess(stain, fabric, confidence, recommendation = null) {
        const isCotton = fabric.includes("Cotton") || fabric.includes("Bulak");
        const isStainSupported = stain !== "Unknown Stain";

        // Extract or lookup precise dynamic recommendation based on active language
        let rec = getCleaningRecommendation(stain, fabric, I18N.currentLang);
        if (!rec && recommendation && recommendation.steps && recommendation.steps.length) {
            rec = recommendation;
        } else if (!rec && recommendation && recommendation.recommendation && recommendation.recommendation.steps) {
            rec = recommendation.recommendation;
        }

        const steps = (rec && rec.steps) ? rec.steps : [];
        const caution = (rec && rec.caution) ? rec.caution : "";
        const materials = (rec && rec.materials) ? rec.materials : [];
        const fabricColor = (rec && rec.fabric_color) ? rec.fabric_color : fabric;

        // Reset badge styles in case a previous scan failed
        const resultsBadge = document.querySelector(".results-card .results-badge");
        resultsBadge.innerHTML = `<i class="fas fa-check-circle"></i> <span data-i18n="badge_detection_successful">${I18N.t("badge_detection_successful")}</span>`;
        resultsBadge.style.backgroundColor = "";
        resultsBadge.style.color = "";
        document.getElementById("btnViewCleaningGuide").style.display = "inline-flex";

        const resultStainType = document.getElementById("resultStainType");
        const resultFabricType = document.getElementById("resultFabricType");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultConfidenceBar = document.getElementById("resultConfidenceBar");
        const resultCautionCard = document.getElementById("resultCautionCard");
        const resultCautionText = document.getElementById("resultCautionText");
        const resultCautionTitle = document.getElementById("resultCautionTitle");
        const resultProcedureCard = document.getElementById("resultProcedureCard");
        const resultProcedureStepCount = document.getElementById("resultProcedureStepCount");

        scanResultData = {
            id: "h_" + Math.random().toString(36).substr(2, 9),
            email: currentUser.email,
            stain: stain,
            fabric: fabricColor.includes("100%") ? fabricColor : `${fabricColor} (100%)`,
            fabric_color: fabricColor,
            confidence: confidence,
            steps: steps,
            caution: caution,
            materials: materials,
            timestamp: new Date().toISOString(),
            status: "Pending",
            image: ""
        };

        resultStainType.textContent = I18N.tStain(stain);
        resultFabricType.textContent = I18N.tFabric(scanResultData.fabric);
        resultConfidence.textContent = `${confidence}%`;
        resultConfidenceBar.style.width = `${confidence}%`;

        // Render Highlighted Caution Card
        if (caution && resultCautionCard && resultCautionText) {
            resultCautionText.textContent = caution;
            if (resultCautionTitle) {
                resultCautionTitle.textContent = I18N.currentLang === "hil"
                    ? `Pahibalo sa ${I18N.tStain(stain)}`
                    : `${stain} (${fabricColor.replace(' (100%)', '')}) Alert`;
            }
            resultCautionCard.style.display = "block";
        } else if (resultCautionCard) {
            resultCautionCard.style.display = "none";
        }

        // Render Recommended Cleaning Procedure Card Header and Badge (summary only)
        if (steps.length > 0 && resultProcedureCard) {
            if (resultProcedureStepCount) {
                resultProcedureStepCount.textContent = `${steps.length} ${I18N.t("steps_unit")}`;
            }
            resultProcedureCard.style.display = "block";
        } else if (resultProcedureCard) {
            resultProcedureCard.style.display = "none";
        }

        if (!isCotton) {
            document.getElementById("btnViewCleaningGuide").style.display = "none";
            if (resultCautionCard) resultCautionCard.style.display = "none";
            if (resultProcedureCard) resultProcedureCard.style.display = "none";
            DataStore.addLog("warning", `Unsupported fabric detected: ${fabric}`);
            showToast("Fabric Mismatch: Treatment plans are calibrated for 100% Cotton only.", "warning");
        } else if (!isStainSupported) {
            document.getElementById("btnViewCleaningGuide").style.display = "none";
            if (resultCautionCard) resultCautionCard.style.display = "none";
            if (resultProcedureCard) resultProcedureCard.style.display = "none";
            DataStore.addLog("warning", `Unknown stain classified.`);
            showToast("Unsupported Stain: Unable to classify stain parameters.", "warning");
        } else {
            document.getElementById("btnViewCleaningGuide").style.display = "inline-flex";
        }

        DataStore.saveHistoryCloud(scanResultData);
        DataStore.addLog("info", `Successful CNN scan: ${stain} on ${fabric} (${confidence}% confidence)`);

        resultsCard.style.display = "block";
        showToast(I18N.currentLang === "hil" ? "Tapos na ang Pag-scan sang Tela!" : "Fabric Scan Complete!");
    }

    function processScanningFailure(errorMessage) {
        const resultStainType = document.getElementById("resultStainType");
        const resultFabricType = document.getElementById("resultFabricType");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultConfidenceBar = document.getElementById("resultConfidenceBar");
        const resultsBadge = document.querySelector(".results-card .results-badge");
        const btnViewCleaningGuide = document.getElementById("btnViewCleaningGuide");

        resultsBadge.innerHTML = `<i class="fas fa-circle-exclamation"></i> Analysis Failed`;
        resultsBadge.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
        resultsBadge.style.color = "#ef4444";

        resultStainType.innerHTML = `<span style="color:#ef4444; font-weight: normal;">Backend Error: ${errorMessage}</span>`;
        resultFabricType.textContent = "N/A";
        resultConfidence.textContent = "0%";
        resultConfidenceBar.style.width = "0%";

        btnViewCleaningGuide.style.display = "none";

        const resultCautionCard = document.getElementById("resultCautionCard");
        const resultProcedureCard = document.getElementById("resultProcedureCard");
        if (resultCautionCard) resultCautionCard.style.display = "none";
        if (resultProcedureCard) resultProcedureCard.style.display = "none";

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
        
        let recipe = getCleaningRecommendation(historyItem.stain, historyItem.fabric || historyItem.fabric_color, I18N.currentLang);
        if (!recipe && historyItem.steps && historyItem.steps.length) {
            recipe = {
                materials: historyItem.materials || ["Liquid Detergent", "Clean Water", "Clean Cloth"],
                steps: historyItem.steps,
                caution: historyItem.caution || ""
            };
        }

        if (!recipe) {
            showToast("No active cleaning guide registered for this stain.", "error");
            return;
        }

        const fabricLabel = historyItem.fabric_color || historyItem.fabric || "Cotton Fabric";
        modalStainName.textContent = I18N.currentLang === "hil"
            ? `Giya sa Pagpaninlo sang ${I18N.tStain(historyItem.stain)}`
            : `${historyItem.stain} Care (${fabricLabel.replace(' (100%)', '')})`;

        const modalFabricSafetyText = document.getElementById("modalFabricSafetyText");
        if (modalFabricSafetyText) {
            if (I18N.currentLang === "hil") {
                modalFabricSafetyText.innerHTML = `Aprobado nga Plano sa Pagpaninlo para sa <strong>${I18N.tFabric(fabricLabel)}</strong>`;
            } else {
                modalFabricSafetyText.innerHTML = `Approved Treatment Plan for <strong>${fabricLabel}</strong>`;
            }
        }

        const modalCautionBox = document.getElementById("modalCautionBox");
        const modalCautionText = document.getElementById("modalCautionText");
        if (modalCautionBox && modalCautionText) {
            if (recipe.caution) {
                modalCautionText.textContent = recipe.caution;
                modalCautionBox.style.display = "flex";
            } else {
                modalCautionBox.style.display = "none";
            }
        }

        modalMaterialsList.innerHTML = "";
        (recipe.materials || []).forEach(mat => {
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
                    <span class="step-num-title">${I18N.t("step_prefix")} ${idx + 1}</span>
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
            btnCompleteTreatment.innerHTML = `<i class="fas fa-check-double"></i> <span>${I18N.t("treatment_completed_status")}</span>`;
        } else {
            btnCompleteTreatment.disabled = true;
            btnCompleteTreatment.innerHTML = `<i class="fas fa-circle-check"></i> <span data-i18n="btn_mark_treatment_completed">${I18N.t("btn_mark_treatment_completed")}</span>`;
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

        const isAllChecked = (total > 0 && checked === total);

        // Check if current active scan is already marked Treated
        const history = DataStore.getHistory();
        const activeItem = history.find(h => (h.id === activeGuideScanId || h._id === activeGuideScanId));
        const isAlreadyTreated = activeItem && activeItem.status === "Treated";

        if (isAlreadyTreated) {
            btnCompleteTreatment.disabled = true;
            btnCompleteTreatment.classList.remove("active");
            btnCompleteTreatment.innerHTML = `<i class="fas fa-check-double"></i> <span>${I18N.t("treatment_completed_status")}</span>`;
        } else {
            btnCompleteTreatment.disabled = !isAllChecked;
            if (isAllChecked) {
                btnCompleteTreatment.classList.add("active");
                btnCompleteTreatment.innerHTML = `<i class="fas fa-circle-check"></i> <span data-i18n="btn_mark_treatment_completed">${I18N.t("btn_mark_treatment_completed")}</span>`;
            } else {
                btnCompleteTreatment.classList.remove("active");
                btnCompleteTreatment.innerHTML = `<i class="fas fa-circle-check"></i> <span data-i18n="btn_mark_treatment_completed">${I18N.t("btn_mark_treatment_completed")}</span>`;
            }
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

            // a. Show the 'Treatment Completed' notification alert/toast
            showToast(I18N.currentLang === "hil" ? "Tapos na ang Pagpaninlo!" : "Treatment Completed!", "success");

            // Close the guide modal
            guideModal.style.display = "none";
            activeGuideScanId = null;

            // b. Reset the scan form state (clear uploaded image preview and reset inputs)
            resetScanFormState();

            // Refresh dashboard and history lists
            renderDashboard();
            renderHistory();

            // c. Automatically redirect the user back to the 'Analyze Fabric Stain' view
            navigateTo("screen-scan");

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

    // --- Internationalization Runtime Hook ---
    I18N.onLanguageChange = (lang) => {
        // 1. Dynamic scan results update if present
        if (scanResultData) {
            const rec = getCleaningRecommendation(scanResultData.stain, scanResultData.fabric, lang);
            const resultStainType = document.getElementById("resultStainType");
            const resultFabricType = document.getElementById("resultFabricType");
            const resultCautionText = document.getElementById("resultCautionText");
            const resultCautionTitle = document.getElementById("resultCautionTitle");
            const resultProcedureStepCount = document.getElementById("resultProcedureStepCount");
            const resultsBadge = document.querySelector(".results-card .results-badge");

            if (resultStainType) resultStainType.textContent = I18N.tStain(scanResultData.stain);
            if (resultFabricType) resultFabricType.textContent = I18N.tFabric(scanResultData.fabric);
            if (resultCautionText && rec && rec.caution) resultCautionText.textContent = rec.caution;
            if (resultCautionTitle) {
                resultCautionTitle.textContent = lang === "hil"
                    ? `Pahibalo sa ${I18N.tStain(scanResultData.stain)}`
                    : `${scanResultData.stain} (${(scanResultData.fabric_color || "").replace(' (100%)', '')}) Alert`;
            }
            if (resultProcedureStepCount && rec && rec.steps) {
                resultProcedureStepCount.textContent = `${rec.steps.length} ${I18N.t("steps_unit")}`;
            }
            if (resultsBadge) {
                resultsBadge.innerHTML = `<i class="fas fa-check-circle"></i> <span data-i18n="badge_detection_successful">${I18N.t("badge_detection_successful")}</span>`;
            }
        }

        // 2. Active guide modal update if open
        if (guideModal && guideModal.style.display !== "none" && activeGuideScanId) {
            const history = DataStore.getHistory();
            const activeItem = history.find(h => (h.id === activeGuideScanId || h._id === activeGuideScanId)) || scanResultData;
            if (activeItem) {
                openCleaningGuide(activeItem);
            }
        }

        // 3. Active screen updates
        if (currentUser) {
            dashboardUserName.textContent = currentUser.name;
            const currentScreen = screens.find(s => s && s.classList.contains("active"));
            if (currentScreen && currentScreen.id === "screen-dashboard") {
                renderDashboard();
            } else if (currentScreen && currentScreen.id === "screen-history") {
                renderHistory();
            }
        }
    };

    I18N.updateButtons();
    I18N.applyTranslations();

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
