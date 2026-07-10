# StainScan - Android APK Build Guide

This guide details the fastest methods to package your StainScan web assets (HTML, CSS, JS) into a downloadable Android `.apk` file.

---

## Method 1: WebIntoApp (Fastest & No Code)
[WebIntoApp](https://www.webintoapp.com/) is a cloud-based service that packages web projects into Android App Bundles (AAB) or APKs directly in the browser without requiring any local toolchains (Node.js, Android Studio, Gradle).

### Steps:
1. **Prepare Your Files**:
   - Locate the project folder containing the files:
     - [index.html](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/index.html)
     - [admin.html](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/admin.html)
     - [app.js](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/app.js)
     - [admin.js](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/admin.js)
     - [style.css](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/style.css)
   - Create a ZIP archive containing these files (ensure `index.html` is at the root level of the ZIP archive, not inside a subfolder).
2. **Configure on WebIntoApp**:
   - Go to [WebIntoApp.com](https://www.webintoapp.com/).
   - Click on **Get Started** or **Make App**.
   - Select **HTML Files (Local ZIP Archive)** as the application source.
   - Upload your ZIP archive.
   - Enter your App details:
     - **App Name**: `StainScan`
     - **Package Name**: `com.stainscan.app`
3. **Build and Download**:
   - Click **Generate Android App** (you may be asked to create a free account).
   - Once the build finishes, download the generated `.zip` file which contains the release `.apk` file.
   - Transfer the `.apk` file to your Android device and install it.

---

## Method 2: Capacitor (Developer CLI Method)
[Capacitor](https://capacitorjs.com/) is a modern hybrid app runtime by Ionic that turns web apps into native iOS/Android apps with full native access.

### Prerequisites:
- [Node.js](https://nodejs.org/) (LTS version recommended)
- [Android Studio](https://developer.android.com/studio) (for building the Android APK)

### Steps:

1. **Initialize NPM** (if not already done):
   Open your terminal in the workspace directory and run:
   ```bash
   npm init -y
   ```

2. **Install Capacitor Core and CLI**:
   ```bash
   npm install @capacitor/core @capacitor/cli
   ```

3. **Initialize Capacitor Config**:
   ```bash
   npx cap init StainScan com.stainscan.app --web-dir=.
   ```
   *Note: `--web-dir=.` tells Capacitor that your web assets are located in the root of the project directory. If you copy them to a build folder (e.g. `www`), set `--web-dir=www`.*

4. **Install and Add the Android Platform**:
   ```bash
   npm install @capacitor/android
   npx cap add android
   ```

5. **Sync Assets with the Android Project**:
   Whenever you make changes to your HTML, CSS, or JS files, run the sync command to copy them into the native Android folder:
   ```bash
   npx cap sync
   ```

6. **Open in Android Studio & Build APK**:
   Launch Android Studio with the Capacitor project configured:
   ```bash
   npx cap open android
   ```
   Once Android Studio opens:
   - Wait for Gradle sync to complete.
   - In the top menu, go to **Build** > **Build Bundle(s) / APK(s)** > **Build APK(s)**.
   - Android Studio will build the APK file. When finished, a notification popup will appear in the bottom right corner with a **Locate** link to open the folder containing your generated `app-debug.apk` file.

---

## Hybrid Network Considerations
- **CORS Config**: Your backend [server.py](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/server.py) is now configured with wildcard origins `*`, allowing the packaged web layers to hit the Flask endpoints directly.
- **Backend Routing**: The client application [app.js](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/app.js) dynamically detects if it is running on a local development server or packaged inside a hybrid environment, automatically shifting requests to your live Render instance.
