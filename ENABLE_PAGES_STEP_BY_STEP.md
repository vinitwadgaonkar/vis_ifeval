# 🔧 Step-by-Step: Enable GitHub Pages for Analytics

## ⚠️ IMPORTANT: You Must Do This Manually in GitHub Web Interface

GitHub Pages cannot be enabled via command line - you need to do it through the web interface.

---

## 📋 Exact Steps (with Screenshots Guide)

### Step 1: Go to Your Repository Settings
1. Open: https://github.com/vinitwadgaonkar/vis_ifeval
2. Click on the **"Settings"** tab (it's at the top of the repository page, next to "Code", "Issues", "Pull requests", etc.)

### Step 2: Find Pages Section
1. In the left sidebar, scroll down
2. Look for **"Pages"** under the "Code and automation" section
3. Click on **"Pages"**

### Step 3: Configure Pages
1. Under **"Source"**, you'll see a dropdown - select: **"Deploy from a branch"**
2. Under **"Branch"**, select: **"main"** (or "master" if that's your default branch)
3. Under **"Folder"**, select: **"/docs"** ⚠️ **IMPORTANT: Select `/docs`, NOT `/docs/analytics`**
4. Click the **"Save"** button

### Step 4: Wait for Deployment
1. You'll see a message: "Your site is being built from the main branch /docs folder"
2. Wait 1-2 minutes
3. You'll see a green checkmark ✅ when it's ready
4. The URL will appear: `https://vinitwadgaonkar.github.io/vis_ifeval/`

### Step 5: Access Your Dashboard
- Your analytics dashboard will be at: `https://vinitwadgaonkar.github.io/vis_ifeval/analytics/`

---

## 🎯 Direct Links

- **Repository Settings**: https://github.com/vinitwadgaonkar/vis_ifeval/settings
- **Pages Settings**: https://github.com/vinitwadgaonkar/vis_ifeval/settings/pages

---

## ✅ Alternative: Use GitHub Insights (NO SETUP NEEDED!)

**You don't need GitHub Pages!** GitHub Insights works immediately:

👉 **Click here**: https://github.com/vinitwadgaonkar/vis_ifeval/insights/traffic

This shows:
- ✅ Unique visitors
- ✅ Total views
- ✅ Clones
- ✅ Referring sites
- ✅ Popular content

**This is PRIVATE** - only you can see it!

---

## ❓ Troubleshooting

### "Pages" option not visible?
- Make sure you're logged in
- You need to be the repository owner or have admin access
- Some repositories require GitHub Pro/Team for Pages (but free accounts work for public repos)

### Still getting 404 after enabling?
- Make sure you selected `/docs` (not `/docs/analytics`) as the folder
- Wait 2-3 minutes for GitHub to build the site
- Check the Actions tab to see if there are any build errors
- Verify the branch is `main` (not `master`)

### Can't find Settings tab?
- Make sure you're on the repository main page
- You need admin/owner permissions
- Try: https://github.com/vinitwadgaonkar/vis_ifeval/settings/pages

---

## 💡 My Recommendation

**Just use GitHub Insights** - it's easier and works immediately:
- No setup required
- Real-time data from GitHub
- Fully private (only you can see it)
- Works right now: https://github.com/vinitwadgaonkar/vis_ifeval/insights/traffic

The GitHub Pages dashboard is optional - it just provides a prettier interface with charts.

