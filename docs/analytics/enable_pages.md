# How to Enable GitHub Pages for Analytics Dashboard

## Step-by-Step Instructions

1. **Go to your repository on GitHub:**
   - Visit: https://github.com/vinitwadgaonkar/vis_ifeval

2. **Navigate to Settings:**
   - Click on the **"Settings"** tab (top right of the repository page)

3. **Go to Pages section:**
   - Scroll down in the left sidebar
   - Click on **"Pages"** (under "Code and automation")

4. **Configure Pages:**
   - Under **"Source"**, select: **"Deploy from a branch"**
   - Under **"Branch"**, select: **"main"** (or "master" if that's your default)
   - Under **"Folder"**, select: **"/docs"** (NOT /docs/analytics - GitHub Pages will serve from /docs root)
   - Click **"Save"**

5. **Wait for deployment:**
   - GitHub will show: "Your site is being built from the main branch /docs folder"
   - Wait 1-2 minutes for the first deployment
   - You'll see a green checkmark when it's ready

6. **Access your dashboard:**
   - Your site will be available at: `https://vinitwadgaonkar.github.io/vis_ifeval/analytics/`
   - Note: The URL structure is: `https://[username].github.io/[repository-name]/[folder-path]`

## Alternative: Use GitHub Insights (No Setup Required!)

**You don't need GitHub Pages** - GitHub Insights is already available:

1. Go to: https://github.com/vinitwadgaonkar/vis_ifeval
2. Click on **"Insights"** tab
3. Click on **"Traffic"**
4. You'll see all visitor statistics immediately!

This is **private** and **only visible to you** (repository owner).

## Troubleshooting

- **404 Error**: Make sure you selected `/docs` as the folder (not `/docs/analytics`)
- **Still 404**: Wait a few more minutes for GitHub to build the site
- **Can't find Settings**: Make sure you're logged in and have admin access to the repository
- **No Pages option**: Your repository might need to be public, or you need GitHub Pro/Team

## Quick Check

After enabling Pages, you can verify it's working:
- Go to: https://github.com/vinitwadgaonkar/vis_ifeval/settings/pages
- You should see: "Your site is live at https://vinitwadgaonkar.github.io/vis_ifeval/"

