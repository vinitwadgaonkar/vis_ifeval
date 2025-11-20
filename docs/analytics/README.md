# Private Analytics Dashboard

**⚠️ PRIVATE - DO NOT SHARE THIS URL PUBLICLY**

This directory contains a private analytics dashboard for tracking repository visitors. Only the repository owner should access this.

## Access Instructions

1. **Via GitHub Pages** (if enabled):
   - Go to: `https://vinitwadgaonkar.github.io/vis_ifeval/analytics/`
   - This URL is not indexed by search engines (noindex, nofollow)

2. **Via GitHub Insights** (Recommended):
   - Go to your repository on GitHub
   - Click on the **"Insights"** tab
   - Click on **"Traffic"** to see:
     - Unique visitors
     - Total views
     - Clones
     - Referring sites
     - Popular content
   - This is only visible to repository owners

3. **Direct File Access**:
   - View `data/visits.json` directly in the repository
   - This file is updated by the GitHub Action

## How It Works

1. **GitHub Action** (`.github/workflows/analytics.yml`):
   - Runs hourly to aggregate visitor data
   - Tracks repository visits and clones
   - Updates `data/visits.json` with statistics

2. **Analytics Dashboard** (`index.html`):
   - Displays visitor statistics
   - Shows trends and recent visitors
   - Auto-refreshes every 5 minutes

3. **Privacy**:
   - No personal information is collected
   - IP addresses are hashed
   - Data is stored in the repository (private to you)

## Enabling GitHub Pages

To make the analytics dashboard accessible via GitHub Pages:

1. Go to repository **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** / **docs**
4. Folder: **/docs/analytics**
5. Save

The dashboard will be available at:
`https://vinitwadgaonkar.github.io/vis_ifeval/analytics/`

## Manual Tracking

If you want to manually track a visit, you can trigger the workflow:

```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/vinitwadgaonkar/vis_ifeval/dispatches \
  -d '{"event_type":"track-visit"}'
```

## Notes

- The analytics dashboard uses GitHub's built-in traffic data when available
- For more detailed tracking, consider using GitHub Insights (built-in)
- All data is stored locally in the repository
- The dashboard is not indexed by search engines

