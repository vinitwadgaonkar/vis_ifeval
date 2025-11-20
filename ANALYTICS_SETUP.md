# Private Analytics Setup Guide

This guide explains how to set up and access private visitor tracking for your GitHub repository.

## Quick Setup

1. **Enable GitHub Insights** (Easiest - Built-in):
   - Go to: `https://github.com/vinitwadgaonkar/vis_ifeval`
   - Click on **"Insights"** tab
   - Click on **"Traffic"**
   - You'll see:
     - Unique visitors
     - Total views
     - Clones
     - Referring sites
     - Popular content
   - ⚠️ **Only you can see this** (repository owner only)

2. **Enable GitHub Pages for Analytics Dashboard** (Optional):
   - Go to repository **Settings** → **Pages**
   - Source: **Deploy from a branch**
   - Branch: **main**
   - Folder: **/docs/analytics**
   - Click **Save**
   - Wait 1-2 minutes for deployment
   - Access at: `https://vinitwadgaonkar.github.io/vis_ifeval/analytics/`
   - ⚠️ **Keep this URL private** - don't share it publicly

3. **GitHub Action** (Automatic):
   - The workflow (`.github/workflows/analytics.yml`) runs automatically
   - It aggregates visitor data every hour
   - Updates `docs/analytics/data/visits.json`
   - Commits are marked with `[skip ci]` to avoid loops

## Accessing Analytics

### Method 1: GitHub Insights (Recommended)
- **URL**: `https://github.com/vinitwadgaonkar/vis_ifeval/insights/traffic`
- **Visibility**: Only you (repository owner)
- **Data**: Real-time from GitHub
- **No setup required** - works immediately

### Method 2: Analytics Dashboard
- **URL**: `https://vinitwadgaonkar.github.io/vis_ifeval/analytics/` (after enabling Pages)
- **Visibility**: Private (not indexed, but accessible if URL is known)
- **Data**: Aggregated by GitHub Action
- **Features**: Charts, trends, recent visitors

### Method 3: Direct File Access
- **File**: `docs/analytics/data/visits.json`
- **Visibility**: Only you (if repo is private) or anyone with repo access
- **Format**: JSON
- **Updated**: Hourly by GitHub Action

## Privacy & Security

✅ **What's Private:**
- GitHub Insights: Only visible to repository owner
- Analytics dashboard: Not indexed by search engines (noindex, nofollow)
- Raw data: Stored in repository (private if repo is private)

⚠️ **Important Notes:**
- If your repository is **public**, the analytics dashboard URL could be discovered
- If your repository is **private**, only you and collaborators can access it
- IP addresses are hashed for privacy
- No personal information is collected

## Troubleshooting

### GitHub Insights not showing data?
- Make sure your repository is public (or you have GitHub Pro/Team)
- Wait 24-48 hours for initial data collection
- Check that the repository has some activity

### Analytics Dashboard not loading?
- Check that GitHub Pages is enabled
- Verify the branch and folder settings
- Wait 1-2 minutes after enabling Pages
- Check the Actions tab for workflow runs

### GitHub Action not running?
- Check `.github/workflows/analytics.yml` exists
- Verify the workflow file is valid YAML
- Check the Actions tab for errors
- Ensure you have write permissions

## Manual Tracking

If you want to manually trigger tracking:

```bash
# Using the Python script
python scripts/analytics/track_visit.py [IP] [User-Agent] [Referrer]

# Using GitHub API
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/vinitwadgaonkar/vis_ifeval/dispatches \
  -d '{"event_type":"track-visit"}'
```

## Next Steps

1. ✅ Enable GitHub Insights (if not already enabled)
2. ✅ Commit and push the analytics files
3. ✅ Enable GitHub Pages (optional)
4. ✅ Wait for first data collection
5. ✅ Access your private analytics dashboard

## Support

For issues or questions:
- Check GitHub Actions logs
- Verify file permissions
- Ensure GitHub Pages is properly configured
- Review the analytics workflow file

