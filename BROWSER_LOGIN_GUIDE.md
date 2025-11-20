# Browser-Based GitHub Login Guide

## Method 1: Personal Access Token (Recommended)

### Step 1: Create Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "vinit_benchmark_push"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

### Step 2: Push Using Token
```bash
cd /home/csgrad/vinitwad/vinit_benchmark
git push origin main
```

When prompted:
- **Username**: `vinitwadgaonkar`
- **Password**: `<paste_your_personal_access_token_here>`

The credentials will be saved for future pushes.

---

## Method 2: GitHub CLI (Browser Login)

If you can install GitHub CLI with sudo:

```bash
# Install GitHub CLI
sudo apt-get update
sudo apt-get install -y gh

# Login via browser
gh auth login --web

# Follow the browser prompts to authenticate
# Then push:
git push origin main
```

---

## Method 3: SSH Key (One-time Setup)

### Step 1: Copy Your SSH Public Key
```bash
cat ~/.ssh/id_rsa.pub
```

### Step 2: Add to GitHub
1. Go to: https://github.com/settings/keys
2. Click "New SSH key"
3. Title: "vinit_benchmark_server"
4. Paste your public key
5. Click "Add SSH key"

### Step 3: Push
```bash
git remote set-url origin git@github.com:vinitwadgaonkar/vis_ifeval.git
git push origin main
```

---

## Quick Push (After Token Setup)

Once you have a Personal Access Token:

```bash
cd /home/csgrad/vinitwad/vinit_benchmark
git push origin main
# Enter username: vinitwadgaonkar
# Enter password: <your_token>
```

Credentials will be saved automatically!

