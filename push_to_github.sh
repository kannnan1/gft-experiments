#!/bin/bash
# Script to push gft_experiments to GitHub
# 
# INSTRUCTIONS:
# 1. First, create a new repository on GitHub at https://github.com/new
#    - Name it: gft-experiments (or your preferred name)
#    - DO NOT initialize with README, .gitignore, or license
# 
# 2. Replace YOUR_USERNAME below with your GitHub username
# 3. Replace REPO_NAME if you used a different name
# 4. Run this script: bash push_to_github.sh

# Configuration
GITHUB_USERNAME="kannnan1"  # ← CHANGE THIS
REPO_NAME="gft-experiments"      # ← CHANGE THIS if different

# Add remote
echo "Adding GitHub remote..."
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
