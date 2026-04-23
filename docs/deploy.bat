@echo off
echo ==========================================
echo   Pronax AI Showcase - GitHub Pages Deploy
echo ==========================================
echo.

cd /d D:\ProNax-Ai

echo [1/4] Adding showcase files to git...
git add showcase/
git commit -m "Update Pronax AI showcase website"

echo.
echo [2/4] Pushing to main branch...
git push origin main

echo.
echo [3/4] Creating gh-pages branch...
git checkout --orphan gh-pages-temp
git rm -rf .

echo.
echo [4/4] Copying showcase files to root...
xcopy /E /I /Y showcase\* . >nul 2>&1
git add .
git commit -m "Deploy showcase to GitHub Pages"
git push origin gh-pages-temp:gh-pages --force

git checkout main
git branch -D gh-pages-temp

echo.
echo ==========================================
echo   DEPLOYMENT COMPLETE!
echo ==========================================
echo.
echo Your site will be live at:
echo https://zkg-786.github.io/pronax-ai/
echo.
echo (It may take 2-3 minutes to propagate)
echo.
pause
