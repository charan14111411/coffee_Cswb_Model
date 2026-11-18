# Synopsis PDF source

Source for `Varsapradaya_CWSB_Synopsis.pdf` (repo root), the management synopsis of the CWSB project.

To regenerate after editing `synopsis.html` (Windows, Git Bash):

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=20000 \
  --print-to-pdf="$(pwd)/../../Varsapradaya_CWSB_Synopsis.pdf" "file:///$(pwd)/synopsis.html"
```

- Fonts (Lexend Deca, Lora; SIL Open Font License) are bundled in `fonts/` because headless Chrome
  prints before Google Fonts finish loading.
- Brand colours and logo come from the varsapradaya-web project (`--green #004225`, `--black #003153`).
- Example images are crops of the Grad-CAM pages in `reports/cwsb_test_report_test_real.pdf`
  (t2_012, t2_078) and `reports/predictions_test_new.pdf` (543.jpg). `hero.jpg` is
  `Data_final/Unhealthy(Infected)/FIELD_IMG_1670.jpg`, resized, metadata stripped.
- **Audience: officials.** At the user's request (2026-11-18) this version contains no dataset
  counts, no test results, no limitations table and no technical appendix, and no
  "Prepared by / Project synopsis / prototype / Kodagu & Chikkamagaluru" labels. Keep it that way
  when editing. The earlier 10-page internal version (with results) is in git history at `feee910`.
