# RAM++ on Windows

This setup installs the same RAM++ Swin-L 384 checkpoint and English tag-score runner used for the local ModelShield demo. It uses CPU PyTorch and downloads the approximately 3 GB checkpoint from the publisher. The installer checks the checkpoint and vocabulary against their SHA-256 hashes.

Install Git for Windows and Python 3.11 with the Python launcher. Then run these commands in PowerShell from the directory where you want the clone.

```powershell
git clone --branch 'RAM++' https://github.com/manik0303-glitch/ModelShield.git
cd ModelShield\ram-plus
.\install.ps1
.\.venv\Scripts\python.exe classify.py "C:\path\to\your\photo.jpg"
```

Replace the quoted path with your own photo. RAM++ returns several tags per photo with individual model scores. It runs locally after installation. See the [upstream project](https://github.com/xinyu1205/recognize-anything) for the original implementation and licensing.
