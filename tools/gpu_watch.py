"""gpu_watch.py (9 Nov): every 120 s print 'HH:MM:SS dedicated_MiB <n> shared_MB <m>' for the band0813 fit_spiral process
(m = -1 when none runs). Same output as the PowerShell gpu_watch of job_patches_ab_arm.sh, read with nvidia-smi and
typeperf instead (Windows PowerShell hangs at startup on this PC; PREREG_selfpatch.md amendment 1)."""
import csv, io, subprocess, sys, time
import psutil
while True:
    ded = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0]
    pid = None
    for p in psutil.process_iter(['name', 'cmdline']):
        c = ' '.join(p.info['cmdline'] or [])
        if (p.info['name'] or '').lower() == 'python.exe' and 'fit_spiral' in c:
            try:
                if 'band0813' in ' '.join(p.cmdline()) + p.cwd(): pid = p.pid; break
            except psutil.Error: pass
    sh = -1
    if pid:
        out = subprocess.run(['typeperf', r'\GPU Process Memory(*)\Shared Usage', '-sc', '1'], capture_output=True, text=True).stdout
        rows = [r for r in csv.reader(io.StringIO(out)) if r]
        if len(rows) >= 2:
            sh = int(sum(float(v) for h, v in zip(rows[0][1:], rows[1][1:]) if f'pid_{pid}_' in h and v.strip()) / 2**20)
    print(f"{time.strftime('%H:%M:%S')} dedicated_MiB {ded} shared_MB {sh}", flush=True)
    time.sleep(120)
