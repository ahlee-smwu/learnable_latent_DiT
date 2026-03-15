import torch
import re

with open('analysis.txt', 'r') as f:
    text = f.read()

pattern = re.compile(
    r'\[Cluster (\d+)\].*?'
    r'\[Real.*?\]\s+mean±std=([\d.]+)±([\d.]+)',
    re.DOTALL
)

shell_stats = {}
for m in pattern.finditer(text):
    cid = int(m.group(1))
    shell_stats[cid] = {
        'mean': float(m.group(2)),
        'std':  float(m.group(3))
    }

torch.save(shell_stats, 'cluster_shell.pt')

# 확인 출력
for k, v in sorted(shell_stats.items()):
    print(f"Cluster {k:02d}: μ_d={v['mean']:.4f}, σ_d={v['std']:.4f}")