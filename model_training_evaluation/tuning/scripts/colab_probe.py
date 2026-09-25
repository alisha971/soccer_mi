"""Colab plumbing probe: no repo data needed. Prints env for tuning runs."""
import sys
print("python=" + sys.version.split()[0])
try:
    import torch
    print("torch=" + torch.__version__ + " cuda=" + str(torch.cuda.is_available()))
except Exception as e:
    print("torch-missing " + str(e))
