CUDA_VISIBLE_DEVICES="MIG-GPU-6ff250df-07f5-cf8e-bfdb-d56c3c464126/2/0" python semrec.py --data GO/1_1 &
CUDA_VISIBLE_DEVICES="MIG-GPU-6ff250df-07f5-cf8e-bfdb-d56c3c464126/3/0" python semrec.py --data GO/1_n &
CUDA_VISIBLE_DEVICES="MIG-GPU-6ff250df-07f5-cf8e-bfdb-d56c3c464126/2/0" python semrec.py --data GO/n_n &