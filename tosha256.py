import hashlib
import os
filepath='gym_malware/envs/utils/samples'
files = os.listdir(filepath)
count=1
for fi in files:
    count+=1
    fi_d = os.path.join(filepath,fi)
    if os.path.isdir(fi_d) is False:
        os.rename(fi_d, os.path.join(filepath, hashlib.sha256(str(count).encode("utf8")).hexdigest()))

