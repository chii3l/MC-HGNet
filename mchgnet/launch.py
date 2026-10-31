"""Run folds on explicitly chosen GPUs, one experiment per device."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--encoders',nargs='+',default=['moderntcn'])
    p.add_argument('--variants',nargs='+',default=['baseline','full'])
    p.add_argument('--gpus',nargs='+',default=['0','1'])
    p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    a=p.parse_args();jobs=Queue()
    if len(set(a.gpus))!=len(a.gpus):p.error('GPU IDs must be distinct')
    for f in a.folds:
        for e in a.encoders:
            for v in a.variants:jobs.put((e,v,f))
    def worker(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
        while True:
            try:encoder,variant,fold=jobs.get_nowait()
            except Empty:return
            dest=a.out/encoder/variant/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
            commands=[]
            if not (dest/'complete.json').exists():
                commands.append([sys.executable,'-m','mchgnet.train','--data',str(a.data),'--encoder',encoder,
                    '--variant',variant,'--fold',str(fold),'--out',str(dest),'--device','cuda:0'])
            commands.append([sys.executable,'-m','mchgnet.evaluate','--data',str(a.data),'--run',str(dest),'--device','cuda:0'])
            with (dest/'console.log').open('a',encoding='utf-8') as log:
                for cmd in commands:
                    print('GPU',gpu,subprocess.list2cmdline(cmd),flush=True)
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,g) for g in a.gpus]
        for future in futures:future.result()


if __name__=='__main__':main()
