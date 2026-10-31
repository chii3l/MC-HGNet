"""Download pinned upstream assets on explicit request, never on model import."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from .assets import asset,sha256

REPOS={
 'tslib':('thuml/Time-Series-Library','4e938a1767106324dd753b2a44832bf870a0252e'),
 'moderntcn':('luodhhh/ModernTCN','56a9a2c018385cd5acef015378cae7f084d1b11c'),
 'papagei':('Nokia-Bell-Labs/papagei-foundation-model','0c537dad4d2850e15b724260de820dd68d77f0b0'),
 'pulseppg':('maxxu05/pulseppg','716eaf9cf966e8f76436f2263872ef38b1f90166'),
 'anyppg':('Ngk03/AnyPPG','661b877aa96eac3bba320a462cb2a3bfea991103'),
 'sigmappg':('ZonghengGuo/SigmaPPG','b1cc8d90da3878958b890c6e352cd3ae2a644005'),
 'srformer':('chii3l/SRFormer','5fdca68d219e025c1630b43b83f5acce72b64944'),
}


def download(url,path,expected=None,algorithm='sha256'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    req=urllib.request.Request(url,headers={'User-Agent':'MC-HGNet-reproduction'})
    with urllib.request.urlopen(req,timeout=120) as src,temporary.open('wb') as dst:
        shutil.copyfileobj(src,dst,4*1024*1024)
    if expected:
        h=hashlib.new(algorithm)
        with temporary.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if h.hexdigest()!=expected:raise ValueError('Downloaded asset checksum mismatch: '+str(path))
    temporary.replace(path)


def extract(archive,dest,strip_root=True,code_only=True):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            parts=PurePosixPath(info.filename).parts
            if strip_root:parts=parts[1:]
            if not parts:continue
            if any(p in ('..','.') for p in parts):raise ValueError('Unsafe archive entry')
            target=dest.joinpath(*parts).resolve()
            if not target.is_relative_to(dest):raise ValueError('Unsafe archive path')
            if code_only and not (target.suffix in ('.py','.json','.md','.txt','.yaml','.yml') or
                                  target.name.upper().startswith(('LICENSE','COPYING','NOTICE'))):continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)


def code(name):
    if name=='papagei_pretrained':name='papagei'
    if name=='timefilter':name='tslib'
    if name=='moderntcn':code('tslib')
    repo,commit=REPOS[name];dest=asset(name)
    if (dest/'SOURCE.json').exists():
        if json.loads((dest/'SOURCE.json').read_text())['commit']!=commit:raise ValueError('Source version mismatch')
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'source.zip'
        download(f'https://codeload.github.com/{repo}/zip/{commit}',archive)
        extract(archive,dest)
    (dest/'SOURCE.json').write_text(json.dumps(dict(repository=repo,commit=commit),indent=2)+'\n')
    print('Source:',name,commit,flush=True)


def pretrained(name):
    if name=='qwen':
        subprocess.run(['hf','download','Qwen/Qwen3-Embedding-0.6B','--revision',
            '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3','--local-dir',str(asset('qwen')),
            '--include','*.json','*.txt','*.safetensors','README.md','LICENSE*','NOTICE*'],check=True)
    elif name=='sigmappg':
        subprocess.run(['hf','download','zonhengu/sigmappg','sigma.pth','--revision',
            '5aaf19ded8438b659593edb4edeeecf33e8d5d22','--local-dir',str(asset('sigma_weights'))],check=True)
    elif name=='papagei_pretrained':
        download('https://zenodo.org/records/13983110/files/papagei_s.pt?download=1',asset('papagei')/'papagei_s.pt',
                 'a4cdb32392e2a7b25999128af92813b5','md5')
    elif name=='anyppg':
        repo,commit=REPOS[name]
        download(f'https://raw.githubusercontent.com/{repo}/{commit}/load_anyppg/anyppg_ckpt.pth',
            asset(name)/'load_anyppg/anyppg_ckpt.pth','99b9bb0a3c2b83a1f5d8ca2963fbd25329b6530e8d337de8825722fc6fd5f4fa')
    elif name=='pulseppg':
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'weights.zip'
            download('https://zenodo.org/api/records/17345536/files/pulseppg_model_weights.zip/content',archive,
                '5685093114036e9ceb720b98342d68188c8786ed4e48466aa608a325cd890918')
            extract(archive,asset('pulse_weights'),strip_root=False,code_only=False)
    else:raise ValueError('This encoder is initialized from scratch: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['code','pretrained','crhe'])
    p.add_argument('--encoder',default='moderntcn')
    p.add_argument('--dest',type=Path)
    a=p.parse_args()
    if a.kind=='code':code(a.encoder)
    elif a.kind=='pretrained':pretrained(a.encoder)
    else:
        if not a.dest:p.error('--dest is required for CRHE')
        if a.dest.exists() and any(a.dest.iterdir()):raise FileExistsError('Use an empty CRHE destination')
        with tempfile.TemporaryDirectory() as tmp:
            z=Path(tmp)/'crhe.zip'
            download('https://codeload.github.com/FDU-Bio-Monitoring/CRHE-Long-Term-PPG-BP-Dataset/zip/298c6c8c50b0b04062b1c8776d18a61a4eb242e6',z)
            extract(z,a.dest,code_only=False)


if __name__=='__main__':main()
