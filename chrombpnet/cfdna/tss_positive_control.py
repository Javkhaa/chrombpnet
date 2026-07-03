"""TSS positive control: composite cfDNA coverage / dyad-midpoint / WPS around
protein-coding TSS (strand-oriented), GC-corrected. Canonical Ulz/Snyder signal =
central NDR dip + phased flanking nucleosomes. No ATAC reference used."""
import numpy as np, pyarrow.parquet as pq, pyfaidx, sys

CFDNA = "/mnt/data/jganbat/cfdna_test/NC-PKU-mix15-wgs_min_mapq_50.fragments.parquet"
TSS   = "/mnt/data/jganbat/cfdna_test/tss_pc.tsv"
HG38  = "/mnt/data/jganbat/scatac_corpus/refs/hg38.fa"
W = 1500; BIN = 10; WPS_K = 120
MAIN = set(f"chr{c}" for c in list(range(1,23))+["X","Y"])

print("loading fragments + per-fragment GC ...", flush=True)
t = pq.read_table(CFDNA, columns=["chromosome","start_position","end_position","sequence"])
ch = t.column("chromosome").to_numpy(zero_copy_only=False)
s = t.column("start_position").to_numpy().astype(np.int64)
e = t.column("end_position").to_numpy().astype(np.int64)
seq = t.column("sequence").combine_chunks(); bufs = seq.buffers()
offs = np.frombuffer(bufs[1], dtype=np.int32); vals = np.frombuffer(bufs[2], dtype=np.uint8)
n = len(ch); st_, en_ = offs[:n], offs[1:n+1]
isgc = ((vals==71)|(vals==67)|(vals==103)|(vals==99)).astype(np.int64)
csum = np.concatenate([[0], np.cumsum(isgc)])
gcf = np.where((en_-st_)>0, (csum[en_]-csum[st_])/np.maximum(en_-st_,1), 0.5)
L = e - s; keep = (L>=100)&(L<=250)
ch, s, e, gcf = ch[keep], s[keep], e[keep], gcf[keep]; Lk = L[keep]
print(f"  mono-nuc fragments: {len(s):,}", flush=True)

print("genome-expected GC (matched lengths) ...", flush=True)
fa = pyfaidx.Fasta(HG38); rng = np.random.default_rng(0); N=150000
CH3 = ["chr1","chr3","chr6"]; sizes={c:len(fa[c]) for c in CH3}
Ls = Lk[rng.integers(0,len(Lk),N)]; egc=[]
for i in range(N):
    c=CH3[rng.integers(0,3)]; ln=int(Ls[i]); p=int(rng.integers(3_000_000, sizes[c]-3_000_000))
    ss=str(fa[c][p:p+ln]).upper()
    if 'N' in ss: continue
    egc.append((ss.count('G')+ss.count('C'))/ln)
egc=np.array(egc)
bins=np.linspace(0,1,51); oh,_=np.histogram(gcf,bins=bins,density=True); eh,_=np.histogram(egc,bins=bins,density=True)
wc=np.where(oh>1e-6, eh/oh, 0.0); fw=wc[np.clip(np.digitize(gcf,bins)-1,0,len(wc)-1)]; fw/= (fw.mean()+1e-9)
print(f"  GC weight range {fw.min():.2f}..{fw.max():.2f}", flush=True)

# index fragments per chrom, sorted by start
frby={}
for c in MAIN:
    m=ch==c
    if m.any():
        o=np.argsort(s[m]); frby[c]=(s[m][o], e[m][o], fw[m][o])

# load TSS
tss={}
for line in open(TSS):
    c,pos,strand=line.split(); pos=int(pos)
    if c in MAIN: tss.setdefault(c,[]).append((pos,strand))
ntss=sum(len(v) for v in tss.values())
print(f"  TSS: {ntss}", flush=True)

nb=2*W//BIN
cov=np.zeros(nb); mid=np.zeros(nb); wps=np.zeros(nb); used=0
for c,lst in tss.items():
    if c not in frby: continue
    st,en,fwt=frby[c]
    for pos,strand in lst:
        lo,hi=pos-W,pos+W
        j0=np.searchsorted(st,lo-260); j1=np.searchsorted(st,hi)
        cs=st[j0:j1]; ce=en[j0:j1]; cw=fwt[j0:j1]
        sel=ce>lo
        cs,ce,cw=cs[sel],ce[sel],cw[sel]
        if len(cs)==0: used+=1; continue
        # local diff arrays (absolute orientation), length nb
        dcov=np.zeros(nb+1); dmid_idx=[]; dspan=np.zeros(nb+1); dend=np.zeros(nb+1)
        # coverage: fragment covers [cs,ce)
        a=np.clip(((np.maximum(cs,lo)-lo)//BIN).astype(int),0,nb); b=np.clip(((np.minimum(ce,hi)-lo)//BIN).astype(int),0,nb)
        np.add.at(dcov,a,cw); np.add.at(dcov,b,-cw)
        covl=np.cumsum(dcov[:nb])
        # midpoint density
        mids=((cs+ce)//2); mrel=((mids-lo)//BIN).astype(int); msel=(mrel>=0)&(mrel<nb)
        midl=np.zeros(nb); np.add.at(midl,mrel[msel],cw[msel])
        # WPS: spanning positions [cs+K/2, ce-K/2]; endpoints within K/2 of cs or ce
        h=WPS_K//2
        sa=np.clip(((cs+h-lo)//BIN).astype(int),0,nb); sb=np.clip(((ce-h-lo)//BIN).astype(int),0,nb)
        good=sb>sa
        np.add.at(dspan,sa[good],cw[good]); np.add.at(dspan,sb[good],-cw[good])
        for ep in (cs,ce):
            ea=np.clip(((ep-h-lo)//BIN).astype(int),0,nb); eb=np.clip(((ep+h-lo)//BIN).astype(int),0,nb)
            np.add.at(dend,ea,cw); np.add.at(dend,eb,-cw)
        wpsl=np.cumsum(dspan[:nb])-np.cumsum(dend[:nb])
        # orient by strand (minus -> mirror)
        if strand=='-':
            covl=covl[::-1]; midl=midl[::-1]; wpsl=wpsl[::-1]
        cov+=covl; mid+=midl; wps+=wpsl; used+=1
cov/=used; mid/=used; wps/=used
print(f"  aggregated over {used} TSS", flush=True)

def spark(a):
    d=a.reshape(-1,10).mean(1); d=(d-d.min())/(d.max()-d.min()+1e-9); bars=" .:-=+*#%@"
    return "".join(bars[min(9,int(v*9))] for v in d)
def dipdepth(a):  # center (NDR, just downstream) vs flanks
    c=a[nb//2-5:nb//2+15].mean(); fl=np.r_[a[nb//2-100:nb//2-50],a[nb//2+50:nb//2+100]].mean()
    return 1-c/(fl+1e-9)
print("\n=== composites around TSS (-1.5kb .. +1.5kb, 150bp bins; oriented 5'->3') ===")
for name,a in [("coverage(GCcorr)",cov),("dyad-midpoint",mid),("WPS",wps)]:
    print(f"{name:>18} dip={dipdepth(a):+.3f}  {spark(a)}")
np.savez("/mnt/data/jganbat/cfdna_test/tss_control.npz", cov=cov, mid=mid, wps=wps, used=used, W=W, BIN=BIN)
print("\nsaved tss_control.npz")
