"""Matched random-position control for the TSS composite (coverage, GC-corrected)."""
import numpy as np, pyarrow.parquet as pq, pyfaidx
CFDNA="/mnt/data/jganbat/cfdna_test/NC-PKU-mix15-wgs_min_mapq_50.fragments.parquet"
TSS="/mnt/data/jganbat/cfdna_test/tss_pc.tsv"; HG38="/mnt/data/jganbat/scatac_corpus/refs/hg38.fa"
W=1500;BIN=10; MAIN=set(f"chr{c}" for c in list(range(1,23))+["X","Y"]); nb=2*W//BIN
t=pq.read_table(CFDNA,columns=["chromosome","start_position","end_position","sequence"])
ch=t.column("chromosome").to_numpy(zero_copy_only=False)
s=t.column("start_position").to_numpy().astype(np.int64); e=t.column("end_position").to_numpy().astype(np.int64)
seq=t.column("sequence").combine_chunks(); bufs=seq.buffers()
offs=np.frombuffer(bufs[1],dtype=np.int32); vals=np.frombuffer(bufs[2],dtype=np.uint8)
n=len(ch); st_,en_=offs[:n],offs[1:n+1]
isgc=((vals==71)|(vals==67)|(vals==103)|(vals==99)).astype(np.int64); cs_=np.concatenate([[0],np.cumsum(isgc)])
gcf=np.where((en_-st_)>0,(cs_[en_]-cs_[st_])/np.maximum(en_-st_,1),0.5)
L=e-s;keep=(L>=100)&(L<=250); ch,s,e,gcf=ch[keep],s[keep],e[keep],gcf[keep];Lk=L[keep]
fa=pyfaidx.Fasta(HG38);rng=np.random.default_rng(0);N=150000;CH3=["chr1","chr3","chr6"];sz={c:len(fa[c]) for c in CH3}
Ls=Lk[rng.integers(0,len(Lk),N)];egc=[]
for i in range(N):
    c=CH3[rng.integers(0,3)];ln=int(Ls[i]);p=int(rng.integers(3_000_000,sz[c]-3_000_000));ss=str(fa[c][p:p+ln]).upper()
    if 'N' not in ss: egc.append((ss.count('G')+ss.count('C'))/ln)
egc=np.array(egc);bins=np.linspace(0,1,51)
oh,_=np.histogram(gcf,bins=bins,density=True);eh,_=np.histogram(egc,bins=bins,density=True)
wc=np.where(oh>1e-6,eh/oh,0.0);fw=wc[np.clip(np.digitize(gcf,bins)-1,0,len(wc)-1)];fw/=(fw.mean()+1e-9)
frby={}
for c in MAIN:
    m=ch==c
    if m.any(): o=np.argsort(s[m]);frby[c]=(s[m][o],e[m][o],fw[m][o])
# per-chrom TSS counts to match
cnt={}
for line in open(TSS):
    c,pos,strand=line.split()
    if c in MAIN: cnt[c]=cnt.get(c,0)+1
def cov_composite(sites):
    cov=np.zeros(nb);used=0
    for c,lst in sites.items():
        if c not in frby: continue
        st,en,fwt=frby[c]
        for pos,strand in lst:
            lo,hi=pos-W,pos+W;j0=np.searchsorted(st,lo-260);j1=np.searchsorted(st,hi)
            cc=st[j0:j1];ce=en[j0:j1];cw=fwt[j0:j1];sel=ce>lo;cc,ce,cw=cc[sel],ce[sel],cw[sel]
            if len(cc)==0: used+=1;continue
            d=np.zeros(nb+1)
            a=np.clip(((np.maximum(cc,lo)-lo)//BIN).astype(int),0,nb);b=np.clip(((np.minimum(ce,hi)-lo)//BIN).astype(int),0,nb)
            np.add.at(d,a,cw);np.add.at(d,b,-cw);cl=np.cumsum(d[:nb])
            if strand=='-': cl=cl[::-1]
            cov+=cl;used+=1
    return cov/used,used
rng2=np.random.default_rng(123)
rsites={c:[(int(rng2.integers(3_000_000,len(fa[c])-3_000_000)),'+' if rng2.random()<.5 else '-') for _ in range(k)] for c,k in cnt.items() if c in frby}
cov,used=cov_composite(rsites)
def spark(a):
    d=a.reshape(-1,10).mean(1);d=(d-d.min())/(d.max()-d.min()+1e-9);bars=" .:-=+*#%@";return "".join(bars[min(9,int(v*9))] for v in d)
c=cov[nb//2-5:nb//2+15].mean();fl=np.r_[cov[nb//2-100:nb//2-50],cov[nb//2+50:nb//2+100]].mean()
print(f"RANDOM control coverage(GCcorr): used={used} dip={1-c/(fl+1e-9):+.3f}")
print("  "+spark(cov))
