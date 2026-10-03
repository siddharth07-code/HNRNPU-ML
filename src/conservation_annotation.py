import pandas as pd
import pyBigWig
from tqdm import tqdm

INPUT="data/general_missense.csv"
OUTPUT="data/conservation_scores.csv"
URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way.bw"
CHUNK=1_000_000

df=pd.read_csv(INPUT)

df=df[
    (df["Assembly"]=="GRCh37")&
    (df["Type"]=="single nucleotide variant")
].copy()

df["Start"]=df["Start"].astype(int)

print(f"GRCh37 SNVs: {len(df):,}")

bw=pyBigWig.open(URL)
df["PhyloP100way"]=float("nan")

for chrom,group in tqdm(df.groupby("Chromosome"),desc="Processing chromosomes"):
    chrom_name="chr"+str(chrom)

    if chrom_name not in bw.chroms():
        continue

    positions=group["Start"].values
    first_chunk=((positions.min()-1)//CHUNK)*CHUNK
    last_chunk=((positions.max()-1)//CHUNK)*CHUNK

    for chunk_start in range(first_chunk,last_chunk+CHUNK,CHUNK):
        chunk_end=min(chunk_start+CHUNK,bw.chroms()[chrom_name])

        mask=(group["Start"]-1>=chunk_start)&(group["Start"]-1<chunk_end)
        chunk_group=group[mask]

        if chunk_group.empty:
            continue

        values=bw.values(chrom_name,chunk_start,chunk_end)

        for idx,row in chunk_group.iterrows():
            offset=(int(row["Start"])-1)-chunk_start

            if 0<=offset<len(values):
                df.loc[idx,"PhyloP100way"]=values[offset]

bw.close()

result=df[
    ["#AlleleID","VariationID","GeneSymbol","ProteinChange",
     "Assembly","Chromosome","Start","Stop","PhyloP100way"]
]

result.to_csv(OUTPUT,index=False)

print("\nSaved:",OUTPUT)
print("Records:",len(result))
print("Scores found:",result["PhyloP100way"].notna().sum())
print("Scores missing:",result["PhyloP100way"].isna().sum())
print("\nConservation summary:")
print(result["PhyloP100way"].describe())