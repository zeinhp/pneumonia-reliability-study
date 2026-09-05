# MDES / Power Analysis A Priori - Eksperimen A (Common Anchor Test)

Dibekukan: 2026-07-24, sebelum Fase 1 (manifest terkontrol) dimulai.
Status: pre-registered - dihitung sebelum melihat data prediksi model apapun.

## 1. Tujuan

Menutup ambiguitas interpretasi hasil "tidak signifikan": apakah karena efek leakage
memang kecil, atau karena test set (n=279 pasien) kurang bertenaga (underpowered)?
Dokumen ini membekukan Minimum Detectable Effect Size (MDES) sebelum training, supaya
hasil null bisa diberi makna kuantitatif ("mampu mendeteksi delta >= X pada power 80%"),
bukan sekadar "tidak signifikan".

## 2. Setup

- Anchor test set (bagian 6.2 protokol): 279 pasien, 1 citra/pasien.
  n_pos (PNEUMONIA) = 169, n_neg (NORMAL) = 110.
- Perbandingan utama: ΔAUROC = AUROC_leaky - AUROC_clean, per arsitektur, desain
  **berpasangan** (test set identik untuk clean & leaky pada arsitektur & seed yang sama).
- Family koreksi (bagian 16.4 protokol): 3 perbandingan clean-vs-leaky (satu per arsitektur)
  dikoreksi Holm-Bonferroni -> alpha per-perbandingan konservatif dipakai di sini =
  0,05 / 3 = 0,01667 (two-sided), z_crit = 2,394.

## 3. Metode

SE(AUROC) dihitung analitik dengan formula Hanley-McNeil:

```
Q1 = AUC / (2 - AUC)
Q2 = 2*AUC^2 / (1 + AUC)
Var(AUC) = [AUC*(1-AUC) + (n_pos-1)*(Q1-AUC^2) + (n_neg-1)*(Q2-AUC^2)] / (n_pos*n_neg)
```

Untuk desain berpasangan (clean & leaky dievaluasi pada pasien yang sama), varians selisih:

```
SE(delta) = sqrt(SE1^2 + SE2^2 - 2*rho*SE1*SE2)
```

`rho` adalah korelasi antara skor prediksi kondisi clean dan leaky pada pasien yang sama.
**Tidak ada data pilot untuk mengestimasi rho secara empiris sebelum training**, jadi rho
diperlakukan sebagai parameter sensitivitas (bukan diasumsikan satu nilai) - divariasikan
0 (unpaired, batas bawah konservatif) sampai 0,95 (korelasi sangat tinggi, realistis untuk
dua model dengan arsitektur & sebagian besar data training identik).

Power dihitung dari z = delta / SE(delta), lalu:

```
power = 1 - Phi(z_crit - z) + Phi(-z_crit - z)
```

MDES = delta minimum sehingga power >= 80%, dicari lewat bisection.

Kode lengkap (dijalankan langsung, tidak disimpan sebagai script terpisah karena
sifatnya one-off pre-registration, bukan bagian pipeline manifest):

```python
import math
from scipy.stats import norm

n_pos, n_neg = 169, 110
alpha_family = 0.05 / 3
z_crit = norm.ppf(1 - alpha_family / 2)

def se_auroc(auc, n_pos, n_neg):
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    var = (auc*(1-auc) + (n_pos-1)*(Q1-auc**2) + (n_neg-1)*(Q2-auc**2)) / (n_pos*n_neg)
    return math.sqrt(var)

def power_paired(auc_clean, delta, rho, n_pos, n_neg, z_crit):
    auc_leaky = min(auc_clean + delta, 0.999)
    se1 = se_auroc(auc_clean, n_pos, n_neg)
    se2 = se_auroc(auc_leaky, n_pos, n_neg)
    se_diff = math.sqrt(max(se1**2 + se2**2 - 2*rho*se1*se2, 1e-10))
    z = delta / se_diff
    return 1 - norm.cdf(z_crit - z) + norm.cdf(-z_crit - z)

def mdes(auc_clean, rho, n_pos, n_neg, z_crit, target_power=0.80):
    lo, hi = 0.0001, 0.30
    for _ in range(60):
        mid = (lo + hi) / 2
        p = power_paired(auc_clean, mid, rho, n_pos, n_neg, z_crit)
        lo, hi = (mid, hi) if p < target_power else (lo, mid)
    return hi
```

## 4. Hasil - MDES (delta AUROC, power 80%)

| AUROC_clean | rho=0,00 | rho=0,30 | rho=0,50 | rho=0,70 | rho=0,85 | rho=0,95 |
|---|---|---|---|---|---|---|
| 0,85 | 0,0860 | 0,0748 | 0,0651 | 0,0522 | 0,0381 | 0,0227 |
| 0,90 | 0,0672 | 0,0591 | 0,0519 | 0,0420 | 0,0309 | 0,0186 |
| 0,93 | 0,0537 | 0,0479 | 0,0425 | 0,0347 | 0,0258 | 0,0156 |
| 0,95 | 0,0433 | 0,0394 | 0,0353 | 0,0292 | 0,0219 | 0,0134 |

## 5. Hasil - Power aktual pada beberapa delta realistis (AUROC_clean=0,90)

| delta | rho=0,00 | rho=0,30 | rho=0,50 | rho=0,70 | rho=0,85 | rho=0,95 |
|---|---|---|---|---|---|---|
| 0,01 | 0,03 | 0,03 | 0,04 | 0,05 | 0,09 | 0,27 |
| 0,02 | 0,06 | 0,08 | 0,11 | 0,18 | 0,38 | 0,86 |
| 0,03 | 0,13 | 0,19 | 0,27 | 0,45 | 0,77 | 0,99 |
| 0,04 | 0,26 | 0,38 | 0,52 | 0,75 | 0,96 | 1,00 |
| 0,05 | 0,45 | 0,61 | 0,76 | 0,93 | 1,00 | 1,00 |
| 0,06 | 0,66 | 0,82 | 0,92 | 0,99 | 1,00 | 1,00 |
| 0,08 | 0,95 | 0,99 | 1,00 | 1,00 | 1,00 | 1,00 |

## 6. Hasil - McNemar (analisis diskrit sekunder, threshold 0,5)

Power tergantung m (jumlah pasangan prediksi diskordan antara clean & leaky) dan p
(proporsi pasangan diskordan yang berpihak ke leaky, di bawah null p=0,5).

| m (diskordan) | p=0,60 | p=0,65 | p=0,70 | p=0,75 | p=0,80 |
|---|---|---|---|---|---|
| 10 | 0,04 | 0,07 | 0,11 | 0,17 | 0,27 |
| 15 | 0,05 | 0,10 | 0,18 | 0,30 | 0,47 |
| 20 | 0,06 | 0,14 | 0,25 | 0,43 | 0,64 |
| 30 | 0,09 | 0,22 | 0,41 | 0,65 | 0,87 |
| 40 | 0,12 | 0,30 | 0,56 | 0,81 | 0,96 |
| 60 | 0,19 | 0,47 | 0,78 | 0,96 | 1,00 |
| 80 | 0,27 | 0,62 | 0,90 | 0,99 | 1,00 |
| 279 (maksimum teoretis) | 0,83 | 1,00 | 1,00 | 1,00 | 1,00 |

## 7. Interpretasi & keputusan yang dibekukan

1. Dengan n=279 dan tanpa data pilot, **MDES realistis ada di kisaran 0,02-0,05 AUROC**,
   tergantung seberapa tinggi korelasi antara prediksi clean/leaky pada pasien yang sama.
   Korelasi tinggi (rho>=0,7) - yang masuk akal karena kedua model berbagi arsitektur &
   >99% data training - membuat desain ini cukup bertenaga untuk efek moderat (delta>=0,03).
   Korelasi rendah/tidak diketahui membuat efek kecil (delta<0,02) sulit dideteksi.
2. McNemar (endpoint sekunder) butuh cukup banyak pasangan diskordan (m>=40) dan asimetri
   jelas (p>=0,75) untuk power memadai - realistis hanya kalau leakage effect-nya cukup
   besar. Dengan m kecil, McNemar TIDAK BOLEH dijadikan bukti tunggal ketiadaan efek.
3. **Headline pelaporan = effect size (ΔAUROC) + CI, bukan p-value.** Hasil null wajib
   dilaporkan sebagai "efek di bawah ambang deteksi desain ini (~0,02-0,05 tergantung rho)",
   bukan "tidak ada leakage".
4. Rho empiris (bukan asumsi) akan dihitung dari data prediksi aktual begitu Fase 4-5
   selesai, dan dilaporkan sebagai bagian dari analisis realized-power di bagian metode
   hasil akhir - dibandingkan terhadap tabel ini.
5. Angka-angka di dokumen ini **dibekukan** - tidak boleh dihitung ulang atau diubah untuk
   menyesuaikan hasil setelah test set dibuka (melanggar bagian 24 protokol, amendment
   setelah unblinding dilarang untuk confirmatory analysis).
