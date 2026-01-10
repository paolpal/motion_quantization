## Diario di Sviluppo – Modulo di Movimento per Robot Umanoide

**Tesi:** Generazione di pose discrete tramite apprendimento latente da dataset gestuali

### 1. Obiettivi e Definizione del Problema

**Target:**
Sviluppo di un modulo di movimento per un robot umanoide focalizzato sulle articolazioni superiori: braccia, mani, testa e busto.

**Approccio:**
Implementazione di un controllo a stati discreti. Il robot dovrà operare su un insieme definito di “pose tipo”, passando in modo controllato da una posa all’altra.

**Dati:**

* Raccolta iniziale tramite scraping video e stima pose (OpenPose / MediaPipe).
* Successiva migrazione al dataset **PATS**, per garantire coerenza strutturale e maggiore volume di dati.

---

### 2. Analisi Esplorativa e Clustering Statistico

**Formato dei dati:** matrici $T \times J \times 2$ in coordinate cartesiane.

**Esperimenti iniziali – PCA:**

* Proiezione in 2D dei singoli frame ($J = 52$)
* Osservazione: evidente tendenza naturale alla clusterizzazione
* Risultato: possibilità di identificare i centroidi come “pose tipo”

**Evoluzione – UMAP + HDBSCAN:**

* Con l’aumento dei dati, la PCA non era più sufficiente a modellare la varianza non lineare
* UMAP ha migliorato la separazione dei cluster
* Tuttavia, la selezione manuale dei centroidi rimaneva poco flessibile e inadatta a un sistema generativo

**Normalizzazione:**
Introduzione di una normalizzazione spaziale basata sulla distanza delle spalle, per compensare variazioni di scala legate alla distanza del soggetto dalla camera.

---

### 3. Implementazione VQ-VAE (Vector Quantized Variational Autoencoder)

**Architettura:**
Autoencoder con collo di bottiglia discreto e singolo indice latente.

**Configurazione principale:**

* Codebook: 64 codici
* Dimensione latente ridotta, in linea con le osservazioni PCA / UMAP

**Risultati – fase cartesiana:**

* Perplexity ≈ $30 / 64$ (utilizzo effettivo di circa il 50% del codebook)
* Ricostruzione buona per le braccia
* Problema: le mani collassano verso pose medie; l’errore MSE delle dita è numericamente trascurabile rispetto alle braccia, portando alla loro sottorappresentazione

---

### 4. Cambio di Rappresentazione: Cinematica Diretta (FK)

**Motivazione:**
Passaggio da coordinate assolute a coordinate relative per riflettere meglio la struttura gerarchica dello scheletro.

**Nuovo formato dei dati:**
Ogni giunto è descritto come $(r,\ \sin\theta,\ \cos\theta)$.

**Vincoli geometrici:**

* Normalizzazione L2 per garantire:
  $$
  \sin^2\theta + \cos^2\theta = 1
  $$
* Attivazione `tanh` per i componenti angolari
* Layer lineare per il raggio $r$

**Risultati – fase polare:**

* Migliore rappresentazione delle mani
* Peggioramento della stabilità delle braccia, spesso “trascinate” dalla dinamica più complessa delle dita
* Presenza di artefatti, in particolare mani “a raggiera” dovute a medie angolari

---

### 5. Stato Attuale: Z-Normalization e Bias Locale

**Intervento effettuato:**
Applicazione della Z-Normalization sui dati cinematici $(r,\ \sin,\ \cos)$ prima dell’input al VQ-VAE.

**Osservazioni correnti:**

* Riduzione parziale degli artefatti “a raggiera”
* Persistenza di uno sbilanciamento: il modello concentra la capacità rappresentativa sulle mani (molto variabili e complesse) mantenendo le braccia in pose statiche di riposo

