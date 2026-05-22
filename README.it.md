# LocalForge — Guida in Italiano

**LocalForge** è uno strumento di generazione di codice e intelligenza di progetto alimentato da AI locale. Funziona come finestra desktop nativa (tramite pywebview) supportata da un server Flask locale sulla porta 7331. Tutta l'inferenza LLM utilizza **Ollama** in esecuzione locale — nessuna API cloud.

---

## Modalità disponibili

| Modalità | Descrizione |
|----------|-------------|
| **Generate** | Crea nuovi progetti da zero con pianificazione guidata dall'AI |
| **Resume** | Continua o estendi qualsiasi progetto (generato da LocalForge o esterno) |
| **Explain** | Analisi approfondita di qualsiasi codebase, report di intelligence in 11 sezioni, Q&A interattivo |

---

## Requisiti

- Python 3.11+
- [Ollama](https://ollama.ai/) installato e in esecuzione localmente
- Almeno un modello Ollama scaricato (es. `llama3`, `mistral`, `qwen2.5-coder`)
- Per la ricerca semantica RAG: `nomic-embed-text`

---

## Installazione e avvio

```bash
# Clona il repository
git clone https://github.com/tuouser/localforge_web.git
cd localforge_web

# Installa le dipendenze Python
pip install -r requirements.txt

# (Opzionale) Scarica il modello di embedding per la ricerca RAG
ollama pull nomic-embed-text:latest

# Avvia l'applicazione
python main.py
```

L'applicazione si aprirà come finestra nativa su `http://127.0.0.1:7331`.

---

## Configurazione lingua

LocalForge supporta tre lingue:

| Codice | Lingua |
|--------|--------|
| `en` | English |
| `ja` | 日本語 |
| `it` | Italiano |

**Come selezionare la lingua:**

1. Usa il selettore lingua nell'intestazione dell'applicazione (accanto al selettore modello)
2. La preferenza viene salvata nel file `.localforge/config.json` del progetto
3. La lingua selezionata influenza:
   - Tutta l'interfaccia utente
   - I messaggi di stato ed errore
   - Le risposte dell'AI (prompt, report, Q&A)
   - Il contenuto generato (commenti nel codice, docstring, messaggi)

---

## Utilizzo delle modalità

### Generate — Crea un nuovo progetto

1. Apri una cartella vuota con **📁 Apri cartella**
2. Inserisci una descrizione del progetto nel campo testo
3. Clicca **✦ Genera piano** per creare il piano di progetto
4. Rivedi i file pianificati e clicca **✓ Approva e avvia generazione**
5. Attendi il completamento della generazione

**Suggerimenti:**
- Puoi specificare il numero minimo/massimo di file da generare
- Usa "Genera su nuovo branch" per mantenere il codice separato
- Clicca ✎ **Modifica** per modificare il piano JSON direttamente

### Resume — Continua un progetto

1. Apri una cartella con un progetto esistente
2. Per progetti LocalForge: usa **▶ Continua generazione** o **✎ Modifica piano**
3. Per progetti esterni: usa **📄 Report completo** o **💬 Continua Q&A**

### Explain — Analisi della codebase

1. Apri una cartella del progetto
2. Clicca **⚙ Costruisci indice** per indicizzare tutti i file
3. Clicca **✦ Genera report** per creare il report di analisi in 11 sezioni
4. Usa la chat **Q&A** in basso per fare domande sul progetto

**Sezioni del report:**
- Panoramica del progetto
- Architettura del sistema
- Struttura della codebase
- Dipendenze e librerie
- Flussi di dati
- Pattern di design
- Punti di sicurezza
- Qualità del codice
- Test e copertura
- Deployment e configurazione
- Raccomandazioni di miglioramento

---

## Ricerca semantica RAG

LocalForge utilizza ChromaDB + `nomic-embed-text` per la ricerca semantica:

```bash
# Scarica il modello di embedding
ollama pull nomic-embed-text:latest
```

**Come funziona:**
- Durante l'indicizzazione, i file vengono incorporati automaticamente
- La ricerca semantica migliora la pertinenza delle risposte Q&A
- Se `nomic-embed-text` non è disponibile, viene utilizzata la ricerca BM25 come fallback
- Per progetti indicizzati prima dell'aggiunta del RAG: usa il pulsante **⬡ Migrazione RAG**

---

## Gestione dei modelli

- **Selezione modello**: usa il menu a tendina nell'intestazione
- **Cambio modello**: il modello precedente viene scaricato dalla VRAM automaticamente
- **Scarica modello**: pulsante **⏏ Scarica** per liberare VRAM
- **Thread CPU**: regola il numero di thread CPU per l'inferenza nel pannello destro

---

## Struttura del progetto LocalForge

Ogni progetto gestito crea una directory `.localforge/`:

```
.localforge/
├── config.json          # Configurazione (modello, lingua, token limit)
├── plan.json            # Piano di generazione corrente
├── index.jsonl          # Indice dei file (JSONL incrementale)
├── project_index.json   # Sommario del progetto
├── report.md            # Report di analisi salvato
├── qa_history.md        # Cronologia Q&A
├── context.md           # Memoria del progetto
├── generation_log.jsonl # Log delle chiamate LLM
├── chroma/              # Database vettoriale ChromaDB
└── cache/               # Cache risposte e ricerca semantica
    ├── responses/
    └── semantic/
```

---

## Sicurezza

LocalForge è progettato come strumento **completamente locale e offline**:

- Tutto il traffico LLM va a `http://localhost:11434` (Ollama locale)
- Il server Flask è in ascolto su `127.0.0.1:7331` (solo loopback)
- Nessuna dipendenza da CDN esterni in JavaScript
- ChromaDB con telemetria anonima disabilitata
- Protezione da path traversal su tutti gli endpoint file
- Nessuna chiamata `subprocess` con `shell=True`
- Output Markdown sanificato con DOMPurify

**Variabili d'ambiente che modificano la sicurezza:**

| Variabile | Predefinito | Effetto se modificato |
|-----------|-------------|----------------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Reindirizza le chiamate LLM all'host specificato |
| `FLASK_HOST` | `127.0.0.1` | Se impostato a `0.0.0.0`, l'API è raggiungibile dalla rete |

⚠️ **Non impostare `OLLAMA_HOST` a un URL esterno** — il contenuto dei file e i prompt vengono inviati lì.

---

## Risoluzione problemi

| Problema | Soluzione |
|----------|-----------|
| Ollama non disponibile | Assicurati che Ollama sia in esecuzione: `ollama serve` |
| Nessun modello disponibile | Scarica un modello: `ollama pull llama3` |
| Embedding lento | Normale al primo avvio; `nomic-embed-text` viene caricato nella RAM |
| ChromaDB errori | Elimina `.localforge/chroma/` e riesegui l'indicizzazione |
| Cache obsoleta | Elimina `.localforge/cache/` o riesegui `build_index` |
| UI bloccata | Clicca **⏹ Ferma** nell'intestazione |
| Timeout connessione | Riavvia LocalForge; Ollama potrebbe essere sovraccarico |
| Prima risposta lenta | Normale — il modello viene caricato nella RAM; le risposte successive saranno più veloci |

---

## Sviluppo e contributi

```bash
# Esegui i test
pytest tests/

# Struttura principale
localforge/
├── domain/          # Modelli Pydantic, interfacce port, eccezioni
├── application/     # Logica di business (senza I/O né HTTP)
├── infrastructure/  # Adattatori (Ollama, filesystem, git, ChromaDB)
└── interface/       # Route Flask, template, asset statici
```

I test utilizzano adattatori mock — Ollama reale e filesystem non sono richiesti.

---

## Licenza

Vedere `LICENSE` nella root del repository.

---

*LocalForge — Intelligenza artificiale locale per sviluppatori.*
