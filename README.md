# reel-processor

CLI Python per processare in batch reel di esercizi di fisioterapia: trim video, mix audio (voiceover + musica di sottofondo) e aggiunta sottotitoli animati tramite l'API Mirage.

## Prerequisiti

- **Python 3.10+**
- **ffmpeg** e **ffprobe** (installabili via Homebrew su macOS):

```bash
brew install ffmpeg
```

- Account Mirage con API key attiva

## Installazione

```bash
cd reel-processor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Modifica config.yaml con i tuoi percorsi e credenziali
```

## Configurazione

Copia `config.yaml.example` in `config.yaml` e compila tutti i campi obbligatori.

| Campo | Descrizione |
|-------|-------------|
| `icloud_root` | Cartella radice del progetto con le sottocartelle numerate (`1/`, `2/`, …) |
| `output_dir` | Cartella di output per i video finali con sottotitoli |
| `background_music` | File audio della musica di sottofondo globale (es. `.mp3`, `.m4a`) |
| `music_volume_db` | Offset volume musica rispetto al voiceover (default: `-25`) |
| `voiceover_gain_db` | Gain del voiceover in dB (default: `0`) |
| `caption_template_id` | ID template Mirage per lo stile dei sottotitoli |
| `mirage_api_key` | Chiave API Mirage |
| `poll_interval_seconds` | Intervallo tra i poll di stato (default: `5`) |
| `max_poll_attempts` | Tentativi massimi di poll prima di fallire (default: `20`) |
| `video_trim_extra_seconds` | Secondi extra oltre la durata del voiceover (default: `1`) |
| `max_file_size_mb` | Soglia MB per warning dimensione file (default: `50`) |

### Ottenere la Mirage API key

1. Vai su [platform.mirage.app](https://platform.mirage.app)
2. Accedi o crea un account
3. Genera una API key dalla dashboard
4. Inseriscila in `mirage_api_key` nel file `config.yaml`

### Trovare il caption_template_id

Esegui:

```bash
python main.py list-templates
```

Copia l'ID del template desiderato (es. `ctpl_DxflLOnuKkb198FNdI9E`) in `caption_template_id`.

## Struttura cartelle di input

```
~/Desktop/progetto RP/
├── 1/
│   ├── squat.mov
│   └── squat.m4a
├── 2/
│   ├── lunge.mov
│   └── lunge.m4a
...
└── 344/
```

Ogni sottocartella numerata deve contenere **esattamente** un file `.mov` (video) e un file `.m4a` (voiceover) con lo stesso basename.

## iCloud e file non scaricati

Se i file sono su iCloud Drive, macOS può mantenerli solo nel cloud finché non vengono aperti. In quel caso troverai:

- File da **0 byte**
- File con estensione `.icloud` (es. `squat.mov.icloud`)

Il tool rileva questi casi e li salta con status `not_downloaded_from_icloud`. **Apri ogni cartella in Finder** e attendi il download completo prima di lanciare il batch.

## Comandi

### Scansione

Verifica tutte le cartelle e mostra una tabella con match ed errori:

```bash
python main.py scan
```

### Elaborazione completa

```bash
python main.py run
```

### Simulazione (senza FFmpeg né API)

```bash
python main.py run --dry-run
```

### Solo cartelle fallite o senza output

```bash
python main.py run --only-failed
```

### Singola cartella

```bash
python main.py run --folder 42
```

### Forza sovrascrittura output esistente

```bash
python main.py run --force
python main.py run --folder 42 --force
```

### Lista template Mirage

```bash
python main.py list-templates
```

## Output

I video finali vengono salvati in:

```
{output_dir}/{folder_number}_{basename}.mp4
```

Esempio: `42_squat.mp4`

Dopo ogni run viene creato/aggiornato `run_log.json` nella cartella di output con:

```json
{
  "folder_number": 42,
  "basename": "squat",
  "status": "success",
  "timestamp": "2026-06-11T10:30:00+00:00",
  "output_path": "/path/to/_output/42_squat.mp4"
}
```

Status possibili: `success`, `failed`, `skipped`.

## Pipeline per ogni cartella

1. **FFmpeg** — legge la durata del voiceover, taglia il video a `durata_voiceover + 1s`, mixa voiceover (principale) e musica di sottofondo (loop, -25 dB)
2. **Mirage API** — upload del video intermedio, poll fino a `COMPLETE`, download del video con sottotitoli
3. **Salvataggio** — file finale in `_output/`

Progresso in console:

```
[42/344] squat → TRIM ✓ | UPLOAD ✓ | PROCESSING... | COMPLETE ✓
```

## Gestione errori

| Caso | Comportamento |
|------|---------------|
| Coppia mov/m4a mancante | Skip (`missing_pair`) |
| Più file mov o m4a | Skip (`ambiguous_files`) |
| Basename diverso | Skip (`name_mismatch`) |
| File iCloud non scaricato | Skip (`not_downloaded_from_icloud`) |
| Output già esistente | Skip (usa `--force` per sovrascrivere) |
| Video più corto del target | Warning, continua |
| File > 50 MB | Warning, tenta upload comunque |
| Rate limit / errore 5xx Mirage | Retry con backoff esponenziale (max 3) |
| Job Mirage FAILED/CANCELLED | Log con codice e messaggio errore |
| API irraggiungibile | Messaggio chiaro, fail graceful |

## Licenza

Uso interno — progetto RP fisioterapia.
