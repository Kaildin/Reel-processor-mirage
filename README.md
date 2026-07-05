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
| `max_file_size_mb` | Soglia MB per warning upload Mirage (default: `50`) |
| `output_width` | Larghezza export finale in pixel (default: `2160` = 4K verticale) |
| `output_height` | Altezza export finale in pixel (default: `3840` = 4K verticale 9:16) |
| `enable_hdr` | Export finale in HLG HDR HEVC 10-bit (default: `true`) |
| `video_crf` | Qualità export finale — più basso = migliore (default: `20`) |
| `mirage_upload_crf` | Qualità file inviato a Mirage (default: `24`) |

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

Ogni sottocartella numerata deve contenere **esattamente** un file `.mov` (video) e un file `.m4a` (voiceover). I nomi non devono coincidere: se c'è una sola coppia nella cartella, viene accettata automaticamente.

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
{output_dir}/captions_{nome_video}.mp4
```

Esempio: video `squat.mov` → output `captions_squat.mp4`

Dopo ogni run viene creato/aggiornato `run_log.json` nella cartella di output con:

```json
{
  "folder_number": 42,
  "basename": "squat",
  "status": "success",
  "timestamp": "2026-06-11T10:30:00+00:00",
  "output_path": "/path/to/_output/captions_squat.mp4"
}
```

Status possibili: `success`, `failed`, `skipped`.

## Export 4K + HLG HDR

Il video finale viene esportato in **4K verticale (2160×3840, 9:16)** con **HLG** (HEVC 10-bit, BT.2020, arib-std-b67), allineato al riferimento Mirage Captions.

La pipeline ha due passaggi di encoding:

1. **Intermedio per Mirage** — 4K/HLG con `mirage_upload_crf` (default `24`)
2. **Export finale per il cliente** — dopo i sottotitoli Mirage, re-encode a 4K/HLG con `video_crf` (default `20`)

> **Nota:** i file sorgente `.mov` sono tipicamente SDR (BT.709). La conversione a HLG avviene via `zscale` senza `tonemap=hable` (inappropriato per SDR→HDR). Per HDR nativo servirebbe footage sorgente già in HDR.

Se i file superano il limite upload Mirage, alza `mirage_upload_crf` (es. `28` o `30`) nel `config.yaml`.

## Pipeline per ogni cartella

1. **FFmpeg** — legge durata voiceover `.m4a`, taglia `.mov`, upscale 4K (2160×3840), mix audio stereo, encode HLG per upload Mirage
2. **Mirage API** — upload intermedio, poll fino a `COMPLETE`, download video con sottotitoli
3. **Finalizzazione** — re-encode 4K HLG stereo per consegna cliente
4. **Salvataggio** — file finale in `_output/`

Progresso in console:

```
[42/344] squat → TRIM ✓ | UPLOAD ✓ | PROCESSING... | COMPLETE ✓
```

## Gestione errori

| Caso | Comportamento |
|------|---------------|
| Coppia mov/m4a mancante | Skip (`missing_pair`) |
| Più file mov o m4a | Skip (`ambiguous_files`) |
| File iCloud non scaricato | Skip (`not_downloaded_from_icloud`) |
| Output già esistente | Skip (usa `--force` per sovrascrivere) |
| Video più corto del target | Warning, continua |
| File > 50 MB | Warning, tenta upload comunque |
| Rate limit / errore 5xx Mirage | Retry con backoff esponenziale (max 3) |
| Job Mirage FAILED/CANCELLED | Log con codice e messaggio errore |
| API irraggiungibile | Messaggio chiaro, fail graceful |

## Licenza

Uso interno — progetto RP fisioterapia.
