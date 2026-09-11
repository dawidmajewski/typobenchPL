# typobenchpl

`typobenchpl` jest benchmarkiem polskich tekstów generowanych przez modele causal LM. Uruchamiany na stałym zestawie promptów, zwraca wynik od `0` do `100` na podstawie jednoznacznych, prymitywnych błędów zapisu.

Środowisko uruchomieniowe zawiera gotowy zestaw do testowania autoregresyjnych modeli językowych typu decoder-only (causal LM).

## Metryka

Benchmark ocenia jednoznaczne błędy na poziomie znaków i interpunkcji. Zakres
obejmuje błędne odstępy, powtórzone separatory, niepoprawne ciągi kropek,
uszkodzony Unicode i niezbilansowane nawiasy.

Każdy wygenerowany tekst otrzymuje:

- `1` (`PASS`), jeśli nie znaleziono żadnego obsługiwanego błędu;
- `0` (`FAIL`), jeśli znaleziono przynajmniej jeden pewny błąd.

Wynik modelu jest procentem zaliczonych generacji:

```text
score = 100 * liczba PASS / liczba wszystkich generacji
```

Wynik `100` oznacza brak wykrytych błędów z aktualnej listy reguł. Zakres wyniku
obejmuje wyłącznie reguły opisane na końcu README.

## Przebieg benchmarku

```text
100 promptów z polish-prose-v1
              ↓
model generuje n completions
              ↓
prompt + completion
              ↓
deterministyczny PASS/FAIL dla każdego tekstu
              ↓
średnia wyników * 100
```

Model jest ładowany raz. Prompty są wybierane cyklicznie, a kolejne
generacje otrzymują seedy `42`, `43`, `44` [...], co pozwala na odtworzenie konfiguracji.

## Instalacja

Projekt wymaga Python >3.12 oraz `uv`.

Do używania samego walidatora wystarczy:

```bash
uv sync
```

Do uruchamiania modeli Hugging Face:

```bash
uv sync --extra hf
```

`hf` instaluje `torch` i `transformers` wymagane do uruchomienia komendy `run`.
Podstawowa instalacja obsługuje komendy `check`, `score` i `verify`.

## Uruchomienie modelu z Hugging Face

Model można wskazać bezpośrednio przez identyfikator Hugging Face. Zostanie
pobrany do standardowego cache Hugging Face i uruchomiony lokalnie:

```bash
uv run typobenchpl run \
  --model SlayerLab/GoLLeM-110M-PL-v3 \
  --suite polish-prose-v1 \
  -n 1000
```

Model można też pobrać do katalogu `models/` i uruchomić ze ścieżki. Katalog
`models/` jest wpisany do `.gitignore`.

```bash
hf download SlayerLab/GoLLeM-110M-PL-v3 \
  --local-dir models/GoLLeM-110M-PL-v3

uv run typobenchpl run \
  --model models/GoLLeM-110M-PL-v3 \
  --suite polish-prose-v1 \
  -n 1000
```

Po zakończeniu komenda zapisuje wynik procentowy do `stdout`, a ścieżkę katalogu
z artefaktami do `stderr`. W terminalu widoczne są obie linie:

```text
97.40
results: runs/GoLLeM-110M-PL-v3/polish-prose-v1-<timestamp>
```

Rozdzielenie strumieni pozwala skryptom automatyzującym benchmark przechwycić sam wynik liczbowy bez
ścieżki do artefaktów.

## Parametry `run`

```bash
uv run typobenchpl run --model MODEL [opcje]
```

| Opcja | Domyślna wartość | Znaczenie |
| --- | --- | --- |
| `--model` | wymagana | Identyfikator Hugging Face albo lokalny katalog modelu |
| `--suite` | `polish-prose-v1` | Nazwa wbudowanej suite albo ścieżka do własnej |
| `-n`, `--runs` | `1000` | Liczba generacji i ocenianych tekstów |
| `--revision` | domyślna rewizja | Branch, tag lub commit modelu Hugging Face |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N`, `mps` albo wybór automatyczny |
| `--output-dir` | automatyczny | Nowy katalog na artefakty uruchomienia |
| `--json` | wyłączone | Zwraca pełne podsumowanie JSON zamiast samego wyniku |

Tryb `auto` wybiera kolejno CUDA, MPS albo CPU. Zakres wersji 1.0 obejmuje lokalne modele
completion zgodne z `AutoModelForCausalLM`.

## Artefakty

Każdy `run` tworzy z suffixem daty, aby zapobiec przypadkowemu nadpisaniu wyników:

```text
runs/<model>/<suite>-<timestamp>/
├── manifest.json
├── outputs.jsonl
└── summary.json
```

`manifest.json` zapisuje model, revision, urządzenie, wersje Torch i
Transformers, konfigurację suite, hash suite oraz czas uruchomienia.

`outputs.jsonl` zawiera prompt, completion, pełny oceniany tekst, seed, liczbę
tokenów, czas generacji, wynik `0/1` i szczegóły wykrytego błędu. Puste
completion zawsze otrzymuje `FAIL`.

`summary.json` zawiera końcowy score, liczbę PASS/FAIL, czas wykonania i rozkład
błędów według reguł.

## Zestaw promptów

Suite `polish-prose-v1` zawiera 100 promptów typu completion:

| Kategoria | Liczba promptów |
| --- | ---: |
| Narracja | 20 |
| Opis | 15 |
| Życie codzienne | 15 |
| Tekst objaśniający | 15 |
| Dialog pośredni | 10 |
| Kultura | 10 |
| Tekst techniczny | 10 |
| Tekst formalny | 5 |

Parametry porównawcze są zapisane w suite:

```toml
[generation]
max_new_tokens = 80
do_sample = true
temperature = 0.7
top_k = 40
repetition_penalty = 1.3
seed = 42
prepend_bos = "auto"
```

Pliki suite znajdują się w:

- `src/typobenchpl/suites/polish-prose-v1/suite.toml`
- `src/typobenchpl/suites/polish-prose-v1/prompts.jsonl`

Własną suite można przekazać jako katalog zawierający pliki `suite.toml` i
`prompts.jsonl` o tej samej strukturze.

## Walidacja tekstu

```bash
uv run typobenchpl check --text "To jest poprawna proza."
# 1

uv run typobenchpl check --text "To jest,, błędna proza."
# 0
```

Tekst można również przekazać przez plik lub standardowe wejście:

```bash
uv run typobenchpl check wynik.txt
printf '%s' 'To jest poprawne.' | uv run typobenchpl check -
```

Opcja `--json` zwraca regułę oraz zakres błędu. Kod wyjścia procesu to `0` dla
PASS, `1` dla FAIL oraz `2` dla błędnego użycia lub wejścia.

## Scoring JSONL

Komenda `score` ocenia gotowy plik JSONL z tekstami wygenerowanymi przez dowolny
system.

Plik wejściowy musi być w formacie JSONL:

```json
{"id":"run-0001","text":"Pierwszy poprawny tekst."}
{"id":"run-0002","text":"Drugi,, błędny tekst."}
```

Domyślnie ocenianych jest dokładnie 1000 rekordów:

```bash
uv run typobenchpl score outputs.jsonl
uv run typobenchpl score outputs.jsonl -n 5000
uv run typobenchpl score outputs.jsonl -n 1000 --json
```

Jeżeli plik ma mniej niż `n` rekordów, komenda kończy się błędem. Rekordy ponad
limit są pomijane.

## Weryfikacja

Wbudowany zestaw `MUST_PASS`/`MUST_FAIL` sprawdza, czy sam benchmark podejmuje
oczekiwane decyzje:

```bash
uv run typobenchpl verify
# 52/52 tests passed
```

Własny zestaw można uruchomić przez:

```bash
uv run typobenchpl verify --cases-dir path/to/cases
```

Katalog musi zawierać `must_pass.jsonl` i `must_fail.jsonl`. Każdy przypadek ma
stabilne `id` oraz oczekiwane `0/1`; przypadki negatywne mogą dodatkowo określać
regułę i dokładny zakres błędu.

## Obsługiwane teksty

Rozpoznane URL-e, adresy e-mail, liczby dziesiętne, godziny, wyniki liczbowe,
adresy IPv4 i numery wersji są chronione przed regułami interpunkcji.

Wersja 1.0 jest przygotowana do oceny zwykłej prozy. Markdown, kod źródłowy i
ścieżki plików używają składni, której walidator obecnie nie rozpoznaje, dlatego
wyniki dla takich treści mogą być błędne.

## Testy

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

## Reguły walidacji

| Reguła | Warunek FAIL | Przykład |
| --- | --- | --- |
| `G000` | Pusty tekst albo same białe znaki | `""`, `"   "` |
| `G001` | Wejście nie jest poprawnym UTF-8 | bajt `FF` |
| `G002` | Niedozwolony znak kontrolny | `NUL`, `BEL`, `ESC` |
| `G003` | Niesparowany Unicode surrogate | samotny high surrogate |
| `G010` | Co najmniej dwa przecinki obok siebie | `tekst,, tekst` |
| `G011` | Co najmniej dwa średniki obok siebie | `tekst;; tekst` |
| `G012` | Ciąg kropek ma długość inną niż 1 lub 3 | `..`, `....` |
| `G020` | Biały znak przed przecinkiem | `tekst , tekst` |
| `G021` | Biały znak przed średnikiem | `tekst ; tekst` |
| `G022` | Biały znak przed dwukropkiem | `tekst : tekst` |
| `G023` | Biały znak przed pojedynczą kropką | `tekst .` |
| `G024` | Biały znak przed pytajnikiem | `tekst ?` |
| `G025` | Biały znak przed wykrzyknikiem | `tekst !` |
| `G030` | Brak odstępu po przecinku między literami | `Ala,ma kota.` |
| `G031` | Brak odstępu po średniku między literami | `tekst;drugi tekst` |
| `G032` | Brak odstępu po dwukropku między literami | `Powiedział:tak.` |
| `G033` | Brak odstępu po `?` lub `!` przed kolejnym słowem | `Naprawdę?Tak.` |
| `G040` | Nawiasy zamykają się w złej kolejności | `([tekst)]` |
| `G041` | Nawias otwierający lub zamykający nie ma pary | `(tekst`, `tekst)` |

## Przypadki MUST_PASS

| ID | Tekst lub przypadek |
| --- | --- |
| `pass-normal-001` | `To jest poprawne zdanie.` |
| `pass-normal-002` | `Zażółć gęślą jaźń.` |
| `pass-normal-003` | `Krótki nagłówek` |
| `pass-normal-004` | Dwa poprawne akapity rozdzielone nową linią |
| `pass-number-001` | `Temperatura wynosi 3,14°C.` |
| `pass-number-002` | `Spotkanie zaczyna się o 12:30.` |
| `pass-number-003` | `Wynik to 1:2.` |
| `pass-number-004` | `Wersja 1.2.3 działa poprawnie.` |
| `pass-number-005` | `Serwer ma adres 192.168.1.1.` |
| `pass-punctuation-001` | `Naprawdę?!` |
| `pass-punctuation-002` | `Czyżby!?` |
| `pass-punctuation-003` | `Serio??` |
| `pass-punctuation-004` | `Nie!!` |
| `pass-punctuation-005` | `To było... interesujące.` |
| `pass-punctuation-006` | `To było… interesujące.` |
| `pass-punctuation-007` | `Ha ha, bardzo zabawne.` |
| `pass-punctuation-008` | `J.K. Rowling napisała książkę.` |
| `pass-quotes-001` | `Powiedział: „Nie wiem”.` |
| `pass-quotes-002` | `Jan zapytał: „Naprawdę?!”` |
| `pass-brackets-001` | `To jest (poprawny) przykład.` |
| `pass-brackets-002` | `To jest ([zagnieżdżony]) przykład.` |
| `pass-brackets-003` | `Zbiory {a, b} oraz [c, d] są opisane.` |
| `pass-url-001` | Standardowy URL zakończony kropką zdania |
| `pass-url-002` | URL zawierający `,,` w ścieżce |
| `pass-url-003` | URL zawierający `;;` oraz `...` |
| `pass-url-004` | URL zawierający zbilansowane nawiasy |
| `pass-email-001` | `Napisz na test@example.com.` |
| `pass-whitespace-001` | Dwie kolumny rozdzielone tabulatorem |

## Przypadki MUST_FAIL

| ID | Reguła | Tekst lub przypadek |
| --- | --- | --- |
| `fail-empty-001` | `G000` | Pusty tekst |
| `fail-empty-002` | `G000` | Trzy spacje |
| `fail-encoding-001` | `G001` | Niepoprawny bajt UTF-8 `FF` |
| `fail-control-001` | `G002` | `Tekst<NUL>test` |
| `fail-surrogate-001` | `G003` | `Tekst<high-surrogate>test` |
| `fail-comma-001` | `G010` | `To jest,, błąd.` |
| `fail-comma-002` | `G010` | `To jest,,, błąd.` |
| `fail-semicolon-001` | `G011` | `To jest;; błąd.` |
| `fail-dots-001` | `G012` | `To jest.. błąd.` |
| `fail-dots-002` | `G012` | `To jest.... błąd.` |
| `fail-space-before-comma-001` | `G020` | `To jest , błąd.` |
| `fail-space-before-semicolon-001` | `G021` | `Pierwsza część ; druga.` |
| `fail-space-before-colon-001` | `G022` | `Powiedział : tak.` |
| `fail-space-before-period-001` | `G023` | `Koniec .` |
| `fail-space-before-question-001` | `G024` | `Czy to działa ?` |
| `fail-space-before-exclamation-001` | `G025` | `Uwaga !` |
| `fail-space-after-comma-001` | `G030` | `Ala,ma kota.` |
| `fail-space-after-semicolon-001` | `G031` | `Ala ma kota;Jan ma psa.` |
| `fail-space-after-colon-001` | `G032` | `Powiedział:tak.` |
| `fail-space-after-question-001` | `G033` | `Naprawdę?Tak.` |
| `fail-space-after-exclamation-001` | `G033` | `Uwaga!Uciekaj.` |
| `fail-bracket-order-001` | `G040` | `To jest ([błąd)] tutaj.` |
| `fail-bracket-open-001` | `G041` | `To jest (błąd.` |
| `fail-bracket-close-001` | `G041` | `To jest błąd).` |
