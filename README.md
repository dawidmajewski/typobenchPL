# typobenchPL

`typobenchPL` jest benchmarkiem polskich tekstów generowanych przez małe modele causal LM.
Uruchamiany na stałym zestawie promptów, zwraca liczbę jednoznacznych błędów zapisu
na 10 000 ocenionych znaków.

Środowisko uruchomieniowe zawiera gotowy zestaw do testowania autoregresyjnych modeli językowych typu decoder-only (causal LM).

## Metryka

Benchmark ocenia jednoznaczne błędy na poziomie znaków i interpunkcji. Zakres
obejmuje błędne odstępy, powtórzone separatory, niepoprawne ciągi kropek,
uszkodzony Unicode i niezbilansowane nawiasy.

Każdy wygenerowany tekst otrzymuje pomocniczy wynik:

- `1` (`PASS`), jeśli nie znaleziono żadnego obsługiwanego błędu;
- `0` (`FAIL`), jeśli znaleziono przynajmniej jeden pewny błąd.

Główna metryka modelu uwzględnia wszystkie wykryte błędy:

```text
issues_per_10k_chars = 10 000 * liczba błędów / liczba ocenionych znaków
```

Niższy wynik jest lepszy, a `0` oznacza brak wykrytych błędów z aktualnej listy
reguł. Do mianownika wliczane są wyłącznie ocenione znaki completion; prompt jest
sprawdzany razem z completion, ale nie zwiększa pokrycia. Pomocniczy
`clean_output_rate` podaje procent generacji bez błędu.

## Przebieg benchmarku

```text
100 promptów z polish-prose-v1, po co najmniej 3000 ocenionych znaków completion
              ↓
model generuje do osiągnięcia kwoty dla każdego promptu
              ↓
prompt + completion
              ↓
deterministyczna lista wszystkich błędów
              ↓
liczba błędów na 10 000 ocenionych znaków
```

Model jest ładowany raz. Dla każdego promptu generowanie trwa do osiągnięcia kwoty
znaków albo limitu prób. Seed jest wyliczany z indeksu promptu i numeru próby, dzięki
czemu kolejność generacji jest deterministyczna.

Jeśli generacja osiągnie `max_new_tokens`, oceniany jest jej najdłuższy prefiks
zakończony pełnym zdaniem. Urwany ogon nie zwiększa pokrycia i nie powoduje
fałszywego błędu niedomkniętego nawiasu. Informacja o ucięciu pozostaje w artefaktach.

## Instalacja

Projekt wymaga Pythona 3.12 lub nowszego oraz `uv`.

Do używania samego walidatora wystarczy:

```bash
uv sync
```

Do uruchamiania modeli:

```bash
uv sync --extra hf
```

`hf` instaluje `torch`, `transformers` i `safetensors` wymagane do uruchomienia
komendy `run`. Podstawowa instalacja obsługuje komendy `check`, `score` i `verify`.

## Uruchomienie modelu

Model można wskazać bezpośrednio przez identyfikator Hugging Face. Zostanie
pobrany do standardowego cache Hugging Face i uruchomiony lokalnie:

```bash
uv run typobenchpl run \
  --model SlayerLab/GoLLeM-110M-PL-v3 \
  --suite polish-prose-v1
```

Model można też pobrać do katalogu `models/` i uruchomić ze ścieżki. Katalog
`models/` jest wpisany do `.gitignore`.

```bash
hf download SlayerLab/GoLLeM-110M-PL-v3 \
  --local-dir models/GoLLeM-110M-PL-v3

uv run typobenchpl run \
  --model models/GoLLeM-110M-PL-v3 \
  --suite polish-prose-v1
```

Na macOS można zapobiec uśpieniu komputera podczas długiego uruchomienia:

```bash
caffeinate -i uv run typobenchpl run \
  --model models/GoLLeM-110M-PL-v3 \
  --suite polish-prose-v1
```

Obsługiwane są również lokalne, autoregresyjne modele Transformer operujące na bajtach
UTF-8. Benchmark rozpoznaje zgodną architekturę na podstawie `config.json` i ładuje
wagi `safetensors` bez wykonywania kodu Python dołączonego do repozytorium modelu:

```bash
hf download SlayerLab/claude-data-set-tuned-8m-b74a2550 \
  --local-dir models/claude-data-set-tuned-8m-b74a2550

uv run typobenchpl run \
  --model models/claude-data-set-tuned-8m-b74a2550 \
  --suite polish-prose-v1
```

Podczas generowania komenda raportuje na `stderr` pokrycie znaków, czas i przewidywany
czas zakończenia. Po zakończeniu zapisuje liczbę błędów na 10 000 ocenionych znaków do
`stdout`, a ścieżkę katalogu z artefaktami do `stderr`:

```text
coverage: 300000/300000 chars (100.0%) elapsed 00:09:54 eta 00:00:00
2.000
results: runs/GoLLeM-110M-PL-v3/polish-prose-v1-<timestamp>
```

Rozdzielenie strumieni pozwala skryptom automatyzującym benchmark przechwycić sam wynik
liczbowy bez ścieżki do artefaktów. W trybie `--json` komenda zapisuje na `stdout`
pełne podsumowanie wraz ze ścieżką `output_directory` i nie wypisuje osobnej linii
`results:`.

## Parametry `run`

```bash
uv run typobenchpl run --model MODEL [opcje]
```

| Opcja | Domyślna wartość | Znaczenie |
| --- | --- | --- |
| `--model` | wymagana | Identyfikator Hugging Face albo lokalny katalog modelu |
| `--suite` | `polish-prose-v1` | Nazwa wbudowanej suite albo ścieżka do własnej |
| `--chars-per-prompt` | `3000` | Minimalna liczba ocenionych znaków dla każdego promptu |
| `--max-attempts-per-prompt` | `100` | Maksymalna liczba generacji dla jednego promptu |
| `--revision` | domyślna rewizja | Branch, tag lub commit modelu Hugging Face; nie dotyczy lokalnych modeli bajtowych |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N`, `mps` albo wybór automatyczny |
| `--output-dir` | automatyczny | Ścieżka nowego, nieistniejącego katalogu na artefakty |
| `--json` | wyłączone | Zwraca pełne podsumowanie JSON zamiast wyniku liczbowego |

Tryb `auto` wybiera kolejno CUDA, MPS albo CPU. Obsługiwane są modele completion zgodne
z `AutoModelForCausalLM` oraz lokalne Transformery z tokenizerem bajtowym UTF-8.

## Artefakty

Bez `--output-dir` każde uruchomienie tworzy katalog z datą, aby zapobiec przypadkowemu
nadpisaniu wyników:

```text
runs/<model>/<suite>-<timestamp>/
├── manifest.json
├── outputs.jsonl
└── summary.json
```

`manifest.json` zapisuje model, revision, urządzenie, wersje użytych bibliotek,
konfigurację suite, hash suite, czas uruchomienia i status. Podany przez
`--output-dir` katalog musi być nowy; benchmark nie nadpisuje istniejących artefaktów.

`outputs.jsonl` zawiera prompt, numer próby, completion, pełny wygenerowany tekst,
oceniany prefiks, seed, liczbę znaków i tokenów, przyczynę zakończenia, czas generacji,
wynik `0/1` oraz wszystkie wykryte błędy. Niepoprawne bajty UTF-8 są zachowane
szesnastkowo i nie zwiększają pokrycia. Puste completion zawsze otrzymuje `FAIL`.

`summary.json` zawiera pokrycie każdego promptu, liczbę prób, wygenerowanych,
ocenionych i odrzuconych znaków, `issues_per_10k_chars`, `clean_output_rate`, liczbę
niepoprawnych i uciętych wyjść oraz rozkład wszystkich błędów według reguł. Status
`incomplete` w `manifest.json` oznacza, że co najmniej jeden prompt nie osiągnął kwoty
przed limitem prób. Komenda nadal zapisuje artefakty, wypisuje informację o brakującym
pokryciu na `stderr` i kończy się kodem `2`.

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

Opcja `--json` zwraca listę wszystkich błędów wraz z regułami i zakresami. Kod wyjścia
procesu to `0` dla PASS, `1` dla FAIL oraz `2` dla błędnego użycia lub wejścia.

## Scoring JSONL

Komenda `score` ocenia rekordy JSONL z tekstami wygenerowanymi przez dowolny system.

Wejście musi być w formacie JSONL. Każdy rekord zawiera tekst w polu `text` albo
surowe bajty zapisane szesnastkowo w `text_bytes_hex`:

```json
{"id":"run-0001","text":"Pierwszy poprawny tekst."}
{"id":"run-0002","text":"Drugi,, błędny tekst."}
```

Opcjonalne pola `evaluated_chars` i `generated_chars` pozwalają zachować informacje
o przycięciu tekstu. Dla pola `text` domyślną wartością `evaluated_chars` jest długość
tekstu. Dla `text_bytes_hex` wynosi ona `0`, dlatego liczbę ocenionych znaków należy
podać jawnie. `generated_chars` domyślnie przyjmuje wartość `evaluated_chars`.

Domyślnie ocenianych jest dokładnie 1000 rekordów:

```bash
uv run typobenchpl score outputs.jsonl
uv run typobenchpl score outputs.jsonl -n 5000
uv run typobenchpl score outputs.jsonl -n 1000 --json
```

Jeśli ścieżka nie zostanie podana, `score` czyta dane ze standardowego wejścia.
Jeżeli wejście ma mniej niż `n` rekordów, komenda kończy się błędem. Rekordy ponad
limit są pomijane. Domyślnie komenda wypisuje `issues_per_10k_chars`; wariant `--json`
zwraca pełne podsumowanie, w tym liczbę wszystkich błędów i ocenionych znaków.

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

Bieżący zestaw reguł jest przeznaczony do oceny zwykłej prozy. Markdown, kod źródłowy
i ścieżki plików używają składni, której walidator obecnie nie rozpoznaje, dlatego
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
