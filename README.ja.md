# LocalForge — パーソナライズドローカルAIコード生成・プロジェクトインテリジェンスIDE

完全ローカル動作のAI駆動コード生成・プロジェクト解析ツールです。LocalForgeはOllamaをLLMバックエンドとして使用し、ポート7331のローカルFlaskサーバーを介して`pywebview`によるネイティブデスクトップウィンドウとして動作します。クラウドAPIは一切使用しません。

## 機能

### 3つのモード

| モード | 説明 |
|---|---|
| **Generate（生成）** | AI駆動のプランニングとファイル生成で新規プロジェクトをゼロから作成 |
| **Resume（再開）** | 既存プロジェクト（LocalForge製・外部製問わず）の続行・拡張 |
| **Explain（解説）** | コードベースの深層解析、11セクションの詳細インテリジェンスレポート生成、インタラクティブQ&A |

### RAG対応コードインテリジェンス

LocalForgeはExplainモードで**RAG（Retrieval-Augmented Generation）**を使用します：
- 全ファイルが`nomic-embed-text:latest`によって**ChromaDB**ベクトルストアにサマリー・埋め込みされます
- Q&Aとレポート生成は全ファイルをスキャンするのではなく、最も意味的に関連するファイルサマリーを取得します
- 埋め込みはJSONLインデックス後に**並列処理**（4並列ワーカー）で実行されます — ステータスバーに進捗が表示されます
- 増分処理：変更されたファイルのみ再インデックス・再埋め込みされます
- フォールバック：ChromaDBまたは埋め込みモデルが利用不可の場合、BM25キーワード検索が自動的に使用されます

### 多層キャッシュによる高速レスポンス

ホットパスデータは全てキャッシュされるため、同一の質問や未変更ファイルへの再処理を回避できます：

| キャッシュ | 保存内容 | 無効化タイミング |
|---|---|---|
| **レスポンスキャッシュ** | 質問+コンテキストハッシュをキーとしたQ&A回答全文 | インデックス再構築または質問変更時 |
| **ファイル内容キャッシュ** | パス+mtime+サイズをキーとしたファイルテキスト | ファイル変更時 |
| **セマンティック検索キャッシュ** | クエリ+チャンク数をキーとしたTop-N結果 | インデックス再構築時 |
| **index_jsonキャッシュ** | プロジェクトインデックスJSON文字列 | インデックスmtime変更時 |

キャッシュのホット層はメモリ内に保持され、コールド層は`.localforge/cache/`に永続化されるためサーバー再起動後も有効です。

### バックグラウンドモデルウォームアップ

Q&AおよびレポートパイプラインのMost開始時に、LocalForgeはバックグラウンドスレッドを起動してOllamaモデルのRAMへのロードを開始します。これはコンテキスト組み立て（ファイル読み込み、ベクトル検索、プロンプト構築）と並行して実行されるため、最初のOllama呼び出し時にはモデルが既にロード済み（またはロードがかなり進んだ状態）となり、最初のトークンまでの待機時間が大幅に短縮されます。

### シンキングモデル対応

`thinking`フィールドを公開するモデル（Gemma、QwQなど）や`<think>` XMLタグを使用するモデル（DeepSeek-R1など）の推論過程は、メイン生成出力を汚染することなく**Ollamaライブパネル**（折りたたみ可能な右サイドバー）にストリーミングされます。

---

## 前提条件

### 1. Ollamaのインストール

[https://ollama.com](https://ollama.com)からOllamaをダウンロード・インストールし、モデルをpullします：

```bash
ollama pull llama3.2
```

RAG対応のExplainモードには埋め込みモデルも必要です：

```bash
ollama pull nomic-embed-text:latest
```

Ollamaサーバーを起動します（ほとんどのシステムで自動起動されます）：
```bash
ollama serve
```

### 2. Python 3.10以上

Python 3.10以降がインストールされていることを確認してください。

---

## インストール

### オプションA — pip

```bash
# リポジトリをクローン
git clone <repo-url>
cd localforge_web

# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### オプションB — Poetry

#### 1. Poetryのインストール

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

確認：

```bash
poetry --version
```

> 一部のシステムでは`~/.local/bin`を`PATH`に追加する必要があります。

#### 2. プロジェクト依存関係のインストール

```bash
git clone <repo-url>
cd localforge_web
poetry install
```

独立した仮想環境が作成され、全てのランタイム・開発依存関係が自動的にインストールされます。

#### 3. 環境のアクティベート（オプション）

```bash
poetry shell
```

これ以降は`poetry run`プレフィックスなしで`python main.py`を直接実行できます。

### オプションC — Docker（LANアクセス、ヘッドレス）

Dockerコンテナ内でLocalForgeをヘッドレスFlaskサーバーとして実行します。ネイティブデスクトップウィンドウは使用できませんが、**同じWi-Fiネットワーク上の任意のブラウザ**からフルUIにアクセスできます — 高性能マシンで重いOllamaの処理を行いつつ、ラップトップ・タブレット・スマートフォンからブラウジングする際に便利です。

#### 前提条件

- ホストに[Docker](https://docs.docker.com/get-docker/)と[Docker Compose](https://docs.docker.com/compose/)がインストール済み
- **ホストマシンでOllamaが動作していること**（コンテナはホストのOllamaに接続するため、GPUはホスト側で使用されます）

#### ビルドと起動

```bash
git clone <repo-url>
cd localforge_web
docker compose up --build
```

初回ビルドは数分かかります（chromadbのネイティブ拡張のコンパイル）。以降の起動は即座です。

#### 同一ネットワーク上のデバイスからのアクセス

ホストマシンのローカルIPアドレスを確認：

```bash
# macOS / Linux
hostname -I | awk '{print $1}'
```

```powershell
# Windows PowerShell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object InterfaceAlias -ne Loopback).IPAddress
```

同じWi-Fi上の任意のブラウザで`http://<ホストIP>:7331`を開きます。

#### プロジェクトを開く

コンテナ内にはデスクトップ環境がないため、ネイティブフォルダダイアログは使用できません。**📁 Open Folder**をクリックするとテキストプロンプトが表示されますので、コンテナ内のパスを入力します：

```
/projects/my-app
```

デフォルトでは`docker-compose.yml`の隣に`projects/`フォルダが作成され、コンテナ内の`/projects`にマウントされます。既存のディレクトリを指定する場合：

```bash
# Linux / macOS
PROJECTS_DIR=/home/alice/code docker compose up

# Windows CMD
set PROJECTS_DIR=C:\Users\alice\code && docker compose up

# Windows PowerShell
$env:PROJECTS_DIR="C:\Users\alice\code"; docker compose up
```

#### Ollama接続

Ollamaはホストマシンで動作し、`host.docker.internal:11434`経由でアクセスされます（事前設定済み）。別のOllamaコンテナは不要で、GPUはホストのOllamaプロセスが直接使用します。

別のOllamaインスタンスを指定する場合は`docker-compose.yml`を編集：

```yaml
environment:
  OLLAMA_HOST: "http://192.168.1.10:11434"
```

#### localhostのみに制限

LANアクセスを無効にする場合は`docker-compose.yml`のポートバインディングを変更：

```yaml
ports:
  - "127.0.0.1:7331:7331"
```

#### コンテナの停止

```bash
docker compose down          # 停止（インデックス・キャッシュデータは保持）
docker compose down -v       # 停止してデータも削除
```

### pywebviewのプラットフォーム固有の注意事項

| プラットフォーム | 注意事項 |
|---|---|
| **macOS** | 追加依存関係不要（WebKitを使用） |
| **Linux** | GTK3 + WebKit2GTKが必要: `sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0` |
| **Windows** | 追加依存関係不要（Edge WebView2を使用） |

---

## 実行方法

### pip

```bash
python main.py
```

### Poetry

```bash
poetry run python main.py
# またはアクティベート済みシェル内（poetry shell）:
python main.py
```

### Docker

```bash
docker compose up            # 起動してhttp://<ホストIP>:7331を開く
docker compose up --build    # 先にイメージを再ビルド（依存関係変更後）
docker compose up -d         # バックグラウンドで実行
```

起動すると：
1. `http://127.0.0.1:7331`でFlaskサーバーが起動
2. pywebviewによるネイティブデスクトップウィンドウが開く
3. 閉じると、VRAM/RAMを解放するためにOllamaプロセスが自動終了

pywebviewが利用できない場合は、ブラウザで`http://127.0.0.1:7331`を手動で開いてください。

---

## 使い方

### Generateモード

1. **📁 Open Folder**をクリックして空のディレクトリを選択
2. プロンプトエリアにプロジェクトの説明を入力
3. **Generate Plan**をクリックしてAI生成プロジェクト構造を作成
4. プランを確認・編集して**Approve & Generate**をクリック
5. LocalForgeが各ファイルをgitコミットと共に生成する様子を確認

> 生成中は全アクションボタンが無効化され、ヘッダーに**⏹ 停止**ボタンが表示されます。クリックすると生成が即座に停止しUIがアンロックされます。

### Resumeモード

以前作業したプロジェクトフォルダが検出された場合に自動的に開きます：
- **LocalForgeプロジェクト**：未完のファイル生成を継続またはプランを変更
- **外部プロジェクト**：解析レポートの閲覧、Q&Aの継続、新規ファイルの生成

### Explainモード

`.localforge/`のないコードを含むフォルダが検出された場合に自動的に開きます：
1. **⚙ インデックス構築**をクリックしてコードベースを解析（増分再インデックス対応）
2. インデックス構築は2段階で実行：LLMサマリー生成 → 並列RAG埋め込み
3. **レポート生成**をクリックして11セクションのインテリジェンスレポートを作成
4. **Q&Aチャット**でコードベースについての質問に対話形式で回答

**Q&Aパフォーマンスについて：**
- 同一の質問はキャッシュされた回答が即座に返されます（LLM呼び出しなし）
- ファイル内容はmtime単位でキャッシュ — 質問間で未変更ファイルは再読み込みされません
- コンテキスト組み立て中にモデルのRAMロードが開始されるため、プロンプト後の待機時間が短縮されます

---

## アーキテクチャ

```
localforge/
├── domain/           # Pydanticモデル、ポートインターフェース、例外
├── application/      # ビジネスロジックサービス（I/Oなし、HTTPなし）
├── infrastructure/   # アダプター — 全てのI/Oはここに集約
└── interface/        # Flaskルート、Jinja2テンプレート、静的アセット
```

### 主要設計方針

| 方針 | 詳細 |
|---|---|
| **クリーンアーキテクチャ** | 厳格なレイヤー分離 — ルートにビジネスロジックなし |
| **SSEストリーミング** | 全LLM出力はサーバー送信イベントでストリーミング。15秒ごとのハートビートはLLMブロッキングとは独立してスレッドで送信 |
| **並列前処理** | ファイル読み込み、ワークスペースロード、セマンティック検索はOllama呼び出し前に`ThreadPoolExecutor`で並列実行 |
| **バックグラウンドモデルウォームアップ** | `OllamaClient.preload_model_async()`がコンテキスト組み立て中にモデルのRAMロードを開始 |
| **多層キャッシュ** | レスポンス・ファイル内容・セマンティック検索・index_jsonキャッシュがQ&A呼び出し間の冗長な処理を排除 |
| **非同期ログ書き込み** | `generation_log.jsonl`への追記はバックグラウンドスレッドで実行 — `done` SSEイベントはディスクI/Oでブロックされない |
| **増分インデックス** | 変更されたファイルのみ再処理（mtime + サイズチェック） |
| **並列RAG埋め込み** | `ThreadPoolExecutor(max_workers=4)`でJSONLインデックス後にChromaDBへ並列埋め込み |
| **ChromaDB自動修復** | `.localforge/chroma/`が欠落または古い場合、`build_index()`が自動的に再構築 |
| **UIロック** | アクティブなストリーム中は全アクションボタンを無効化。グローバル**⏹ 停止**ボタンで即時キャンセル |
| **トークン予算ガード** | 全LLM呼び出しは実行前にトークン予算を確認 |
| **ハイブリッドファイル読み込み** | 350行超のファイルはAST（Python）またはregex（JS/TS）による構造的ランドマークを使用 |
| **シンキングモデル対応** | `thinking`フィールド（Gemma）または`<think>`タグ（DeepSeek）を持つモデルの推論はOllamaライブパネルにルーティング |
| **Ollamaクリーンアップ** | SIGTERMハンドラー、atexit、pywebviewシャットダウン後の呼び出しでアプリ終了時にOllamaプロセスを終了 |
| **モデル切り替え時VRAM解放** | ユーザーが別のモデルを選択した際、`keep_alive: 0`で前のモデルを即時VRAMから退避 |
| **デフォルトモデルなし** | `ProjectConfig.model`はデフォルト`""`。全ルートでモデル選択を検証し、未選択の場合は明確なエラーを返す |

---

## `.localforge/`ディレクトリ

各プロジェクトには`.localforge/`メタデータディレクトリが作成されます（gitignore対象）：

```
.localforge/
├── config.json           # プロジェクト設定（model、token_limit）
├── context.md            # ローリングプロジェクトメモリ
├── project_index.json    # マスタープロジェクトサマリードキュメント
├── index.jsonl           # ファイルごとのサマリー（増分インデックス）
├── chroma/               # ChromaDBベクトルコレクション（RAG埋め込み）
├── generation_log.jsonl  # LLMインタラクションログ（非同期書き込み）
├── report.md             # 保存済み解説レポート
├── qa_history.md         # Q&A会話ログ
├── cache/
│   ├── responses/        # Q&Aレスポンスキャッシュ（回答ごとに1JSONファイル）
│   └── semantic/         # セマンティック検索結果キャッシュ
└── app.log               # ローテーションアプリケーションログ
```

---

## テストの実行

```bash
# pip
pip install pytest pytest-mock
python -m pytest tests/ -v

# Poetry（`poetry install`で開発依存関係は既にインストール済み）
poetry run pytest tests/ -v
```

テストはモックアダプターを使用 — 実際のOllama、ChromaDB、ファイルシステムアクセスは不要です。

---

## セキュリティ

- 非Dockerモードではポート7331はローカル専用（`127.0.0.1`のみ）
- ファイルアクセスAPIはプロジェクトルートに対してパスを検証（パストラバーサル防止）
- 全LLM呼び出しは`ollama_client.py`のみを経由
- ChromaDBテレメトリー無効（`anonymized_telemetry=False`）

---

## 設定

プロジェクトの`.localforge/config.json`を編集してカスタマイズできます：

| キー | デフォルト | 説明 |
|---|---|---|
| `model` | `""` | Ollamaモデル名（UIセレクターで設定必須） |
| `token_limit` | `131072` | LLM呼び出しごとのトークン予算 |
| `mode` | 自動検出 | `generate`、`resume`、または`explain` |
