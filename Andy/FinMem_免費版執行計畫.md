# FinMem 免費版完整執行計畫（半年、真實資料）

## 0. 範圍界定

- **股票**：TSLA（沿用專案原本的 config 設定，之後可換其他股票）
- **時間區間**：最近半年（約 126 個交易日），例如 2026-02 ~ 2026-08
- **目標**：用真實股價 + 真實（但疏落）新聞 + 少量真實財報，跑一次完整的訓練+測試流程，驗證整套記憶系統跟決策邏輯是否能正常運作

## 1. 需要的 API Key 現況

| API | 狀態 | 用途 |
|---|---|---|
| Gemini API Key | 已申請 | Chat（決策）+ Embedding（記憶向量化） |
| SEC API Key | 已申請（額度終身 100 次，省著用） | 抓真實 10-K/10-Q 財報 |
| Tavily API Key | 尚未申請 | 抓真實新聞（免費 1,000 額度/月，只要 email 註冊，不用信用卡、不用身分驗證） |

放棄使用的 API 及原因：

- **OpenAI API Key**：改用 Gemini 取代 chat 與 embedding，不需要
- **HF_TOKEN**：不使用 TGI / gated 的 HuggingFace 模型，不需要
- **Alpaca API**：申請流程要求提供真實稅務識別碼（KYC），改用 Tavily 取代新聞來源，避免交出身分資訊

## 2. 資料組成

| 資料 | 來源 | 真實性 | 抓法 |
|---|---|---|---|
| 股價 | `yfinance` | 真實 | 免費套件直接抓，不用 Key |
| 新聞 | Tavily | 真實（每週查一次，疏落分布） | 約 18~20 次查詢（126 天 ÷ 7） |
| 財報 | SEC API | 真實（數量有限） | 半年內 TSLA 可能有 1~2 份 10-Q，抓這幾份就好，省額度 |

## 3. 程式碼修改清單

| 檔案 | 修改內容 | 原因 |
|---|---|---|
| `.venv` | 重建成 Python 3.10 | 現在是 3.14，跟專案要求（`pyproject.toml` 限定 >=3.10,<3.11）不符 |
| `puppy/chat.py` | Gemini 驗證方式改成 API Key（不用 `gcloud auth`），端點改成 AI Studio；加上 429 自動重試/節流邏輯 | 目前寫死打 Vertex AI（已無免費額度），也沒處理速率限制中斷的問題 |
| `puppy/embedding.py` | 新增 Gemini embedding class，取代寫死的 OpenAI 版本 | 目前完全綁死呼叫 `OpenAIEmbeddings`，沒有其他分支 |
| `config/tsla_gemini_config.toml` | 模型名稱、端點、embedding 設定改成免費 Gemini 版本 | 目前指向舊的 Vertex AI 專案網址（`elite-destiny-371016`） |
| 新增：`fetch_price.py` | 用 `yfinance` 抓半年份 TSLA 股價 | 免費、不用 Key |
| 新增：`fetch_news_tavily.py` | 每週查一次 Tavily，把結果整理成新聞文字 | 取代 Alpaca，避開身分驗證 |
| 新增：`fetch_filing_sec.py`（改寫自 `data-pipeline/01_SEC_API_10k10q_download.py`） | 限定 1 檔股票、半年區間，避免亂用額度 | 原腳本股票清單是空的佔位字串，日期區間也要改 |
| 新增：`merge_dataset.py` | 把股價+新聞+財報直接合併成 `puppy/environment.py` 要的格式 | 跳過官方 `data-pipeline/04-data_pipeline.py` / `05-get_sentiment_by_ticker.py`，因為這兩支已發現有 bug（tuple 格式跟 `OneDateRecord` 對不上、`filing_q`/`filling_q` 打錯字），改自己寫一支乾淨的合併邏輯 |

## 4. 執行步驟流程

```
步驟 0：申請 Tavily API Key（email 註冊即可）

步驟 1：環境準備
  - 重建 Python 3.10 虛擬環境
  - 安裝依賴套件
  - .env 填入 GEMINI_API_KEY / SEC_KEY / TAVILY_API_KEY

步驟 2：抓真實資料
  - 跑 fetch_price.py       → 半年份 TSLA 真實股價
  - 跑 fetch_news_tavily.py → 每週真實新聞（約 18~20 次查詢）
  - 跑 fetch_filing_sec.py  → 該區間內的真實 10-Q（省著抓）

步驟 3：合併資料
  - 跑 merge_dataset.py → 產出 data/03_model_input/tsla_demo.pkl
    （格式驗證：符合 puppy/environment.py 的 OneDateRecord schema）

步驟 4：訓練模式（用前面約 4 個月的資料建立記憶）
  python run.py sim -mdp data/03_model_input/tsla_demo.pkl
                    -st <半年起始日> -et <約第4個月的某日>
                    -rm train -cp config/tsla_gemini_config.toml
                    -ckp data/06_train_checkpoint -rp data/05_train_model_output

步驟 5：測試模式（用剩下約 2 個月的資料，讓 Gemini 真的做買賣決策）
  python run.py sim -mdp data/03_model_input/tsla_demo.pkl
                    -st <第4個月某日> -et <半年結束日>
                    -rm test -cp config/tsla_gemini_config.toml
                    -tap data/06_train_checkpoint
                    -ckp data/08_test_checkpoint -rp data/09_results

步驟 6：匯出結果
  - 仿照 save_file.py，把每日 buy/sell/hold 決策匯出成 CSV
```

## 5. 跑完會拿到什麼

- `data/04_model_output_log/TSLA_run.log`：每一步的完整日誌（查到哪些記憶、LLM 原始輸出）
- `data/09_results/agent_1/`：訓練後的 agent checkpoint（記憶庫、投資組合狀態）
- 一份 CSV：半年內每天的 buy/sell/hold 決策

## 6. 預估時間

以 126 個交易日估算：

- Embedding 呼叫約 630 次、Chat 呼叫約 150 次，都在 Gemini 免費層每日 1,000 次上限內
- 不用跨天，一次坐著大概 45 分鐘～1 小時內可以跑完

## 7. 注意事項

- 新聞是「每週查一次」的疏落真實新聞，不是每天都有，這是為了配合 Tavily 的特性、也節省額度，架構上完全沒問題（`puppy/agent.py` 本來就允許某天沒新聞）
- SEC API 額度全專案只有 100 次，這次抓完可能只剩 90 幾次，之後想再抓別的股票要省著用
- 這次的交易績效不具備研究參考價值（資料量小、記憶密度稀疏），純粹是驗證架構能不能跑起來
