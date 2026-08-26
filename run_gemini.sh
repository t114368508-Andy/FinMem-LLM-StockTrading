#!/bin/bash
# Free-tier Gemini setup. GEMINI_API_KEY / SEC_KEY / TAVILY_API_KEY /
# FINNHUB_API_KEY must already be set in .env (see README.md) -- this script
# does not export them itself, unlike the legacy OpenAI/TGI scripts, since
# puppy/chat.py and puppy/embedding.py read GEMINI_API_KEY via python-dotenv.

# train
python run.py sim \
  -mdp data/03_model_input/tsla_demo.pkl \
  -st 2026-01-02 \
  -et 2026-04-30 \
  -rm train \
  -cp config/tsla_gemini_config.toml \
  -ckp data/06_train_checkpoint \
  -rp data/05_train_model_output

# # train-checkpoint (resume an interrupted train run)
# python run.py sim-checkpoint \
#   -ckp data/06_train_checkpoint \
#   -rp data/05_train_model_output \
#   -cp config/tsla_gemini_config.toml \
#   -rm train

# # test
# python run.py sim \
#   -mdp data/03_model_input/tsla_demo.pkl \
#   -st 2026-05-01 \
#   -et 2026-06-29 \
#   -rm test \
#   -cp config/tsla_gemini_config.toml \
#   -tap data/06_train_checkpoint \
#   -ckp data/08_test_checkpoint \
#   -rp data/09_results

# # test-checkpoint (resume an interrupted test run)
# python run.py sim-checkpoint \
#   -ckp data/08_test_checkpoint \
#   -rp data/09_results \
#   -cp config/tsla_gemini_config.toml \
#   -rm test

# # export the daily buy/sell/hold decisions to CSV
# python save_file.py \
#   --checkpoint data/08_test_checkpoint/agent_1 \
#   --out data/09_results/tsla_decisions.csv
