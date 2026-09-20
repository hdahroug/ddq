### 1. Install / check requirements

```bash
pip install -r requirements.txt
```


### 2. Download the model (only if you don't already have it)

Place `Qwen2.5-0.5B/` next to this `DDD` folder

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir Qwen2.5-0.5B \
  --include config.json model.safetensors tokenizer.json
```

### 3. Run

```bash
python run.py "The capital of France is" 30
```

