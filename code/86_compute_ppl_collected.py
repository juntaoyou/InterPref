"""
Compute token perplexity for all JSONL files in collected_results/.

Reference models are base models from the same family and parameter scale as
each instruction-tuned generator (Qwen3-8B-Base, Llama-3.1-8B, gemma-2-9b).
calculate_file_ppl() can apply an optional IQR outlier filter; the default is
remove_outliers=False.
"""
import sys, os, json
sys.path.insert(0, '<REPO_ROOT>/code')
import importlib

# The pipeline file was renamed to 86_ppl2.py; a numeric prefix is not a valid
# module identifier, so import it via importlib (no behaviour change).
_ppl2 = importlib.import_module('86_ppl2')
calculate_file_ppl = _ppl2.calculate_file_ppl
load_model_and_tokenizer = _ppl2.load_model_and_tokenizer

COLLECTED = '<REPO_ROOT>/eval/collected_results'
OUTPUT = '<REPO_ROOT>/eval/collected_results_ppl'
MODELS_DIR = '<MODELS_DIR>'

MODEL_MAP = {
    # Base model matched by family and parameter scale.
    'qwen':  os.path.join(MODELS_DIR, 'Qwen3-8B-Base'),
    'llama': os.path.join(MODELS_DIR, 'Llama-3.1-8B'),
    'gemma': os.path.join(MODELS_DIR, 'gemma-2-9b'),
}

def main():
    os.makedirs(OUTPUT, exist_ok=True)

    for model_key, model_path in MODEL_MAP.items():
        print(f"\n{'='*60}")
        print(f"加载模型: {model_key} ({model_path})")
        print(f"{'='*60}")

        tokenizer, model = load_model_and_tokenizer(model_path)

        # Find all JSONL files for this model
        model_files = []
        for root, dirs, files in os.walk(COLLECTED):
            if model_key not in root.split('/'):
                continue
            for f in files:
                if f.endswith('.json'):
                    fp = os.path.join(root, f)
                    # Skip large files that are metadata (like tree.json)
                    if os.path.getsize(fp) < 100:
                        continue
                    model_files.append(fp)

        model_files.sort()
        print(f"找到 {len(model_files)} 个文件")

        results = {}
        for i, fp in enumerate(model_files):
            rel = os.path.relpath(fp, COLLECTED)
            print(f"\n[{i+1}/{len(model_files)}] {rel}")

            try:
                avg_ppl = calculate_file_ppl(fp, tokenizer, model)  # no outlier filtering
                print(f"  PPL: {avg_ppl:.4f}")
                results[rel] = round(avg_ppl, 4) if avg_ppl != float('inf') else None
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                results[rel] = None

        # Save per-model results
        out_path = os.path.join(OUTPUT, f'{model_key}_ppl.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ {model_key} 结果已保存: {out_path}")

        # Free memory
        del tokenizer, model
        import torch
        torch.cuda.empty_cache()

    # Merge all results
    print(f"\n{'='*60}")
    print("合并结果...")
    all_results = {}
    for model_key in MODEL_MAP:
        p = os.path.join(OUTPUT, f'{model_key}_ppl.json')
        if os.path.exists(p):
            with open(p) as f:
                all_results[model_key] = json.load(f)

    merged = os.path.join(OUTPUT, 'all_ppl.json')
    with open(merged, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 合并结果: {merged}")

if __name__ == '__main__':
    main()
