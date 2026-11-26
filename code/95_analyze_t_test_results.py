import json
import os
import re

results_dir = '<REPO_ROOT>/code/t-test-res-files/qwen/deactivate_vary/pref'

data = {}

for filename in os.listdir(results_dir):
    if filename.startswith('t_test_result_') and filename.endswith('.json'):
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            result = json.load(f)
        
        file2 = result['file2']
        mean2 = result['mean2']
        std2 = result['std2']
        
        match = re.search(r'(Expertise|Informativeness|Style)_(high|low)_([\d.]+)\.average_1\.scores\.json', file2)
        if match:
            pref_type = f"{match.group(1)}_{match.group(2)}"
            ratio = float(match.group(3))
            
            if pref_type not in data:
                data[pref_type] = {}
            data[pref_type][ratio] = {'mean': mean2, 'std': std2}

for pref_type in sorted(data.keys()):
    print(f"\n{pref_type}:")
    for ratio in sorted(data[pref_type].keys()):
        print(f"  {ratio}: mean={data[pref_type][ratio]['mean']:.4f}, std={data[pref_type][ratio]['std']:.4f}")

output_lines = []
output_lines.append("data = {")
for pref_type in sorted(data.keys()):
    means = [data[pref_type][ratio]['mean'] for ratio in sorted(data[pref_type].keys())]
    stds = [data[pref_type][ratio]['std'] for ratio in sorted(data[pref_type].keys())]
    output_lines.append(f"    '{pref_type}': {{")
    output_lines.append(f"        'LLM Score': {means},")
    output_lines.append(f"        'Std': {stds},")
    output_lines.append(f"        'Fluency': []")
    output_lines.append(f"    }},")
output_lines.append("}")

print("\n" + "\n".join(output_lines))

with open('<REPO_ROOT>/code/updated_data.py', 'w') as f:
    f.write("\n".join(output_lines))

print("\n数据已保存到 updated_data.py")