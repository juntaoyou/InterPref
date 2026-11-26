import os
import json

def fix_json_file(file_path):
    """将 JSON 文件中的 "answers" 键替换为 "answer" """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'answers' in data and 'answer' not in data:
                        data['answer'] = data['answers']
                        del data['answers']
                        line = json.dumps(data, ensure_ascii=False) + '\n'
                        modified = True
                except json.JSONDecodeError:
                    pass
            new_lines.append(line)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"已修改: {file_path}")
        else:
            print(f"无需修改: {file_path}")
            
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

def process_directory(directory):
    """递归处理目录下的所有 JSON 文件"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                fix_json_file(file_path)

if __name__ == '__main__':
    target_dir = '<REPO_ROOT>/llama/alpaca/answers_deactivate_preference_neurons_vary'
    print(f"开始处理目录: {target_dir}")
    process_directory(target_dir)
    print("处理完成!")
