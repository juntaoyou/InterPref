#!/bin/bash

PYTHON_FILE="95_t_test_analysis.py"

# if [ $# -lt 1 ]; then
#     echo "用法: $0 <file1.json> [file2.json]"
#     echo ""
#     echo "示例:"
#     echo "  比较同一文件的前两次运行:"
#     echo "    $0 result.json"
#     echo ""
#     echo "  比较两个不同文件的第一次运行:"
#     echo "    $0 result1.json result2.json"
#     exit 1
# fi

FILE1="<REPO_ROOT>/code/t-test_results copy/res_2026-05-11_11:01:32/Expertise_high_2.7.average_1.scores.json"
FILE2="<REPO_ROOT>/code/t-test_results copy/res_2026-05-11_11:18:56/None.average_1.scores.json"

if [ ! -f "$FILE1" ]; then
    echo "错误: 文件 $FILE1 不存在"
    exit 1
fi

if [ -n "$FILE2" ] && [ ! -f "$FILE2" ]; then
    echo "错误: 文件 $FILE2 不存在"
    exit 1
fi

echo "=== 开始t检验分析 ==="
echo ""

if [ -n "$FILE2" ]; then
    python $PYTHON_FILE --file1 "$FILE1" --file2 "$FILE2"
else
    python $PYTHON_FILE --file1 "$FILE1"
fi

echo ""
echo "=== 分析完成 ==="
