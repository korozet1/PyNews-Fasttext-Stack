# 原始方式（不推荐）：手动维护索引
fruits = ["apple", "banana", "orange"]
i = 0
for fruit in fruits:
    print(f"索引 {i}: {fruit}")
    i += 1

# 使用 enumerate（推荐）：自动生成索引
print("--- 分割线 ---")
for idx, fruit in enumerate(fruits):
    print(f"索引 {idx}: {fruit}")