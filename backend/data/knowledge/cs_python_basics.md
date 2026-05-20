# Python 编程基础

## 变量与数据类型

Python 是动态类型语言，常见类型包括 `int`、`float`、`str`、`bool`、`list`、`dict`、`tuple`、`set`。

```python
name = "小明"
score = 95.5
passed = True
```

## 控制流

- **条件**：`if / elif / else`
- **循环**：`for item in iterable` 与 `while condition`
- **推导式**：`[x * 2 for x in range(10) if x % 2 == 0]`

## 函数

使用 `def` 定义函数，支持默认参数、可变参数 `*args`、关键字参数 `**kwargs`。

```python
def greet(name, prefix="你好"):
    return f"{prefix}, {name}!"
```

## 面向对象

```python
class Student:
    def __init__(self, name):
        self.name = name

    def study(self, subject):
        return f"{self.name} 正在学习 {subject}"
```

## 常用标准库

| 模块 | 用途 |
|------|------|
| `os` / `pathlib` | 文件与路径 |
| `json` | JSON 序列化 |
| `datetime` | 日期时间 |
| `collections` | 高级数据结构 |

## 学习建议

1. 每天编写 30 行以上代码
2. 完成 LeetCode 简单题 2 道
3. 阅读官方文档 [docs.python.org](https://docs.python.org/zh-cn/3/)
