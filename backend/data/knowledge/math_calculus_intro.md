# 微积分入门

## 极限

数列极限：$$\lim_{n \to \infty} a_n = A$$

函数极限：$$\lim_{x \to x_0} f(x) = L$$

### 两个重要极限

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e$$

## 导数

导数定义：$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$$

### 求导法则

| 函数 | 导数 |
|------|------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |

链式法则：$(f(g(x)))' = f'(g(x)) \cdot g'(x)$

## 导数应用

1. **单调性**：$f'(x) > 0$ 递增，$f'(x) < 0$ 递减
2. **极值**：导数变号点
3. **最值**：比较端点与驻点

## 不定积分

$$\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)$$

$$\int \frac{1}{x} \, dx = \ln|x| + C$$

## 定积分

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

几何意义：曲边梯形面积。

## 学习路径

极限 → 导数 → 积分 → 微分方程（大学）
