
import numpy as np

def psnr(ref: np.ndarray, rx: np.ndarray, max_val=255.0) -> float:
    ref = ref.astype(np.float32)
    rx = rx.astype(np.float32)
    mse = np.mean((ref - rx) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)

def cer(ref_text: str, rx_text: str) -> float:
    a, b = ref_text, rx_text
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len(a)][len(b)] / max(1, len(a))
