# 时频重分配与低能量声源：代码路径分析与可复算实验

- 分析对象：本仓库 `librosa` 1.1.0dev（分支 `task/127`，HEAD `087a860`）
- 核心代码：`librosa/core/spectrum.py`
  - `__reassign_frequencies`（`spectrum.py:638`）
  - `__reassign_times`（`spectrum.py:799`）
  - 公开入口 `reassigned_spectrogram`（`spectrum.py:979`，`@cache(level=20)`）
- 实测环境：Python 3.12.3、numpy 2.5.5、scipy 1.18.1、numba 0.68.1；全部信号均为内存合成，复算脚本见文末附录。

---

## 1. 两条重分配路径在代码中的实际计算

### 1.1 频率重分配（瞬时频率，IF）

`__reassign_frequencies`（`spectrum.py:744-796`）做三次确定的运算：

1. 用原窗 `h` 做一次 STFT 得到 `S_h`（外部传入 `S` 时直接复用）；
2. 用 `util.cyclic_gradient(window)`（`util/utils.py:2200`，环形 wrap 差分）求窗的离散导数 `h'`，再做一次 STFT 得到 `S_dh`；
3. 按 Flandrin, Auger & Chassande-Mottin (2002) 式 5.20 计算校正量：

```python
# spectrum.py:789-794
with np.errstate(invalid="ignore"):
    correction = -np.imag(S_dh / S_h)          # 单位：弧度/样本
freqs = fft_frequencies(...) + correction * (0.5 * sr / np.pi)
```

即

```
ω_reassigned = ω_bin − Im(S_dh / S_h)
f_reassigned = f_bin + correction × sr/(2π)
```

输出的是**每个 (bin, frame) 单元各自的瞬时频率估计**，单位 Hz，而不是新的频谱网格。

### 1.2 时间重分配（局域群延迟/事件时间）

`__reassign_times`（`spectrum.py:906-975`）：

1. 同样先得到 `S_h`；
2. 构造以窗中心为零点的时间权重 `window_times`（偶数 `n_fft` 时从 `0.5 − n_fft/2` 开始，`spectrum.py:937-944`），得到时间加权窗 `t·h`，做 STFT 得 `S_th`；
3. 按式 5.23：

```python
# spectrum.py:961-973
with np.errstate(invalid="ignore"):
    correction = np.real(S_th / S_h)           # 单位：样本
times = frames_to_time(frame_index, ...) + correction / sr
```

即

```
t_reassigned = t_frame + Re(S_th / S_h) / sr
```

两个比值都是**复数除法**。当 `S_h[f,t] = 0`（该单元没有任何支撑）时，IEEE 754 下 0/0 得到 `NaN`；`np.errstate(invalid="ignore")` 使该运算不产生 `RuntimeWarning`（实测见 §5.6，与函数 docstring 中“will produce a divide-by-zero warning”的描述不符——这是文档与实现的偏差，不是数值异常）。

### 1.3 公开入口的后处理顺序

`reassigned_spectrogram`（`spectrum.py:1180-1284`）严格按以下顺序处理，顺序本身是源码保证的行为：

1. 按开关调用上述两个私有函数（两者都为 `False` 直接抛 `ParameterValueError`，`spectrum.py:1183`）；
2. `mags = np.abs(S)`——**幅度从不被阈值修改，也从不为 NaN**；
3. 计算低能量掩膜（详见 §2）；
4. 对参与重分配的坐标，把掩膜单元置 NaN；
5. `fill_nan=True` 时用 bin 中心频率 / 帧时间替换 NaN；
6. `clip=True`（默认）时把频率裁到 `[0, sr/2]`、时间裁到 `[0, len(y)/sr]`。

注意 NaN 先于 clip 产生：`np.clip` 对 NaN 不做处理，因此“无支撑单元为 NaN”在 clip 之后仍然成立。

---

## 2. 幅度阈值与坐标有效性的分界

### 2.1 阈值是**绝对幅度**阈值，且默认值固定

参数名虽叫 `ref_power`，实际判定代码是（`spectrum.py:1247-1251`）：

```python
ref_p   = ref_power(mags**2) if callable(ref_power) else ref_power
mags_low = np.less(mags, ref_p**0.5, where=~np.isnan(mags), out=None)
```

- 默认 `ref_power=1e-6` ⇒ 实际分界是 **幅度 `|S| < 0.001`**（功率 1e-6 开平方）；
- 这是一个与信号增益、单位刻度绑定的**绝对**分界，不是相对最大能量的 dB 门限；
- `ref_power=0` 时跳过阈值掩膜这一步（`if ref_p > 0`，`spectrum.py:1257,1272`），但零支撑单元仍因 0/0 而为 NaN；
- 传入可调用对象（如 `ref_power=np.max` 或 `lambda P: c*np.max(P)`）时，门限随信号整体能量浮动，才能得到相对阈值。

### 2.2 分界两侧的语义截然不同

| 单元状态 | 坐标（freqs / times） | 幅度 mags | 可解释性 |
|---|---|---|---|
| `|S| ≥ 0.001` 且非零 | 重分配估计值（可能已被 clip） | 原始 `|S|` | “存在一个满足单分量假设的估计”，**不等价于确认的声学事件** |
| `|S| < 0.001`（默认设置） | 强制 NaN（`fill_nan=False`） | 仍返回原始小幅度 | 源码主动声明“此处坐标无效” |
| `S = 0`（零支撑） | 除法直接产生 NaN | 0 | 无任何局域能量，无坐标可言 |
| `fill_nan=True` 的上述单元 | bin 中心频率 / 帧时间（网格值） | 同上 | 网格占位，**不是测量结果** |

阈值的后果只作用于坐标数组，`mags` 保留全部动态范围。因此不能用 `mags` 判断某个散点是否通过了门限——必须检查对应坐标是否为 NaN。

### 2.3 绝对门限对“弱能量声源”的实际含义

同一个弱声源，物理信噪比不变，仅改变整体增益，就可能从“有效”翻转为“NaN”。实测（§5.1）：

- 信号 `sin(2π·200t) + 5e-5·sin(2π·400t)`，`sr=2000, n_fft=256, center=False`；
- 增益 1 时，400 Hz 弱音单元幅度约 `4.34e-3 > 1e-3`，坐标有效；
- 整体衰减 1000 倍后，弱音单元幅度约 `4.34e-6 < 1e-3`，**坐标变为 NaN**，尽管它相对强音的 −86 dB 关系完全没有变；
- 改用 `ref_power=lambda P: 1e-10*np.max(P)` 后，两种增益下弱音坐标都有效。

结论：默认门限衡量的是“STFT 幅度在当前数值刻度下是否足够大”，而不是“该单元是否对应真实声源”。低电平录音、归一化前的整型 PCM 缩放、不同窗长造成的幅度缩放，都会改变同一个声源的有效性判定。

---

## 3. 单坐标模式与帧定位变化时，另一个坐标和幅度怎么理解

### 3.1 只重分配一个坐标（`reassign_frequencies=False` 或 `reassign_times=False`）

实测（§5.4）与代码（`spectrum.py:1229, 1268-1282`）一致：

- 被关闭的坐标返回 `np.broadcast_to` 生成的**只读广播视图**：
  - 关闭频率 ⇒ `freqs[f,t] ≡ fft_frequencies()[f]`（所有帧相同）；
  - 关闭时间 ⇒ `times[f,t] ≡ frames_to_time(t)`（所有 bin 相同）；
  - 视图不可写，写入会抛 `ValueError: assignment destination is read-only`。
- 仍参与重分配的那个坐标**照常做除法、照常受 `ref_power` 掩膜**，因此会出现“一个坐标是 NaN、另一个坐标是有效网格值”的单元。例如 `reassign_times=False` 时，低能量单元的 `freqs` 为 NaN 而 `times` 为规则帧时间。
- `mags` 行为完全不变，永远是原始幅度。

解释规则：

1. 散点的可信度由**两个坐标共同决定**——只要任一坐标为 NaN，该点就不构成有效的时频定位；单坐标模式下必须自行用 `np.isfinite` 同时检查两者。
2. 关闭某坐标得到的是**分析网格的几何位置**，不是该能量的测量位置。把 `reassign_times=False` 的时间轴当作“事件时间”会退回到普通 STFT 的时间分辨率（一帧之内无法区分事件）。
3. docstring（`spectrum.py:1149-1153`）指出：关闭时间重分配时，时间坐标按 `center` 左对齐或居中对齐；此时若 `center=True`，边界帧的 IF 估计需要有效的填充策略。

### 3.2 `center=True/False` 对时间坐标的影响

帧时间基准来自 `frames_to_time(..., n_fft=pad_length)`（`spectrum.py:965-971, 1230-1243`）：

- `center=True`：`pad_length=None`，帧时间为 `t·hop/sr`，帧 `t` 以样本 `t·hop` 为**中心**；
- `center=False`：`pad_length=n_fft`，帧时间额外加 `n_fft/2` 样本偏移，帧 `t` 以样本 `t·hop` 为**左端点**，报告的是该帧的几何中心时间。

实测（§5.2）两种约定下，同一帧号的标称时间相差恰好 `n_fft/(2·sr)`（示例中 0.0625 s）。

但对**已经重分配的时间**而言，校正是相对于各自帧基准再加上 `Re(S_th/S_h)/sr`，而 `S_th` 的时间权重以窗中心为零点，所以同一个冲击在两种 `center` 下最终都收敛到它的真实样本时间（实测误差 2.4e-4 s，约 0.5 个样本）。差异只体现在：

1. 事件落在哪个**帧号**里（帧编号整体平移）；
2. 关闭时间重分配时，报告的是哪种网格时间；
3. 边界帧：`center=True` 默认 `pad_mode="constant"` 做零填充，边界帧混入人工不连续；`center=False` 不填充，直接丢掉不完整帧。docstring 因此推荐 `center=False`（`spectrum.py:1142-1147`）。

### 3.3 幅度始终是“原 STFT 的幅度”

无论两个开关、`fill_nan`、`clip` 如何设置，`mags = np.abs(S_h)` 在阈值处理之前计算且之后不再改动。这意味着：

- 幅度给的是**未重分配网格上的能量**，重分配只移动坐标、不搬移能量；
- 一个 NaN 坐标单元仍可能携带不可忽略的 `mags`（它只是没过门限）；
- 若下游用 `fill_nan=True` 再做幅度加权统计（这正是 `feature/spectral.py:149,288` 文档示例的做法），低能量噪声会以网格坐标**带着权重进入统计量**。实测纯噪声输入（§5.9）照样算出 532 Hz 这样“看起来确定”的谱质心。

---

## 4. 源码保证的行为 vs 数值估计的不确定性

### 4.1 源码可以保证的（结构性事实）

| 保证 | 依据 |
|---|---|
| 校正公式、单位换算（rad/样本→Hz；样本→秒） | `spectrum.py:789-794, 961-973` |
| 零支撑与过门限的单元坐标为 NaN（默认 `fill_nan=False`） | 复数除法 + `spectrum.py:1256-1258, 1271-1273` |
| `mags` 恒为原始 `|S|`，不被阈值/填充/裁剪改动 | `spectrum.py:1223` |
| 单坐标关闭时返回 bin/帧网格的只读广播视图 | `spectrum.py:1269, 1282`；实测只读 |
| `clip=True` 时坐标落在 `[0, sr/2]` 与 `[0, 时长]` 内 | `spectrum.py:1263-1264, 1278-1279`；测试 `tests/test_core.py:454-487` |
| `ref_power<0`、两个开关全 False 时抛错 | `spectrum.py:1180-1184`；测试 `test_core.py:608-615` |
| 冲击信号：含冲击帧内所有 bin 的重分配时间收敛到冲击样本，空帧全 NaN | 测试 `test_core.py:414-451`（容差 0.5% 窗长），实测复现 |
| 稳态单分量信号：IF 收敛到分量频率（测试容差 3 Hz） | 测试 `test_core.py:395-408`，实测孤立 100 Hz 音 → 100.0 Hz |
| 窗导数用环形差分，与 FFT 周期窗的边界约定一致 | `util/utils.py:2247-2259` |

### 4.2 源码不保证、属于数值估计的部分

1. **单分量假设**。docstring 明确：每个 FFT bin 的能量必须只对应“一个信号分量与一个冲击事件”（`spectrum.py:1145-1147`）。实测两个相距仅 1 Hz、落入同一 bin 的等幅正弦（§5.3），重分配 IF 稳定停在 100.46 Hz，**两个分量（100、101 Hz）都不是**；而孤立的 100 Hz 音给出 100.0 Hz。这种偏差没有任何告警。
2. **窗内多事件**。同一帧窗口内含两个冲击时，时间重分配得到的是加权混合位置（测试注释 `test_core.py:411` 同样承认“longer windows containing multiple impulses will be unstable”）。
3. **边界帧失真**。`center=True` + 零填充在信号两端制造人工不连续，实测 200 Hz 纯音第 0 帧 IF 偏到 201.82 Hz，而 `center=False` 第 0 帧即为 200.0 Hz（§5.5）。失真范围有多大、是否可接受，取决于信号本身，源码不量化。
4. **弱单元的比值噪声**。`S_dh/S_h`、`S_th/S_h` 在 `|S_h|` 接近门限时由两个小数相除得到，相对误差大。实测宽带噪声上，未经 clip 的原始 IF 出现 −977 Hz、3254 Hz（奈奎斯特仅 2000 Hz）等越界值，约 0.37% 的有限单元越界；时间估计越界约 0.017%（§5.7）。
5. **clip 会掩盖越界**。默认 `clip=True` 把这些值压到 0、`sr/2`、0、时长四条边上；实测一次噪声分析中频率轨上有 2236 个单元恰好贴边。**贴边聚集是估计失效的征兆，不是真实存在 DC/奈奎斯特/零时刻分量的证据**；需要诊断时应设 `clip=False`。
6. **精度依赖输入 dtype**。float32 输入走 complex64 STFT，float64 走 complex128。实测同一弱音两种精度的 IF 差约 6e-6 Hz（本例差异可忽略），但越接近门限，除法的舍入差异越大；源码不提供误差棒或置信度。
7. **频谱泄漏偏差**。短窗下弱音 bin 受邻近强音泄漏污染，实测 400 Hz 弱音在 `n_fft=256` 时被拉到 394.9 Hz；加长窗到 1024 后偏差降到 0.34 Hz。窗长选择是偏差与时间定位能力之间的折中，不由源码裁决。

---

## 5. 可复算的简短信号情形

复算方式（脚本同时保存于 `/srv/fb2/feng0732/scratch-reassign-127/probe_reassign.py` 与 `probe_extra.py`）：

```bash
/srv/fb2/feng0732/venv-librosa/bin/python -I /srv/fb2/feng0732/scratch-reassign-127/probe_reassign.py
```

下表每个情形都可独立复制；除 E5/E7 外统一用 `center=False` 排除填充干扰。

### 5.1 绝对阈值随增益翻转（对应 §2.3）

```python
sr, n_fft = 2000, 256
t = np.arange(2*sr)/sr
y = np.sin(2*np.pi*200*t) + 5e-5*np.sin(2*np.pi*400*t)
for g in (1.0, 1e-3):
    f, tt, m = librosa.reassigned_spectrogram(y=g*y, sr=sr, n_fft=n_fft,
                                              hop_length=n_fft, center=False)
    # g=1   : 400 Hz 单元 |S|≈4.34e-3 → 坐标有效
    # g=1e-3: |S|≈4.34e-6 < 1e-3       → 坐标 NaN，SNR 未变
```

### 5.2 冲击的时间定位与 center 约定（对应 §3.2）

```python
sr, n_fft = 2048, 256
y = np.zeros(4096); y[300] = 1.0
tt, _ = librosa.core.spectrum.__reassign_times(y, sr=sr, n_fft=n_fft,
                                               hop_length=n_fft, center=False)
# 含冲击帧所有 bin → 0.146729 s（真实 0.146484 s，误差 < 0.5% 窗长）；
# 无冲击帧全部 NaN。center=True/False 的重分配结果一致，帧号与网格基准平移 n_fft/2。
```

### 5.3 同 bin 双分量导致 IF 失真（对应 §4.2 之 1）

```python
sr, n_fft = 512, 128                       # bin 间距 4 Hz
t = np.arange(4*sr)/sr
y = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*101*t)
f, _ = librosa.core.spectrum.__reassign_frequencies(y, sr=sr, n_fft=n_fft,
                                                    hop_length=n_fft, center=False)
# bin 25（中心 100 Hz）的 IF ≈ 100.46 Hz，不等于任一分量；孤立 100 Hz 音 → 100.0 Hz
```

### 5.4 单坐标关闭（对应 §3.1）

```python
f, tt, m = librosa.reassigned_spectrogram(y, sr=sr, n_fft=n_fft,
                                          hop_length=n_fft, center=False,
                                          reassign_times=False)
# f 仍含 NaN；tt 为每 bin 相同的帧中心时间、只读；m 无 NaN
```

### 5.5 零填充边界帧偏差（对应 §4.2 之 3）

200 Hz 纯音、`n_fft=256`：`center=True`（默认零填充）第 0 帧 IF = 201.82 Hz，内帧 200.0 Hz；`center=False` 第 0 帧即 200.0 Hz。

### 5.6 零信号与 docstring 偏差

```python
f, tt, m = librosa.reassigned_spectrogram(y=np.zeros(2048), sr=64,
                                          n_fft=64, hop_length=64, center=False)
# f、tt 全部 NaN，m 全部 0；实测不产生 RuntimeWarning（errstate 已抑制）
# ref_power=0 时坐标仍全 NaN：0/0 的 NaN 先于阈值逻辑
```

### 5.7 clip 与越界估计（对应 §4.2 之 4、5）

`σ=1e-2` 的高斯白噪声、`sr=4000, n_fft=128`：私有函数输出的原始 IF 范围 −977～3254 Hz；默认 `clip=True` 后 2236 个单元贴在 0 或 2000 Hz 轨上。`clip=False` 可暴露越界。

### 5.8 fill_nan 的网格占位本质

同一噪声输入 `fill_nan=True`：NaN 计数降为 0，但这些单元取的是 bin/帧网格值，不是新测量；`mags` 不变。

### 5.9 噪声被下游统计“实体化”（对应 §3.3）

```python
y = np.random.default_rng(0).standard_normal(2*sr) * 1e-5
f, _, m = librosa.reassigned_spectrogram(y, sr=sr, n_fft=n_fft,
                                         hop_length=n_fft, center=False,
                                         fill_nan=True)
centroid = np.sum(f*m**2, axis=0) / np.sum(m**2, axis=0)
# 纯噪声也得到有限、平滑的“谱质心”（实测第 2 帧 532.0 Hz）
```

### 5.10 文档建议在本版本不可直接使用

`__reassign_frequencies` 与公开函数的 docstring 仍建议 `pad_mode="wrap"`，但本仓库 `stft` 在 `center=True` 下显式拒绝该模式（`spectrum.py:260-270`，有测试 `tests/test_core.py:3269` 锁定该行为），调用即抛 `ParameterValueError`。当前可行的边界处理只有默认零填充或 `center=False`。

---

## 6. 仍需实测判断的事项（源码无法替你决定）

1. **门限怎么设**：默认 1e-3 幅度门限是否合适，取决于录音增益与单位刻度。弱声源分析前应先实测本数据中目标单元的 `|S|` 分布，必要时改用可调用的相对门限；不能假设默认值具有物理意义。
2. **事件确认需要跨证据**：重分配坐标有效只说明“该 bin 内有一个可拟合的局域分量”。确认弱声学事件至少还需：相邻帧/相邻 bin 的时频连续性、与参考本底噪声的幅度对比、必要时多窗长交叉验证。单帧单点的有效坐标不足以定案。
3. **窗长与 hop**：窗长决定频率选择性与多分量/多事件冲突概率（§4.2 之 1、2、7），hop 决定散点密度；目标声源的调制速率与频率间距不同，最优取值必须实测。
4. **边界帧的取舍**：丢弃多少首尾帧（`center=False`）或容忍多少零填充偏差（`center=True`），取决于信号两端是否含关键事件，需按数据检查。
5. **clip 诊断**：若怀疑估计失效，先用 `clip=False` 复跑并统计越界比例与贴边聚集；贴边点应从解释中剔除而不是当作 DC/奈奎斯特分量。
6. **dtype 与门限的交互**：float32 流水线在门限附近的稳定性应与 float64 对照实测，尤其是上采样/下采样或归一化之后。
7. **fill_nan 的下游风险**：只有在下游运算无法接受 NaN（如加权谱特征）时才使用，并应同时配合显式幅度门限做加权衰减；否则优先保留 NaN，让绘图（matplotlib `scatter` 自动跳过 NaN）和统计自然忽略无效单元。

### 防止把低能量噪声定位为确定事件的操作清单

- 以 `np.isfinite(freqs) & np.isfinite(times)` 作为有效点的**必要**条件，而非充分条件；
- 不使用 `fill_nan=True` 的输出做事件判读（它只适合喂给不能处理 NaN 的加权特征）；
- 对候选弱事件检查多帧连续性与幅度显著性，而不是只看重分配图上的锐利点——重分配的“锐利”本身是算法几何效应，噪声单元只要过门限一样会被聚成尖点；
- 解释前用 `clip=False` 检查越界/贴边比例，评估本参数下估计失效率；
- 记录并报告所用 `ref_power`、增益刻度与 dtype，因为默认绝对门限下这些选择直接决定坐标有效性的分界。

---

## 附：关键代码位置索引

| 内容 | 位置 |
|---|---|
| 频率校正公式与单位换算 | `librosa/core/spectrum.py:786-796` |
| 时间校正公式与帧基准 | `librosa/core/spectrum.py:958-975` |
| 阈值掩膜（绝对幅度判定） | `librosa/core/spectrum.py:1245-1251` |
| NaN 填充、clip、单坐标网格视图 | `librosa/core/spectrum.py:1229-1284` |
| 幅度恒为原始 `|S|` | `librosa/core/spectrum.py:1223` |
| 环形窗导数 | `librosa/util/utils.py:2247-2259` |
| 帧时间的 center 偏移 | `librosa/core/convert.py:123-127` |
| `pad_mode="wrap"` 被拒 | `librosa/core/spectrum.py:260-270` |
| 行为级测试（容差、NaN、clip、flags） | `tests/test_core.py:392-615` |
| 缓存级别（默认 level 10，level 20 默认不激活） | `librosa/_cache.py:42-55,90` |
