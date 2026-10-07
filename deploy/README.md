# Smash_Potato_HR v3(混合最优）— 提交说明 & 注意点

dPickleBall 球拍智能体。**v3 = 逐槽取最强的混合档**:左右两侧各装"赢下该侧真机门"的那个检查点。

| 槽位 | 模型 | 真机门(vs router,serve 212) |
|---|---|---|
| 左 teamX(`model_left.pt`) | `realft_01600004`(1.6M PFSP 微调) | **4/4,平均 21:9.8** |
| 右 teamY(`model_right.pt`) | `realft_00300000` 镜像(300k 基线镜像) | **3/3,平均 21:13.3** |

合计 **7 场全胜、零败**,两侧都是各自侧别的真机最强(优于全 1600k 的 v2 和全 realft 的 v1)。

## 文件清单
| 文件 | 作用 |
|---|---|
| `teamX.py` / `teamY.py` | 左/右选手类(自包含,内嵌 OpenCV 特征提取) |
| `model_left.pt` | 左拍 = 1600k(TorchScript) |
| `model_right.pt` | 右拍 = 300k 基线的镜像(TorchScript) |
| `README.md` | 本说明 |

> 注意:**v3 左右是不同检查点**(左 1600k、右 300k 镜像),这是有意为之——1600k 微调强了左侧,但镜像到右侧反弱;300k 的镜像在右侧反而更强。teamX/teamY 各自独立加载,互不相关,故可混搭取最优。

## 怎么跑
1. 4 个文件(两 `.py` + 两 `.pt`)放进竞赛脚本目录(`Competition.py` 旁边)。
2. `Competition.py` 导入 `TeamX`(左)/`TeamY`(右),各自加载同目录的 `model_left.pt`/`model_right.pt`。
3. 照常运行。**CPU 即可**,依赖 `torch / numpy / opencv-python`,约 1ms/step,确定性 argmax。

## 注意点(重要)
1. **旋转动作恒为 0**(有意为之,角度盲模型;看到拍子不转是正常)。
2. **两侧靠 `side_sign` 区分**(TeamX→feature[14]=-1,TeamY→+1),不是靠 observation;已兼容官方只传 agent0 视觉的方式。
3. **视觉识别依赖标准竞赛画面**(168×84、特定 HSV 阈值);分辨率/配色不同会掉识别。
4. **左右不同档**:左 1600k(攻击性强)、右 300k 镜像(均衡)。这是逐槽真机门选出的最优组合。
5. **选档只信真机门**(真 dp.exe 对打),不信 sim 指标——这是我们分支的做法。

## 血统
13 维角度盲学生(V54 监督蒸馏)→ 真机 PPO 微调 300k(= 右拍来源 realft_00300000)→ PFSP 对手池真机微调到 1.6M(= 左拍来源 realft_01600004)。右拍 = 各自左模型绕中线镜像(`tools/export_mirror_right.py`,反射恒等式验零误差,旋转0)。
