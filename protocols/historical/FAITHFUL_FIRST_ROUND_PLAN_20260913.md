# 第一轮：J/M/F 均值训练机制对照

2026-09-13；开发实验。依据上级行动计划 `../OPAL2_ACTION_PLAN_20260913.md` 执行阶段 0 和阶段 1。本文件在新对照训练前写定。

## 问题与实际实现

检验以物理测量空间 MSE 保护均值的训练机制，能否改善当前联合概率模型的均值预测。不是测试新生物知识、JEPA、厚尾或新收益终点。

当前 A 臂实际训练目标为精确 predictive NLL，不是 ELBO。当前方差相关参数还通过潜变量精度更新、rho/correlation 描述量和共享 loading 影响均值，不能直接把这些原参数当作均值无关的噪声参数。

因此保留完整 MeasurementWorldModel 作为均值主干，复制其最终 compound/source/batch/plate、within-well 与 diagonal 协方差解码头。原主干中的相应参数继续服务于均值计算；新的协方差头服务于联合预测方差。潜变量与条件信息不删除，保留全部 3,617 个测量坐标。

J、M、F 均从相同的这个参数化初始化。J 是“解绑定参数化下的匹配联合训练”，不是旧 A 的原样复跑。旧 A 与已修正的完整 L 只能作为明确标注的历史参照。

| 条件 | 均值参数梯度 | 独立协方差参数梯度 |
| --- | --- | --- |
| J | 精确联合预测 NLL | 同一 NLL |
| M | 原空间均值 MSE | 不更新；不把其未训练分布用于比较 |
| F | 与 M 完全相同的 MSE | NLL；mean、state、posterior variance 在此路径停止梯度 |

三臂均关闭 reference reconstruction 辅助损失，化学 KL 正则保持原值 0；参考、化学、library 和测量规律输入全部保留。这样不会把辅助损失对均值的更新混入 M/F 一致性。此实验不裁决参考辅助损失是否有益。

## 固定范围与配方

- 仅 `data/source5_primary_fullcontrols`：639 个已开放 DEV、四个角色，完整 3,617 维。
- 原 TRAIN/validation/calibration/evaluation：383/96/64/96；不重划、不删异常对象。
- 不访问第五重复、FINAL、未开放候选、旧 custody 或外部源缓存。
- 无新 JEPA；生物机制先验及新增 biological kernel 关闭，与已有 A 主配置一致；measurement kernel 保留。
- seed=20260912；hidden=256；group attention layers=2、heads=4；latent rank=32、residual rank=8。
- AdamW，learning rate=0.0003，weight decay=0.0001；cosine 退火到 0.000003；warmup=60 steps。
- batch=32，12 batches/epoch，100 epochs、每臂 1,200 steps。不使用各臂独立 early stopping。
- CPU，4 threads。当前精确 float64 联合似然不切到 MPS。
- 每批 J/M/F 共用相同训练对象、随机角色、上下文数量和原 10% zero-context 设置；各臂匹配模型随机流。
- 均值与协方差使用独立优化器状态、分别按 norm=5 裁剪。记录裁剪前梯度范数与裁剪比例。
- MSE 是物理空间的全部已观测目标坐标平均；NLL 按已观测坐标数归一化。记录两者数值尺度，不为某臂事后单独改学习率。

## 实现检查

训练前完成以下针对性检查，不把检查样本作为实验证据：

1. 新参数化初始化时，均值及全部协方差分量与原完整模型一致。
2. F 的 NLL 单独反传不会给任何均值主干参数梯度。
3. 同批、同优化器设置的 M/F 多步均值参数和实际预测保持一致。
4. 缺失掩码及层级共享随机分量行为保留；不确定性参数更新不改变实际均值。
5. 真实 TRAIN 的完整维度入口可执行且损失/梯度有限；据实测速度修正 ETA。

## 报告节点与选择规则

三臂按 minibatch 交错执行，避免先跑完 J 才开始 M/F。

- 每 5 epochs 计算固定角色 validation 的原空间均值 MSE。
- 同一预定前缀 **20 epochs/240 steps** 输出早期配对均值报告。cosine 总日程仍是 1,200 steps，不压缩到 240 steps。
- 三臂随后共同继续至 100 epochs/1,200 steps。早期弱结果不作为提前判死依据。
- 主要比较：三臂固定第 100 epoch；20 epoch 是早期机制诊断。每臂 validation-MSE 最佳 checkpoint 另列辅助结果，不能替代主比较。
- 首轮结束只分析阶段 1；不自动启动 FC/FCΓ、五折、新种子或 kernel 扩搜。未来阶段 2 如启动，从预先声明的 F 第 100 epoch checkpoint 分叉。
- 每 20 epochs 检查 M/F 全均值主干参数一致性。若不一致或非有限，停止并报告实现问题，不把它解释为科学结果。

训练日志逐 epoch 保存，但不按 epoch 推送给用户。运行时估算首个 20-epoch 报告与完整 100-epoch 完成时间；定时跟进只按用户约定的一次检查执行，届时如未完成，说明实际进度并重新估时，不高频轮询。

## 第一轮指标与交付

20 和 100 epoch 时报告原 TRAIN、validation、evaluation、calibration 上的精确决策时均值：物理空间及标准化 MSE/R²、逐对象误差、极端对象贡献、完整逐对象预测。原结果仍然属于已打开 DEV 上的诊断，不是新认证。

第一份报告不等待 2,000 次联合采样全表。均值结果只回答均值机制，不能宣称孔差、孔平均、Γ、NULL 或策略已修好。这些是下一阶段的完整评价内容。

历史 L 只读取此前相同对象/空间/指标的结果，不重新拟合或更换其定义。原收益、七项合同及旧结果不修改。

需要区分的解释：

- M 优于 J、F 跟上 M：支持以 MSE 保护均值的训练机制有效，但不单凭此声称证实了某一个梯度方向冲突。
- M/F 与 J 都弱：该机制不足以解释差距，不能推论所有非线性模型都无效。
- F 均值改善但仍低于 L：机制修复有迹象，尚无神经模型性能优势。
- F/M 均值轨迹不一致：先修实现，不报告方法优劣。

## 后台资源

用户已批准暂停旧 biological-kernel 训练，优先新对照。旧 PID 66485 于本轮收到 SIGSTOP，进程与已写结果保留，可用 SIGCONT 恢复。新对照采用独立运行目录、日志和 checkpoint，不覆盖旧运行。
