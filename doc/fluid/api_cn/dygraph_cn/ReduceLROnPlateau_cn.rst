.. _cn_api_fluid_dygraph_ReduceLROnPlateau:
    
ReduceLROnPlateau
-------------------------------

**注意：该API仅支持【动态图】模式**

.. py:class:: paddle.fluid.dygraph.ReduceLROnPlateau(learning_rate, mode='min', decay_rate=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, dtype='float32')

该接口提供与 ``loss`` 相关的学习率衰减策略，当 ``loss`` 停止提升时，降低学习率。其基于的原理是：一旦学习停滞不前，将学习率降低2-10倍对模型的训练往往有益。

``loss`` 是 ``optimizer.minimize`` 的对象，且必须是shape为[1]的1-D Tensor。
在 `'min'` 模式下，如果 ``loss`` 停止下降超过 ``patience`` 个epoch，学习率将会减小为 ``learning_rate * decay_rate`` 。 
在 `'max'` 模式下，如果 ``loss`` 停止上升超过 ``patience`` 个epoch，学习率也将减小为 ``learning_rate * decay_rate`` 。

此外，在学习率减小之后，重新恢复正常操作会等待 ``cooldown`` 个epoch，在该等待期间，将无视 ``loss`` 的变化情况。

参数：
    - **learning_rate** (Variable|float|int) - 初始学习率。其类型可以是Python的float或int类型。也可以是shape为[1]的
      1-D Tensor，且相应数据类型为"float32" 或 "float64"。
    - **mode** (str，可选) - 'min' 和 'max' 之一。在'min'模式下，当 ``loss`` 停止下降时，学习率将减小；在'max'模式下，
      当 ``loss`` 停止上升时，学习率将减小。默认：'min'。
    - **decay_rate** (float，可选) - 学习率衰减的比例。 new_lr = origin_lr * decay_rate，它是值小于1.0的float型数字，默认: 0.1。
    - **patience** (int，可选) - 当监控指标连续 ``patience`` 个epoch没有提升时，学习率才会衰减。默认：10。
    - **threshold** (float，可选) - ``threshold`` 和 ``threshold_mode`` 两个参数将会决定 ``loss`` 最低提升的阈值。小于该阈值的提升
      将会被无视，这使得仅有较大的提升会被关注。默认：1e-4。
    - **threshold_mode** (str，可选) - 'rel' 和 'abs' 之一。在'rel'模式下， ``loss`` 的最低提升阈值是 ``last_loss * threshold`` ，
      其中 ``last_loss`` 是 ``loss`` 在上个epoch的值。在'abs'模式下， ``loss`` 的最低提升阈值是 ``threshold`` 。 默认：'rel'。
    - **cooldown** (int，可选) - 在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。默认：0。
    - **min_lr** (float，可选) - 最小的学习率。减小后的学习率最低下界限。默认：0。
    - **dtype** (str，可选) – 学习率值的数据类型，可以为"float32", "float64"。默认："float32"。

返回：与 ``loss`` 的提升相关的学习率

返回类型：Variable

**代码示例**：

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)
            
            adam = fluid.optimizer.Adam(
                learning_rate = fluid.dygraph.ReduceLROnPlateau(
                                    learning_rate = 1.0,
                                    decay_rate = 0.5,
                                    patience = 5,
                                    cooldown = 3),
                parameter_list = linear.parameters())

            for epoch in range(20):
                out = linear(input)
                loss = fluid.layers.reduce_mean(out)
                adam.minimize(loss)
                lr = adam.current_step_lr()
                print("current lr loss is %s, current lr is %s" % (loss.numpy()[0], lr))