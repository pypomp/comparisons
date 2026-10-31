initial_input_key.log 是接入脚本第一次读取冻结输入时的 KeyError('theta')。
冻结输入的实际字段为 theta_published；修正了新的检查脚本，未改输入文件或冻结模型。
这一错误发生在拟合调用前；成功执行的完整检查见 ../dacca_check.log 和 ../dacca_check.json。
