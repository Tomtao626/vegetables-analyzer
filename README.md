# 边缘AI应用：果蔬AI识别（vegetables_analyzer）

- [x] 第一阶段：识别萝卜、茄子、玉米
- [x] 第二阶段：KubeEdge对模型升级，添加对西红柿的识别

## 开发环境（根据自己的情况适配以下环境即可）

- 操作系统：MacOS
- Python版本：Python3.7+（尽量保持一致）
- 代码编译器：Pycharm

## 数据集准备

### 第一阶段

- 训练集：萝卜、茄子、玉米各1500张
- 验证集：萝卜、茄子、玉米各500张
- 测试集：萝卜、茄子、玉米各100张


### 第二阶段

- 训练集：萝卜、茄子、玉米、番茄各1500张
- 验证集：萝卜、茄子、玉米、番茄各500张
- 测试集：萝卜、茄子、玉米、番茄各100张

### Docker镜像打包&测试

docker 打包命令
```bash
docker build -t vegetables_analyzer:v1 .
```

docker 启动(测试)

```bash
docker run -d --name vegetables_analyzer --rm -p5000:5000 vegetables_analyzer:v1
```

docker 测试是否启动

```bash
curl -H "Content-Type:multipart/form-data" -F "file=@/yourpath/tmp.png" localhost:5000/analyzer
```

docker 推送

```bash
docker tag vegetables_analyzer:v1 edge.imooc.com/imooc_containers/vegetables_analyzer:v1
docker push edge.imooc.com/imooc_containers/vegetables_analyzer:v1
```
