

上传一个 Python 包（如 "Hello World" 测试包）到公共仓库 PyPI 并使用 pip 安装，主要分为准备、打包、上传三个核心阶段。
1. 准备工作

* 注册账号：在 PyPI 官网 注册一个账号。
* 生成 Token：为了安全，PyPI 推荐使用 API Token 上传包。在账号设置中生成一个，并妥善保存。
* 安装工具：确保已安装打包和上传所需的库：

pip install setuptools wheel twine


2. 构建项目结构
创建一个测试项目文件夹（例如 hello_pypi_test），其典型结构如下： [1, 3] 

hello_pypi_test/
├── my_hello_pkg/       # 你的包名（导入时的名字）
│   └── __init__.py     # 核心逻辑
├── setup.py            # 打包配置文件
├── README.md           # 项目描述
└── LICENSE             # 开源协议


* __init__.py 示例：

def say_hello():
    print("Hello from PyPI!")

* setup.py 示例：

from setuptools import setup, find_packages

setup(
    name="hello-world-test-yourname", # 包在 PyPI 上的名称，必须唯一
    version="0.1",
    packages=find_packages(),
    description="A simple hello world test package",
    author="Your Name",
)


3. 打包与上传

   1. 打包：在 setup.py 同级目录下运行命令，生成 dist/ 目录下的分发文件：
   
   python setup.py sdist bdist_wheel
   
   2. 上传：使用 twine 将生成的包上传到 PyPI：
   
   twine upload dist/*
   
   * 用户名：输入 __token__。
      * 密码：输入你之前生成的以 pypi- 开头的 Token。 
   
4. 使用 pip 安装
上传成功后（可能需要几分钟同步），你就可以在任何地方通过以下命令安装了： 

pip install hello-world-test-yourname

然后在 Python 中调用：

import my_hello_pkg
my_hello_pkg.say_hello()

小贴士：

* 包名冲突：PyPI 上的 name 必须是全球唯一的。如果提示已存在，请尝试给包名加后缀（如 hello-world-test-123）。
* 测试仓库：初次尝试建议先在 TestPyPI 进行，避免占用正式仓库资源。 




# [问题] 包名不能大小写混用


```
PS F:\github_python\assignment5-alignment\helloworldPkg> twine upload dist/*
Uploading distributions to https://upload.pypi.org/legacy/
WARNING  This environment is not supported for trusted publishing
Enter your API token: 
Uploading hello_world_test_figerZeta-0.1-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 kB • 00:01 • ?
WARNING  Error during upload. Retry with the --verbose option for more details.
ERROR    HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
         Bad Request
```

```
NFO     Response from https://upload.pypi.org/legacy/:
         400 Bad Request
INFO     <html>
          <head>
           <title>400 Filename 'hello_world_test_figerZeta-0.1-py3-none-any.whl' should contain the normalized 
         project name 'hello_world_test_figerzeta', not 'hello_world_test_figerZeta'.</title>
          </head>
          <body>
           <h1>400 Filename 'hello_world_test_figerZeta-0.1-py3-none-any.whl' should contain the normalized    
         project name 'hello_world_test_figerzeta', not 'hello_world_test_figerZeta'.</h1>
           The server could not comply with the request since it is either malformed or otherwise
         incorrect.<br/><br/>
         Filename &#x27;hello_world_test_figerZeta-0.1-py3-none-any.whl&#x27; should contain the normalized    
         project name &#x27;hello_world_test_figerzeta&#x27;, not &#x27;hello_world_test_figerZeta&#x27;.      


          </body>
         </html>
ERROR    HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
         Bad Request
```