---
layout:     post
title:      "「计算机网络」 SMTP邮件Python脚本"
subtitle:   "SMTP e-mail Python script"
date:       2024-05-14 12:00:00
author:     "Mingyu"
header-img: "img/post-bg-computer-network.jpg"
katex: true
tags:
    - 计算机网络
    - 办公脚本
---





考虑一个批量处理文件并且发送到指定邮箱的任务，借助Python脚本可以较为简洁地实现，以下展开介绍。

### 分类处理文件：以压缩文件为例

我们假设需要按照人名对照表进行文件分类，在这一步骤中我们需要用到`pandas`和两个库。

```python
import pandas as pd
import zipfile
import os
```

之后我们读入表格`namelist.csv`，这里面存放了人名与文件号的对应关系。

```python
df_namelist = pd.read_csv('./namelist.csv')
```

如果出现了utf-8不兼容的情况，可以考虑gbk编码：

```python
df_namelist = pd.read_csv('./namelist.csv', encoding='gbk')
```

假设所有的文件都位于`before`文件夹中，我们创建`after`文件夹，并且为分类后的每个类创建子文件夹，在这里我们以人名来分类。我们这里号码是文件名的一部分，因此我们判断号码是否为文件的子串。

```python
before_folder = 'before'
before_files = os.listdir(before_folder)

for file in before_files:
    before_path = os.path.join(before_folder, file)
    for index, row in df_namelist.iterrows():
        number = str(row['号码'])
        if number in file:
            after_folder = os.path.join('after', row['姓名'])
            if not os.path.exists(after_folder):
                os.makedirs(after_folder)
            after_path = os.path.join(after_folder, file)
            the_cmd = f'copy {before_path} {after_path}'
            try:
                os.system(the_cmd)
                print(f"文件 {before_path} 已复制到 {after_path}")
            except IOError as e:
                print(f"复制文件时发生错误: {e}")
```

接下来我们对分类好的文件夹进行压缩。

```python
def compress_folder_to_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
```

以上定义了一个函数，将文件夹`folder_path`中的文件利用`zipfile`库进行压缩，保存到`zip_path`位置，其中`os.walk`将递归地访问路径里面的文件。`os.path.relpath()` 函数用于计算相对路径。它接受两个参数：目标路径和基准路径，并返回从基准路径到目标路径的相对路径。

具体来说，`os.walk(path)`的参数`path`是要遍历的根目录的路径。在每次迭代中，生成器会依次访问根目录下的每个子目录，包括根目录本身。对于每个子目录，生成器会返回当前子目录的路径、子目录下的所有子目录名称列表和子目录下的所有文件名称列表。

例如，假设有以下目录结构：

```
root/
    ├── dir1/
    │     ├── file1.txt
    │     └── file2.txt
    ├── dir2/
    │     ├── dir3/
    │     │     └── file3.txt
    │     └── file4.txt
    └── file5.txt
```

那么，`os.walk("root")`的返回如下：

```python
os.walk("root")
 
"""
第一次迭代将返回：("root", ["dir1", "dir2"], ["file5.txt"])
第二次迭代将返回：("root/dir1", [], ["file1.txt", "file2.txt"])
第三次迭代将返回：("root/dir2", ["dir3"], ["file4.txt"])
第四次迭代将返回：("root/dir2/dir3", [], ["file3.txt"])
"""
```

之后则可以分别对文件夹进行压缩，我们将压缩包都保存到`zip`文件夹下。

```python
for thefolder in os.listdir('after'):
    from_folder = os.path.join('after', thefolder)
    to_zip = os.path.join('zip', f'{thefolder}.zip')
    that_zip = os.path.join('zip-old', f'{thefolder}.zip')
    compress_folder_to_zip(from_folder, to_zip)
    print(f'from {from_folder} to {to_zip} finished')
```

### 脚本SMTP发送邮件

Python中的`smtplib`库可以用于smtp邮件发送。

```python
import pandas as pd
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
```

我们将邮箱数据整理成列表：

```python
df_email = pd.read_csv('./email.csv', encoding='gbk')
the_list = []
for index, row in df_email.iterrows():
    the_list.append([row['姓名'], row['邮箱']])
```

然后我们定义一个发送邮件的类，其构造函数主要传递`config`参数，而通过`generate_email`函数来生成一个邮件的基本信息。

```python
class SendMail:
    def __init__(self,config):
        self.server = self.connect_mailServer(config)
             
    def connect_mailServer(self,config):     
        # 登录并发送邮件
        print('try login')
        try:
            server = smtplib.SMTP(config['stmpServer'], config['stmpPort'])
            server.login(config['fromEmailAddr'], config['password'])
        except smtplib.SMTPException as e:
            print("smtplib 连接服务器报错:", e)
        else:
            print('login success')
            return server
 
    def generate_email(self, subject, text, file, fromEmailAddr, toEmailAddr):
        # 生成email主题、正文、附件信息
        # ---------------------------发送带附件邮件-----------------------------
        # 邮件内容设置
        message = MIMEMultipart()
        # 邮件主题
        message['Subject'] = subject
        # 发送方信息
        message['From'] = fromEmailAddr
        # 接受方信息
        message['To'] = toEmailAddr
        # 邮件正文内容
        message.attach(MIMEText(text, 'plain', 'utf-8'))
        # 添加附件
        with open(file, "rb") as f:
            attach = MIMEApplication(f.read(),_subtype="zip")
            attach.add_header('Content-Disposition','attachment',filename=str(file.split('\\')[-1]))
            message.attach(attach)
        return message
```

这里我们以一个`config`为例说明需要的参数，最重要的即邮箱的客户端密码，这个密码是在邮箱系统中单独生成的用于POP3/SMTP的密码，而非邮箱网站的登陆密码。

```python
config = {
    'stmpServer' : 'smtp.buaa.edu.cn',# 邮件发送服务器地址
    'stmpPort' : 25,# 邮件发送服务器端口：普通为25,QQ邮箱SMTP服务器（端口465或587）
    'fromEmailAddr' : 'xxxxxxxx@buaa.edu.cn', # 邮件发送方邮箱地址
    'password' : 'xxxxxxxxxxxxxxxx', # 邮箱客户端密码
}
```

下面我们则可以根据之前分好的压缩文件打包发送了。

```python
for name, email in the_list:
    subject = 'XXXXXXXX通知'
    text = name + "，您好！XXXXXXXX，谢谢～"
    file = os.path.join('zip', name + ".zip")

    if not os.path.exists(file):
        continue
    
    fromEmailAddr = config['fromEmailAddr']
    toEmailAddr = email
    mail = SendMail(config)
    message = mail.generate_email(subject, text, file, fromEmailAddr, toEmailAddr)

    try:
        mail.server.sendmail(fromEmailAddr, toEmailAddr, message.as_string())
        print('sendmail to ' + name + ' success')
        mail.server.quit()
    except smtplib.SMTPException as e:
        print("smtplib " + name + "发送邮件报错：", e)
    time.sleep(10)
```

注意可能会出现 SMTP错误，如`552：Message Exceeds Maximum Fixed Size`，此时需要查看是否有邮件过大等问题。



### 参考文献

1. 脚本之家的文章，原文链接：https://www.jb51.net/article/229831.htm
1. CSDN发送邮件常见错误及解决办法：https://blog.csdn.net/weixin_42615847/article/details/103579588
1. CSDN博客文章：https://blog.csdn.net/qq_38964360/article/details/136173623

