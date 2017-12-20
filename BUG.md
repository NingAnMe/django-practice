# 修改中文和时区
settiongs.py
```python
## 其它配置代码...

# 把英文改为中文
LANGUAGE_CODE = 'zh-hans'

# 把国际时区改为中国时区
TIME_ZONE = 'Asia/Shanghai'

## 其它配置代码...
```

# python3+django+mysql 的驱动问题

在 __init__.py 中加入
```python
import pymysql
pymysql.install_as_MySQLdb()
```

# 设置 MySQL 数据库的编码

找到 /etc/mysql/my.cnf,添加如下内容
```
[client]
default-character-set = utf8
[mysqld]
character-set-server = utf8
[mysql]
default-character-set = utf8
```

# 重新定义 django models 模型的 save 方法

```python
from django.db import models

class Blog(models.Model):
    name = models.CharField(max_length=100)
    tagline = models.TextField()

    def save(self, *args, **kwargs):
        if self.name == "Yoko Ono's blog":
            return # Yoko shall never have her own blog!
        else:
            super(Blog, self).save(*args, **kwargs) # Call the "real" save() method.
            # 如果 super 不加参数,调用的是父类 models.Model 的 save 方法
            super().save(*args, **kwargs)
```
