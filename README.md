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
```
