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

# 关于归档过滤不到月份的问题的问题
USE_TZ 用来指定是否使用指定的时区(TIME_ZONE)的时间. 若为 True, 则Django 会使用内建的时区的时间 否则, Django 将会使用本地的时间

如果 settings.py 设置 USE_TZ=True 的话，可以使用系统的时区数据来填充 mysq 时区表,linux 下命令如下:
```shell
shell> mysql_tzinfo_to_sql /usr/share/zoneinfo | mysql -u root -p mysql
```

因为进入django命令行执行
```python
>> print(Post.objects.filter(created_time__month=12).query)
```
可看到sql语句用来时区转换函数’CONVERT_TZ‘，然而mysql由于无法将’UTC‘，’Asia/Shanghai'等转化为具体时差，从而使得sql语句执行结果返回null

# 启动 nginx 没有反应
查看本机的 localhost 地址，查看是否 nginx 已经启动了
