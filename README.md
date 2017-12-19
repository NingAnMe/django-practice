# python3+django+mysql 的驱动问题

在 __init__.py 中加入
```python
import pymysql
pymysql.install_as_MySQLdb()
```
