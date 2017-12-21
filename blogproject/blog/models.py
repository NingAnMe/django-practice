from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Category(models.Model):
    '''
    分类
    '''
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name


class Tag(models.Model):
    '''
    标签
    '''
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name


class Post(models.Model):
    '''
    文章
    '''
    # 文章标题
    title = models.CharField(max_length=100)

    # 文章正文
    body = models.TextField()

    # 创建时间
    created_time = models.DateTimeField()

    # 最后一次修改时间
    modified_time = models.DateTimeField()

    # 文章摘要
    excerpt = models.CharField(max_length=200, blank=True)

    # 文章分类
    category = models.ForeignKey(Category)

    # 文章标签
    tags = models.ManyToManyField(Tag, blank=True)

    # 文章作者
    author = models.ForeignKey(User)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('blog:detail', kwargs={'pk': self.pk})
