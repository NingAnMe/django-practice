# blog_tags.py
'''
模板标签
'''
from django import template
from django.db.models.aggregates import Count
from ..models import Post, Category, Tag

register = template.Library()


@register.simple_tag
def get_recent_posts(num=5):
    '''
    最新文章模板标签
    '''
    return Post.objects.all().order_by('-created_time')[:num]


@register.simple_tag
def archives():
    '''
    归档模板标签
    '''
    # month 为精度
    return Post.objects.dates('created_time', 'month', order='DESC')


@register.simple_tag
def get_categories():
    '''
    分类标签模板
    '''
    # __gt 表示大于， __lte表示小于
    # Count 为标签添加一个计数的属性
    categories = Category.objects.annotate(
        num_posts=Count('post')).filter(num_posts__gt=0)
    return categories


@register.simple_tag
def get_tags():
    '''
    标签标签模板
    '''
    tags = Tag.objects.annotate(num_posts=Count('post')).filter(num_posts__gt=0)
    return tags
