from django.conf.urls import url

from . import views

app_name = 'blog'
urlpatterns = [
    # 主页
    url(r'^$', views.index, name='index'),

    # 文章内容页
    url(r'^post/(?P<pk>[0-9]+)/$', views.detail, name='detail'),

    # 文章日期归档页
    url(r'^archives/(?P<year>[0-9]{4})/(?P<month>[0-9]{1,2})/$',
        views.archives, name='archives'),
        
    # 文章分类页
    url(r'^category/(?P<pk>[0-9]+)/$', views.category, name='category'),
]
