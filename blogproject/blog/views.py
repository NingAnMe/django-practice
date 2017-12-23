import markdown

from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView
from comments.forms import CommentForm
from .models import Post, Category


class IndexView(ListView):
    '''
    主页
    '''
    model = Post
    template_name = 'blog/index.html'
    context_object_name = 'post_list'


def detail(request, pk):
    '''
    文章详情页
    '''
    post = get_object_or_404(Post, pk=pk)

    # 阅读量加 1
    post.increase_views()

    # Markdown 格式解析
    post.body = markdown.markdown(post.body,
                                  extensions=['markdown.extensions.extra',
                                              'markdown.extensions.codehilite',
                                              'markdown.extensions.toc', ],
                                  )

    form = CommentForm()
    comment_list = post.comment_set.all()

    context = {'post': post,
               'form': form,
               'comment_list': comment_list,
               }

    return render(request, 'blog/detail.html', context=context)


class ArchivesView(IndexView):
    '''
    文章归档页
    '''

    def get_queryset(self):
        year = self.kwargs.get('year')
        month = self.kwargs.get('month')
        return super(ArchivesView, self).get_queryset().filter(
            created_time__year=year,
            created_time__month=month,).order_by('-created_time')


class CategoryView(IndexView):
    '''
    文章分类页
    '''

    def get_queryset(self):
        cate = get_object_or_404(Category, pk=self.kwargs.get('pk'))
        return super(CategoryView, self).get_queryset().filter(category=cate)
