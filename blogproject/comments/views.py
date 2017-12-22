from django.shortcuts import render, get_object_or_404, redirect
from blog.models import Post

from .models import Comment
from .forms import CommentForm


def post_comment(request, post_pk):
    post = get_object_or_404(Post, pk=post_pk)

    # 判断是否 POST 请求
    if request.method == 'POST':
        form = CommentForm(request.POST)

        if form.is_valid():
            # commit=False 不提交 comment，仅仅生成一个 Comment 模型实例
            comment = form.save(commit=False)
            comment.post = post
            comment.save()
            return redirect(post)
        else:
            # 反向查询全部评论，相当于 Comment.objects.filter(post=post)
            comment_list = post.comment_set.all()
            context = {
                'post': post,
                'form': form,
                'comment_list': comment_list,
            }
            return render(request, 'blog/detail.html', context=context)

    return redirect(post)
