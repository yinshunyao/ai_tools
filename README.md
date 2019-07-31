# ai_tools
ai相关的一些工具类方法


# 切图
`yield_sub_img 提供yield的方式生成切片之后的子图

```python
from img_slide import yield_sub_img
# 使用切图功能
for bbox, sub_img in yield_sub_img(self.jpg, 0, 0, 180, 60):
    clip = "-".join([str(x) for x in bbox])
    print("分片:{}".format(clip))
    cv2.imshow(clip, sub_img)
    cv2.waitKey(0)
```

