# ai_tool
ai tool



# slice picture

```python
from ai_tools.img_slide import yield_sub_img
# yield the sub image from the jpg
for bbox, sub_img in yield_sub_img("test.jpg", 0, 0, 180, 60):
    clip = "-".join([str(x) for x in bbox])
    print("sub img:{}".format(clip))
    cv2.imshow(clip, sub_img)
    cv2.waitKey(0)
```



# IoU

compute the iou for tow boxes,

example box1 1, 2, 101, 102.  location(1,2)  is  left-up, location(101,102) is right-down.



```python
from ai_tool.bbox import BBox
bbox1 = BBox([1, 2, 101, 102])
bbox2 = BBox([11, 12, 121, 122])
iou = bbox1 / bbox2
print("iou", iou)
assert iou > 0.5

print('box1 S is', bbox1.S)
print('box1 & box2', bbox1 & bbox2)
print('box1 == box2', bbox1 == bbox2)
```



result is :

> iou 0.5785714285714286
> box1 S is 10000
> box1 & box2 [11, 12, 101, 102]
> box1 == box2 True



# multi bbox operation

```python
from ai_tool.bbox import BBoxes, BBox
bb1 = BBoxes(iou_thresh=0.6)
bb2 = BBoxes()

bb1.append([1,2, 101, 102])
bb1.append([1000, 2, 1101, 102])

bb2.append([11, 12, 111, 112])
bb2.append([1, 1002, 101, 1102])

# judge the bbox in bb1
print("[5, 5, 100, 100] in bb1", BBox([5, 5, 100, 100]) in bb1)
print("[100, 5, 200, 100] in bb1", BBox([100, 5, 200, 100]) in bb1)

# bb1 & bb2
print("bb1 & bb2", bb1 & bb2)
print("bb1 - bb2", bb1 - bb2)
print("bb2 - bb1", bb2 - bb1)
```



result is

> [5, 5, 100, 100] in bb1 True
> [100, 5, 200, 100] in bb1 False
> bb1 & bb2 [[1, 2, 101, 102]]
> bb1 - bb2 [[1000, 2, 1101, 102]]
> bb2 - bb1 [[1, 1002, 101, 1102]]