ALL_CLASSES = [
    '_background_',
    'tooth',
    'bone',
    'crown'
]

## 標記的 masks 的值 
LABEL_COLORS_LIST = [
    (0,0,0),
    (128,0,0), # tooth 紅色
    (0,128,0), # bone 綠色
    (128,0,0) # crown 藍色
]

## 預測時要顯示的顏色 
VIS_LABEL_MAP = [
    (0,0,0),
    (128,0,0), # tooth 紅色
    (0,128,0), # bone 綠色
    (128,0,0) # crown 藍色
]