import easygui as g
import os

# 获得打开文件路径
file_path = g.fileopenbox(default='*.txt')

with open(file_path) as old_file:
    # 获得文件名
    title = os.path.basename(file_path)
    msg = '文件【%s】的内容如下：' % title
    text = old_file.read()
    # 显示窗体，传入文件头、文件名、文件内容，最后返回文件内容
    text1 = g.textbox(msg, title, text)

print(text1, text)

# 模拟文件被修改
if text1 != text[:-1]:
    msg = '检测到文件内容发生改变，请选择以下操作：'
    tilte = '警告'
    # 选择框
    choise = g.buttonbox(msg, title, choices=('保存', '放弃保存', '另保存...'))
    if choise == '保存':
        with open(file_path, 'w') as old_file:
            old_file.write(text1[:-1])
    if choise == '放弃保存':
        pass
    if choise == '另保存...':
        another_path = g.filesavebox(default='*.txt')
        if os.path.splitext(another_path)[1] != '.txt':
            another_path += '.txt'
        with open(another_path, 'w') as new_file:
            new_file.write(text1[:-1])
