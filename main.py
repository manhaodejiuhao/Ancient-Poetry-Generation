from parameter import *
from wordvec import *
import data
import rnn_model
import train
import cifar100vgg as cfvg
import estimate as es
import play
from keras.backend import clear_session
 
from PIL import Image, ImageTk   
import tkinter as tk
from tkinter import filedialog

# 销毁当前TF图并创建一个新的TF图，避免旧模型/图层的混乱。
clear_session()

# 记录在运行程序时后面的字符串，默认是default
def defineArgs():
    parser = argparse.ArgumentParser(description="Chinese_poem_generator.")
    parser.add_argument("-m", "--mode", help="select mode by 'train' or test or head",
                        choices=["train", "test", "head"], default="test")
    return parser.parse_args()

if __name__ == "__main__":
    args = defineArgs()
    # 初始化数据的对象
    traindata = data.poems_data(isEvaluate=False)
    # 进行训练或者生成
    if args.mode == "train":
        train.train(traindata)
    else:
        '''l = list('春华秋实')
        for i in range(len(l)):
            l[i] = poem_Data.word_ID[l[i]]
        for j in range(4):
            res = train.generate(poem_Data, l)
            print(res)
        r = 0'''
        
        window = tk.Tk()                 #创建窗口
        window.title('基于图片生成古诗')
        window.geometry('1000x1000')     #窗口大小
        global IMG                       #全局变量
        global file_path
        
        def Open_Img():  
            global IMG
            global file_path
            OpenFile=tk.Tk()
            OpenFile.withdraw()
            file_path=filedialog.askopenfilename()             #图片路径
            IMG=ImageTk.PhotoImage(Image.open(file_path))
            label=tk.Label(window, image=IMG).place(x=20, y=100)  #以label显示图片
        button=tk.Button(window,text='导入图片',width=15, height=2,command=Open_Img)
        button.place(x=160, y=20)
        
        def Run():           # 作诗结果输出
            global file_path            
            Result=play.poem(file_path)   # 输出结果存储为Result
            print(Result)                 
            label_text = tk.Label(window, text=Result)     # 将Result显示在tkinter界面
            label_text.place(x=560, y=100)
        button=tk.Button(window, text='生成古诗',width=15, height=2,command=Run).place(x=560, y=20)
               
        window.mainloop()  # 运行整体窗口
