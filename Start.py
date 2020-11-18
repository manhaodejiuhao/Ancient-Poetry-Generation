#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import filedialog
import tkinter.messagebox
import webbrowser as web
from PIL import Image,ImageTk
import cifar100vgg as cfvg
import os
import train as tr


def main():
    
    def selectImafile():
        global sfname 
        sfname = filedialog.askopenfilename(title='选择图片', filetypes=[('JPG', '*.jpg'), ('PNG', '*.png'),('JPEG','*.jpeg')])
        text1.insert(INSERT,sfname)
        if sfname!=None:
                path=sfname
                outfile=cfvg.resize_image(path,400,200)
                load = Image.open(outfile)
                render = ImageTk.PhotoImage(load)
                img = tkinter.Label(image=render)
                img.image = render
                img.place(x=20, y=140)
                os.remove(outfile)


    def closeThisWindow():
        root.destroy()
        
    
    def choosepic(path_):
        #path.set(path_)
        img_open = Image.open(path_)
        img=ImageTk.PhotoImage(img_open)
        l1=Label(root)
        l1.config(image=img)
        l1.image=img
 

    def change_poem(j,poem_list):        
        j[0]+=1
        j[0]=j[0]%len(poem_list)
        po.set(poem_list[j[0]])
        
    def print_poem():
        try:
            path=sfname
            print(path)
            '''label=cfvg.pic_to_label(path)
            key_words,poem_list=tr.label_poem(label)'''
            poem_list = [['不知身外事,\n想见古来今。\n秃发寻无迹,\n头空一点尘。\n']]
            tmp_poem_list=[]
            for i in range(len(poem_list)):
                p = list(poem_list[i])
                p.insert(6, '\n')
                p.insert(14, '\n')
                p.insert(22, '\n')
                poems = ''.join(p)
                tmp_poem_list.append(poems)
            j=[0]
            change_poem(j,tmp_poem_list)
            #poem=tkinter.Message(root,bg="#B0E0E6",text=poem_list[0], 
            #                    width=200,font=('华文新魏',17))
            button4=Button(root,text='下一首',width=10,command=lambda:change_poem(j,tmp_poem_list),font=('华文楷体',11),bg="#A9A9A9")
            button4.place(x=670,y=360)

        except NameError:
            tkinter.messagebox.showinfo('提示','请先选择一张图片！')
            

    def open_link():
        web.open('https://www.boater.cn/') 
        
    #初始化
    root=Tk()
    root["background"]="#B0E0E6"
    
    #设置窗体标题
    root.title('看图作诗 ver0.10')

    #设置窗口大小和位置
    root.geometry('800x400+570+200')


    label1=Label(root,font=('华文楷体',14),text='请选择图片:',bg="#B0E0E6")
    text1=Entry(root,bg='white',width=45,font=('微软雅黑',10))
    button1=Button(root,text='浏览',width=8,command=selectImafile,font=('华文楷体',12),bg="#D2B48C")
    button2=Button(root,text='作诗',width=8,command=print_poem,font=('华文楷体',12),bg="#FFDAB9")
    button3=Button(root,text='退出程序',width=8,command=closeThisWindow,font=('华文楷体',12),bg="#A9A9A9")
    link=Button(root,text='访问主页',bg='#B0C4DE',width=8,command=open_link,font=('楷体',10))
    
    po=StringVar()
    po.set('')
    poem=tkinter.Message(root,bg="#B0E0E6",textvariable=po, width=200,font=('华文新魏',17))


    

    label1.place(x=90,y=30)
    text1.place(x=200,y=33)
    button1.place(x=580,y=25)
    button2.place(x=250,y=80)
    button3.place(x=430,y=80)
    link.place(x=670,y=360)
    poem.place(x=500,y=150)
    root.attributes("-toolwindow", 1)
    root.mainloop() 

if __name__=="__main__":
    main()


# In[5]:


#pip install networkx


# In[ ]:




