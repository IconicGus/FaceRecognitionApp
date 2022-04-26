from tkinter import *
from tkinter.tix import InputOnly
from matplotlib.pyplot import text

def printarTexto():
    a = entry1.get()
    texto_output["text"] = a

janela = Tk()
janela.title("APP de Reconhecimento Facial")
#janela.geometry("400x400")

texto_titulo = Label(janela, text="Insira o ID da classe:") #criar um label (qual janela fica esse label, o texto do label)
texto_titulo.grid(column=0, row=0, padx=10, pady=10) #em que posição o elemento vai ficar na janela

entry1 = Entry(janela)
entry1.grid(column=0, row=1, padx=10, pady=10)

botao = Button(janela, text="INICIAR", command=printarTexto)
botao.grid(column=0, row=2, padx=10, pady=10)

texto_output = Label(janela, text="")
texto_output.grid(column=0, row=3, padx=10, pady=10)

janela.mainloop()