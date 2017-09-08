from Tkinter import *
import Tkinter.messagebox as messagebox

class Application(Frame):
    """docstring for Application"""
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.helloLabel = Label(self, text = 'Hello, world!')
        self.helloLabel.pack()
        self.quitButton = Button(self, text = 'Quit', command = self.quit)
        self.quitButton.pack()
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alb = Button(self, text = 'Hello', command = self.hello)
        self.alb.pack()

    def hello(self):
        name = self.nameInput.get() or 'world'
        messagebox.showinfo('Message', 'Hello %s' % name)

app = Application()
app.master.title('Hello World!')
app.mainloop()