import tkinter
import customtkinter
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter
import customtkinter

test_image = ''

prototxt = "dependencies/colorization_deploy_v2.prototxt"
caffe_model = "dependencies/colorization_release_v2.caffemodel"
pts_npy = "dependencies/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
pts = np.load(pts_npy)

layer1 = net.getLayerId("class8_ab")
layer2 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(layer1).blobs = [pts.astype("float32")]
net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colourise(img):
    # Read image from the path
    test_image = cv2.imread(img)
    # Convert image into gray scale
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # Convert image from gray scale to RGB format
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

    # Normalizing the image
    normalized= test_image.astype("float32") / 255.0
    # Converting the image into LAB
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    # Resizing the image
    resized = cv2.resize(lab_image, (224, 224))
    # Extracting the value of L for LAB image
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    # Finding the values of 'a' and 'b'
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # Resizing
    ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))
    L = cv2.split(lab_image)[0]
    # Combining L,a,b
    LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Converting LAB image to RGB_colored
    RGB_colored = cv2.cvtColor(LAB_colored,cv2.COLOR_LAB2RGB)
    RGB_colored = np.clip(RGB_colored, 0, 1)
    # Changing the pixel intensity back to [0,255]
    RGB_colored = (255 * RGB_colored).astype("uint8")

    # Converting RGB to BGR
    RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.jpg", RGB_BGR)


image_test = ""

class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        self.test_image = ""
        self.title("Colouriser")
        self.geometry("-100-100")
        self.resizable(width=True, height=True)

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)
        self.navigation_frame.grid_columnconfigure(2, weight=1)


        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="B&W to Colour", font=customtkinter.CTkFont(size=24, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=10)

        self.button_1 = customtkinter.CTkButton(self.navigation_frame, text="Select Image", command=self.openImg)
        self.button_1.grid(row=1, column=0, padx=20, pady=20, sticky="s")

        self.button_2 = customtkinter.CTkButton(self.navigation_frame, text="Run", command=self.buttonPress)
        self.button_2.grid(row=2, column=0, padx=20, pady=20, sticky="s")

        self.navigation_frame_theme = customtkinter.CTkLabel(self.navigation_frame, text="Theme", font=customtkinter.CTkFont(size=10, weight="bold"))
        self.navigation_frame_theme.grid(row=3, column=0, padx=20, pady=50, sticky="s")
        
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=3, column=0, padx=20, pady=20, sticky="s")

        if self.test_image == "":
            self.test_image = "placeholder.jpg"
            
        imgSize = Image.open(self.test_image)
        width, height = imgSize.size
        if width > 600:
            width = width/2
            height = height/2
        self.image_1 = customtkinter.CTkImage(dark_image=Image.open(self.test_image), size=(width,height))
        self.label_1 = customtkinter.CTkLabel(self.navigation_frame, text="", image=self.image_1, compound="right", width=0)
        self.label_1.grid(rowspan=3, column=0, padx=20, pady=20, sticky="nsew")

    def updateImg(self):
        if isinstance(self.test_image,str):
            newImgSize = Image.open(self.test_image)
            updatedImage = self.test_image
        else:
            newImgSize = Image.open(self.test_image.name)
            updatedImage = self.test_image.name
        width, height = newImgSize.size
        if width > 600:
            width = width/2
            height = height/2
        self.image_2 = customtkinter.CTkImage(dark_image=Image.open(updatedImage), size=(width,height))
        self.label_1.configure(image=self.image_2)

    def buttonPress(self):
        print("OUTPUT: ", self.test_image.name)
        colourise(self.test_image.name)
        self.test_image = "result.jpg"
        self.updateImg()

    def openImg(self):
        self.test_image = customtkinter.filedialog.askopenfile()
        self.updateImg()

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()