import kivy.app
import kivy.uix.boxlayout
import kivy.uix.image
import PIL.Image
import numpy
import numpycnn
import threading
import kivy.clock

class AppGUI(kivy.uix.boxlayout.BoxLayout):
    layer_num = 1
    def run_cnn_thread(self):
        print(self.layer_num%3)
        print("Layer num : ", self.layer_num)
        self.curr_img = self.cnn_example(numpy_img=self.curr_img, layer=self.layer_num%3)
        self.refresh_GUI(layer=self.layer_num%3)
        self.ids.lbl_details.text = "Shape of output="+str(self.curr_img.shape)
        self.layer_num = self.layer_num + 1
        print("Output shape : ", self.curr_img.shape)

    def start_cnn(self):
        img1 = self.ids.img1#Original Image
        im = PIL.Image.open(img1.source).convert("L")
        img_arr = numpy.asarray(im, dtype=numpy.uint8)
        if self.layer_num%3 == 1:
            img1 = self.ids.img1#Original Image
            im = PIL.Image.open(img1.source).convert("L")
            img_arr = numpy.asarray(im, dtype=numpy.uint8)
            self.curr_img = img_arr
            threading.Thread(target=self.run_cnn_thread).start()
            self.ids.btn.text = "Wait."
        elif self.layer_num%3 == 2:
            threading.Thread(target=self.run_cnn_thread).start()
            self.ids.btn.text = "Wait."
        elif self.layer_num%3 == 0:
            threading.Thread(target=self.run_cnn_thread).start()
            self.ids.btn.text = "Wait."

    def cnn_example(self, numpy_img, layer=1):
        if layer == 1:
            #**Working with conv layer 1**
            l1_filter = numpy.zeros((2,3,3))
            l1_filter[0, :, :] = numpy.array([[[-1, 0, 1], 
                                               [-1, 0, 1], 
                                               [-1, 0, 1]]])
            l1_filter[1, :, :] = numpy.array([[[1,   1,  1], 
                                               [0,   0,  0], 
                                               [-1, -1, -1]]])
            
            l1_feature_map = numpycnn.conv(numpy_img, l1_filter)
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map[:, :, 0]))
            im.save("conv1_filter1.png")
    
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map[:, :, 1]))
            im.save("conv1_filter2.png")
    
            l1_feature_map_relu = numpycnn.relu(l1_feature_map)
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map[:, :, 0]))
            im.save("conv1_relu1.png")
    
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map[:, :, 1]))
            im.save("conv1_relu2.png")
    
            l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map_relu_pool[:, :, 0]))
            im.save("conv1_relu_pool1.png")
    
            im = PIL.Image.fromarray(numpy.uint8(l1_feature_map_relu_pool[:, :, 1]))
            im.save("conv1_relu_pool2.png")

            return l1_feature_map_relu_pool

        elif layer == 2:
            #**Working with conv layer 2**
            l2_filter = numpy.random.rand(3, 5, 5, numpy_img.shape[-1])
            l2_feature_map = numpycnn.conv(numpy_img, l2_filter)
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map[:, :, 0]))
            im.save("conv2_filter1.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map[:, :, 1]))
            im.save("conv2_filter2.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map[:, :, 2]))
            im.save("conv2_filter3.png")

            l2_feature_map_relu = numpycnn.relu(l2_feature_map)
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu[:, :, 0]))
            im.save("conv2_relu1.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu[:, :, 1]))
            im.save("conv2_relu2.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu[:, :, 2]))
            im.save("conv2_relu3.png")

            l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu, 2, 2)
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu_pool[:, :, 0]))
            im.save("conv2_relu_pool1.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu_pool[:, :, 1]))
            im.save("conv2_relu_pool2.png")
            im = PIL.Image.fromarray(numpy.uint8(l2_feature_map_relu_pool[:, :, 2]))
            im.save("conv2_relu_pool3.png")
            return l2_feature_map_relu_pool

        elif layer == 0:#0 corresponds to layer 3
            #**Working with conv layer 3**
            l3_filter = numpy.random.rand(1, 7, 7, numpy_img.shape[-1])
            l3_feature_map = numpycnn.conv(numpy_img, l3_filter)
            im = PIL.Image.fromarray(numpy.uint8(l3_feature_map[:, :, 0]))
            im.save("conv3_filter1.png")
            l3_feature_map_relu = numpycnn.relu(l3_feature_map)
            im = PIL.Image.fromarray(numpy.uint8(l3_feature_map_relu[:, :, 0]))
            im.save("conv3_relu1.png")
            l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu, 2, 2)
            im = PIL.Image.fromarray(numpy.uint8(l3_feature_map_relu_pool[:, :, 0]))
            im.save("conv3_relu_pool1.png")

            return l3_feature_map_relu_pool
    
    @kivy.clock.mainthread
    def refresh_GUI(self, layer=1):
        img1 = self.ids.img1
        img2 = self.ids.img2
        img3 = self.ids.img3
        img4 = self.ids.img4
        img5 = self.ids.img5
        img6 = self.ids.img6
        img7 = self.ids.img7
        img8 = self.ids.img8
        img9 = self.ids.img9

        lbl1 = self.ids.lbl1
        lbl2 = self.ids.lbl2
        lbl3 = self.ids.lbl3
        lbl4 = self.ids.lbl4
        lbl5 = self.ids.lbl5
        lbl6 = self.ids.lbl6
        lbl7 = self.ids.lbl7
        lbl8 = self.ids.lbl8
        lbl9 = self.ids.lbl9

        if layer == 1:
            lbl2.text = "L1Map1"
            img2.source = "conv1_filter1.png"
            img2.reload()
    
            lbl3.text = "L1Map2"
            img3.source = "conv1_filter2.png"
            img3.reload()
    
            lbl4.text = "L1Map1ReLU"
            img4.source = "conv1_relu1.png"
            img4.reload()
    
            lbl5.text = "L1Map2ReLU"
            img5.source = "conv1_relu2.png"
            img5.reload()    

            lbl6.text = "L1Map1ReLUPool"
            img6.source = "conv1_relu_pool1.png"
            img6.reload()
    
            lbl7.text = "L1Map2ReLUPool"
            img7.source = "conv1_relu_pool2.png"
            img7.reload()

            self.ids.btn.text = "Run Second Conv. Layer"

        elif layer == 2:
            lbl1.text = "L2Map1"
            img1.source = "conv2_filter1.png"
            img1.reload()

            lbl2.text = "L2Map2"
            img2.source = "conv2_filter2.png"
            img2.reload()

            lbl3.text = "L2Map3"
            img3.source = "conv2_filter3.png"
            img3.reload()

            lbl4.text = "L2Map1ReLU"
            img4.source = "conv2_relu1.png"
            img4.reload()

            lbl5.text = "L2Map3ReLU"
            img5.source = "conv2_relu2.png"
            img5.reload()

            lbl6.text = "L2Map3ReLU"
            img6.source = "conv2_relu3.png"
            img6.reload()

            lbl7.text = "L2Map1ReLU"
            img7.source = "conv2_relu_pool1.png"
            img7.reload()

            lbl8.text = "L2Map2ReLU"
            img8.source = "conv2_relu_pool2.png"
            img8.reload()

            lbl9.text = "L2Map3ReLU"
            img9.source = "conv2_relu_pool3.png"
            img9.reload()

            self.ids.btn.text = "Run Third Conv. Layer"

        elif layer == 0:#0 corresponds to layer 3
            lbl1.text = "Original"
            img1.source = "input_image.jpg"
            img1.reload()

            #**Working with conv layer 3**
            lbl2.text = "L3Map1"
            img2.source = "conv3_filter1.png"
            img2.reload()

            lbl3.text = "L3Map1ReLU"
            img3.source = "conv3_relu1.png"
            img3.reload()
            lbl4.text = "L3Map1ReLUPool"
            img4.source = "conv3_relu_pool1.png"
            img4.reload()

            img5.source = ""
            img5.reload()
            img6.source = ""
            img6.reload()
            img7.source = ""
            img7.reload()
            img8.source = ""
            img8.reload()
            img9.source = ""
            img9.reload()

            lbl5.text = ""
            lbl6.text = ""
            lbl7.text = ""
            lbl8.text = ""
            lbl9.text = ""

            self.ids.btn.text = "Repeat Again. Run First Conv. Layer"

class NumPyCNNApp(kivy.app.App):
    pass

if __name__ == "__main__":
    NumPyCNNApp().run()
