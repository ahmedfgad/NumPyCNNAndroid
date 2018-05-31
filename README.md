# NumPyCNNAndroid

This project builds Convolutional Neural Network (CNN) for Android using Kivy and NumPy.

It is based on a previous project called **NumPyCNN (https://github.com/ahmedfgad/NumPyCNN)** but it is now working on Android.
The beaty of Kivy is that it not only allows Python code to work on different platforms (Android is one of them), but also to run the code without changes, as long as all requests are already supported by **python-for-android**.

The major changes done in the NumPyCNN to create its Android version **NumPyCNN** is using supported modules by python-for-android to do the task. In **NumPyCNN**, the unsupprted used modules are **skimage**, **Matplotlib**, and **sys**. The one used while being supported is NumPy.
In **NumPyCNNAndroid**, **python image library (PIL)** is used to do the work by skimage. The GUI inside **Kivy** sufficient to do the job and thus no need to use Matplotlib. Some code changes applied to avoid using **sys**.

Inside the project, the important files are as follows:
1. **main.py** which is the entry point for the application.
2. **numpycnn.kv** that holds the UI design.
3. **numpycnn.py** which is taken from NumPyCNN project to implement the CNN.
4. **buildozer.spec** holding the specifications of the app such as requiremenets, SDK path, NDK path, python-for-android path, title, package name, welcome screen, icon, and other important things that are criticial to successful build of the app.

To build the project yourself, it is recommended to follow these steps:
1. Understanding the NumPyCNN project.
2. Installing Kivy.
3. Installing Buildozer and python-for-android.

For **description about the NumPyCNN project**, refer to this article titled **"Building Convolutional Neural Network using NumPy from Scratch"**:  
https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad/  
https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html  
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741  

For instructions about **installing Kivy and python-for-android**, read this article titled **"Python for Android: Start Building Kivy Cross-Platform Applications"**:  
https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad

The project has the **Android Pckage Kit (APK)** file inside the **/NumPyCNNAndroid/bin/** directory for installation.   
Once installed, the application will appear in the applications list as in the following figure:

![screenshot_2018-05-31-14-11-03](https://user-images.githubusercontent.com/16560492/40783856-09969998-64e4-11e8-9746-f9ec13c1b850.png)

Just open the application and wait until loading. Once opened, it will have the following layout in which the original image is shown at the top-left corner. There is also a label giving brief description about the the three layers and their final output size (output of pooling layer).

![screenshot_2018-05-31-14-11-18](https://user-images.githubusercontent.com/16560492/40783916-3ccbf8b2-64e4-11e8-8de2-a9fa18b3232e.png)

According to the NumPyCNN project and its article, the CNN example created has the following architecture:
1. Conv layer with 2 3x3 filters.
2. ReLU layer.
3. Pooling layer.
4. Conv layer with 3 5x5 filters.
5. ReLU layer.
6. Pooling layer.
7. Conv layer with 1 7x7 filter.
8. ReLU layer.
9. Pooling layer.

Actually the user can not make changes to the app such as using different image, adding, removing, modifiying a layer.   
The application is designed to work on each three successive conv-relu-pool layers, show their outputs, return so that the user can execute the next three layers by clikcing a button at the bottom of the screen. The previous result before clicking the button will be used for further processing. 

The result of applying the first conv-relu-pool layers after pressing the button is shown below. The 2 filters used are for detecting horizontal and vertical edges. The filters used in the remaining two conv layers are randomly generated.

![screenshot_2018-05-31-14-11-27](https://user-images.githubusercontent.com/16560492/40784228-54e4c55e-64e5-11e8-8c88-76e535cb1f7f.png)

Pressing the button again will make the app go to the next conv-relu-pool layers and show their outputs as in the figure below:

![screenshot_2018-05-31-14-11-48](https://user-images.githubusercontent.com/16560492/40784262-6f8d086c-64e5-11e8-8b18-0fcdbc33a43f.png)

Finally, the last conv-relu-pool layers are executed after pressing the button again and their results are shown as below:

![screenshot_2018-05-31-14-11-55](https://user-images.githubusercontent.com/16560492/40784373-ca6d221c-64e5-11e8-9877-d73747834175.png)

For more info.: KDnuggets: https://www.kdnuggets.com/author/ahmed-gad  
LinkedIn: https://www.linkedin.com/in/ahmedfgad  
Facebook: https://www.facebook.com/ahmed.f.gadd  
ahmed.f.gad@gmail.com  
ahmed.fawzy@ci.menofia.edu.eg
