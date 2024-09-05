#pragma once

#include <GLFW/glfw3.h>
#include <string>

class Window {
    private:

        void initWindow();
        
        const int width;
        const int height;
        std::string windowName;
        GLFWwindow *window;



    public:
        Window(int w, int h, std::string name);
        ~Window();
        bool shouldClose() {return glfwWindowShouldClose(window);};

};